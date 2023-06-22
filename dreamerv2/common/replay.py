import collections
import datetime
import io
import pathlib
import uuid

import numpy as np
import tensorflow as tf
import random
import pickle
import time
import csv
import os

from starr import SumTreeArray

class SumTree:
  def __init__(self, capacity):
    self.capacity = int(capacity)
    print(f'capacity: {self.capacity}')
    self.array = SumTreeArray(self.capacity, dtype='float64')
    self.write = 0  # write head.

  def total(self):
    return self.array.sum()

  def add(self, vals):
    """Insert vals at write head, and increment write head."""
    idxs = np.mod(np.arange(self.write, self.write + len(vals)), len(self.array))
    self.array[idxs] = vals
    self.write = np.mod(self.write + len(vals), len(self.array))
    return idxs

  def update(self, idxs, vals):
    self.array[idxs] = vals

  def sample(self, n):
    idxs = self.array.sample(n)
    return idxs

  def __len__(self):
    return len(self.array)


class Priority:
  def __init__(self, capacity, params):
    """
    Prioritized replay buffer, using SumTreeArray for speedup.

    Args:
      capacity: int. size of the SumTreeArray.
      params:
        e: epsilon value to ensure all data has at least some chance of being selected.
        a: alpha value. higher means sharper prioritization, lower means broader prioritization.
        maxval: default value for data added to the buffer, to ensure that they have high priority.
        metric: which metric is used for prioritization.
                Options for metric are: 'losses' (full world model loss), 'kl' (world model kl),
                                         'task_reward', 'task_intr_reward', 'task_actor_loss', 'task_critic_loss',
                                         'expl_reward', 'expl_intr_reward', 'expl_actor_loss', 'expl_critic_loss'
        running_min: bool. If True, then subtract running_min (across the session) from metric before
                           using it to compute prioritization.
        b: how much of previous priority value to blend in. priority = new_priority*(1-b) + b*old_priority
    """
    self.e = params.e
    self.a = params.a
    self.maxval = params.maxval
    self.metric = params.metric
    self.running_min = params.running_min
    self.b = float(params.b)
    self.temporal_method = params.temporal
    self.c = float(params.c)
    print(f'Initializing priority buffer with params: {params}')

    self.tree = SumTree(capacity)
    self.idx_from_ep = {} # key - 'episode:step', val - index into tree (yielding priority value)
    self.ep_from_idx = {} # val - 'episode:step', key - index into tree (yielding priority value)

    self.counter = np.zeros(int(capacity))  # Number of visits (updates) to each epstep

  def _get_priority(self, vals):
    return (np.abs(vals) + self.e) ** self.a

  def sample(self, n):
    ### See https://github.com/rlcode/per/blob/44b42eedbcc452fbe8be221f6597955a706c20cf/prioritized_memory.py#L22
    idxs = self.tree.sample(n)
    batch = [self.ep_from_idx[i] for i in idxs]

    return batch, idxs

  def add(self, vals, keys, is_maxval=False):
    if is_maxval:
      p = vals
    else:
      p = self._get_priority(vals)
    idxs = self.tree.add(p)

    for i in range(len(keys)):
      overwritten_ep = self.ep_from_idx.pop(idxs[i], -1)
      if overwritten_ep != -1:
        self.idx_from_ep.pop(overwritten_ep, -1)
        self.counter[idxs[i]] = 0 # Reset.

      self.idx_from_ep[keys[i]] = idxs[i]
      self.ep_from_idx[idxs[i]] = keys[i]

    return idxs

  def update(self, idxs, vals, increment_counter=True):
    p = self._get_priority(vals)

    if self.b > 0:
      old_p = self.tree.array[idxs]
      if self.temporal_method == 'radd':
        p = (1-self.b)*p + self.b*old_p
      elif self.temporal_method == 'rmult':
        p = np.power(p, (1 - self.b)) * np.power(old_p, self.b)
      elif self.temporal_method == 'add':
        t = self.counter[idxs]
        p = self.c*(self.b**t) + p
      elif self.temporal_method == 'mult':
        t = self.counter[idxs]
        p = (((self.b ** t)**(self.c))*p)**(1/(1+self.c))

    self.tree.update(idxs, p)
    if increment_counter:
      self.counter[idxs] += 1

  def delete(self, keys):
    """Deletes episodes from tree (sets priority to zero)."""
    idxs = np.array([self.idx_from_ep.pop(key, -1) for key in keys])
    idxs = idxs[np.where(idxs >= 0)[0]]
    self.tree.update(idxs, np.zeros(idxs.shape))
    [self.ep_from_idx.pop(idx, -1) for idx in idxs]
    self.counter[idxs] = 0


  def __str__(self):
    vals = [self.tree.array[idx] for idx in self.idx_from_ep.values()]
    return f"{vals}: {len(vals)}"

  def get_extrema(self, fname=None, n=10):
    """Get a list of the episode/steps that have min and max priority."""
    idxs = np.where(self.tree.array > 0)[0]
    vals = self.tree.array[idxs]
    args = np.argsort(vals)

    extrema = []
    for i in np.append(range(0, n), range(-n, 0)):
      extrema.append([self.ep_from_idx[idxs[args[i]]], vals[args[i]]])

    if fname is not None:
      with open(fname, "w") as f:
        csv.writer(f, delimiter='\t').writerows(extrema)

    return extrema


class Replay:

  def __init__(
      self, directory, capacity=0, ongoing=False, minlen=1, maxlen=0,
      prioritize_ends=False, delete_old_trajectories=True, config=None):
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(parents=True, exist_ok=True)
    self._capacity = capacity
    self._ongoing = ongoing
    self._minlen = minlen
    self._maxlen = maxlen
    self._prioritize_ends = prioritize_ends
    self._random = np.random.RandomState()
    # filename -> key -> value_sequence
    self._complete_eps = load_episodes(self._directory, capacity, minlen)
    # worker -> key -> value_sequence
    self._ongoing_eps = collections.defaultdict(
        lambda: collections.defaultdict(list))
    self._total_episodes, self._total_steps = count_episodes(directory)
    self._loaded_episodes = len(self._complete_eps)
    self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())
    self._delete_old_trajectories = delete_old_trajectories

    # Setup priority buffer.
    self._priority = None
    self._ppriority = None
    if config is not None and config.prioritize_until != 0:
        self.min_priority_step = 1 #10  # Only prioritize steps starting this many steps after the beginning of each 'episode' (to allow for burnin of hidden state)
        self.running_min_value = np.inf
        self.running_min_value_pp = np.inf
        self.priority_fname = f'{str(self._directory)}_priority.npz'
        self.pp = config.do_policy_priority
        if self.pp: self.ppriority_fname = f'{str(self._directory)}_ppriority.npz'

        self._priority = Priority(int(1.3 * self._capacity), config.priority_params)
        if self.pp: self._ppriority = Priority(int(1.3*self._capacity), config.policy_priority_params)  # policy priority

        # Initialize priority for existing episodes.
        keys = self._complete_eps.keys()
        keys = list(sorted(keys))
        for filename in keys:
          steps = np.arange(self.min_priority_step, eplen(self._complete_eps[filename])) # Note: eplen removes the final step.
          epsteps = [f'{filename}:{i}' for i in steps]
          self._priority.add(self._priority.maxval * np.ones(steps.shape), epsteps, is_maxval=True)
          if self.pp: self._ppriority.add(self._ppriority.maxval * np.ones(steps.shape), epsteps, is_maxval=True)
        if os.path.isfile(self.priority_fname):
          print(f'Loading priority buffer from {self.priority_fname}.')
          self.load_priority()
    self.config = config

  @property
  def stats(self):
    return {
        'total_steps': self._total_steps,
        'total_episodes': self._total_episodes,
        'loaded_steps': self._loaded_steps,
        'loaded_episodes': self._loaded_episodes,
    }

  def add_step(self, transition, worker=0):
    episode = self._ongoing_eps[worker]
    for key, value in transition.items():
      episode[key].append(value)
    if transition['is_last']:
      self.add_episode(episode)
      episode.clear()

  def add_episode(self, episode):
    length = eplen(episode)
    if length < self._minlen:
      # print(f'Skipping short episode of length {length}.')
      return
    self._total_steps += length
    self._loaded_steps += length
    self._total_episodes += 1
    self._loaded_episodes += 1
    episode = {key: convert(value) for key, value in episode.items()}
    filename = save_episode(self._directory, episode)
    self._complete_eps[str(filename)] = episode
    if self._priority is not None:
      """Add steps in the new episode to priority buffer, with a max priority."""
      epsteps = [f'{str(filename)}:{step}'
                 for step in np.arange(self.min_priority_step, eplen(episode))]
      self._priority.add(self._priority.maxval * np.ones(len(epsteps)), epsteps, is_maxval=True)
      if self.pp: self._ppriority.add(self._ppriority.maxval * np.ones(len(epsteps)), epsteps, is_maxval=True)
    self._enforce_limit()

  def dataset(self, batch, length):
    example = next(iter(self._generate_chunks(length)))
    dataset = tf.data.Dataset.from_generator(
        lambda: self._generate_chunks(length),
        {k: v.dtype for k, v in example.items()},
        {k: v.shape for k, v in example.items()})
    dataset = dataset.batch(batch, drop_remainder=True)
    dataset = dataset.prefetch(5)
    return dataset

  def _generate_chunks(self, length):
    sequence = self._sample_sequence()
    while True:
      chunk = collections.defaultdict(list)
      added = 0
      while added < length:
        needed = length - added
        adding = {k: v[:needed] for k, v in sequence.items()}
        sequence = {k: v[needed:] for k, v in sequence.items()}
        for key, value in adding.items():
          chunk[key].append(value)
        added += len(adding['action'])
        if len(sequence['action']) < 1:
          sequence = self._sample_sequence()
      chunk = {k: np.concatenate(v) for k, v in chunk.items()}
      yield chunk

  def _sample_sequence(self):
    episodes = list(self._complete_eps.values())
    if self._ongoing:
      episodes += [
          x for x in self._ongoing_eps.values()
          if eplen(x) >= self._minlen]
    episode = self._random.choice(episodes)
    total = len(episode['action'])
    length = total
    if self._maxlen:
      length = min(length, self._maxlen)
    # Randomize length to avoid all chunks ending at the same time in case the
    # episodes are all of the same length.
    length -= np.random.randint(self._minlen)
    length = max(self._minlen, length)
    upper = total - length + 1
    if self._prioritize_ends:
      upper += self._minlen
    index = min(self._random.randint(upper), total - length)
    sequence = {
        k: convert(v[index: index + length])
        for k, v in episode.items() if not k.startswith('log_')}
    sequence['is_first'] = np.zeros(len(sequence['action']), bool)
    sequence['is_first'][0] = True
    if self._maxlen:
      assert self._minlen <= len(sequence['action']) <= self._maxlen
    return sequence

  def _enforce_limit(self):
    if not self._capacity:
      return
    while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
      # Relying on Python preserving the insertion order of dicts.
      oldest, episode = next(iter(self._complete_eps.items()))
      self._loaded_steps -= eplen(episode)
      self._loaded_episodes -= 1
      if self._priority is not None:
        epsteps = [f'{oldest}:{step}' for step in np.arange(self.min_priority_step, eplen(episode) )]
        self._priority.delete(epsteps)
        if self.pp: self._ppriority.delete(epsteps)
      del self._complete_eps[oldest]
      if self._delete_old_trajectories:
        os.remove(oldest)

  def specific_dataset(self, batch, length, which_eps, which_inds):
    """Get a batch with a specific composition of specified episodes.
      Args:
        batch - number of chunks in a batch.
        length - number of timesteps in a chunk.
        which_eps -  list of episode ids, with 'batch' elements (negative numbers indicate relative to the oldest episode)
        which_inds - list of indices (into the specified eps) specifying where each chunk starts.
      Returns:
        tf.Dataset batch object.
    """
    example = self._complete_eps[next(iter(self._complete_eps.keys()))]
    types = {k: v.dtype for k, v in example.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}

    generator = lambda: get_specific_eps(self._complete_eps, length, which_eps, which_inds)
    # t00 = time.time() # TODO: Optimize this.
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    # print(f"##PROFILE## Dataset.from_generator: {time.time() - t00:0.6f}")
    dataset = dataset.batch(batch, drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset

  def get_prioritized_batch(self, batch, length, ppriority_weight=0.0):
    """Return a batch sampled based on the Priority ordering.
    Also return a list of the episodes (and specific step in each episode)
    composing the batch.
    batch - number of batchmembers
    length - number of steps in each batchmember
    ppriority_weight - float between 0 and 1. Weight for mixing in policy_priority with priority.
                       If 0, then just uses the original, single priority.
    """
    ppriority_weight = float(ppriority_weight)
    assert(ppriority_weight >=0 and ppriority_weight <= 1)
    if ppriority_weight > 0:
      if self.pp:
        which_epsteps_pp, idxs_pp = self._ppriority.sample(batch)
      else:
        raise('Trying to use policy_priority without setting config.policy_priority=True')
      if ppriority_weight < 1:
        which_epsteps_p, idxs_p = self._priority.sample(batch)

        ### Now combine them.
        pp_inds = np.array(random.sample(range(batch), int(round(ppriority_weight*batch))))
        which_epsteps = np.array(which_epsteps_p)
        which_epsteps[pp_inds] = np.array(which_epsteps_pp)[pp_inds]
        which_epsteps = list(which_epsteps)

        idxs = np.array(idxs_p)
        idxs[pp_inds] = np.array(idxs_pp)[pp_inds]
        idxs = list(idxs)

      else:
        which_epsteps, idxs = which_epsteps_pp, idxs_pp
    else:
      which_epsteps, idxs = self._priority.sample(batch)

    which_eps = [x.split(':')[0] for x in which_epsteps]
    which_inds = [int(x.split(':')[-1]) - self.min_priority_step
                  for x in which_epsteps]   # Select window with 'min_priority_step' steps preceding the step of interest.

    try:
      totals = [eplen(self._complete_eps[x]) for x in which_eps]
    except:
      self.save_priority()
      raise(f'self._complete_eps does not contain all of which_eps: {which_eps}')

    which_inds = [min(x, total - length) for (x, total) in zip(which_inds, totals)]

    # Assemble list of keys corresponding to steps in the batch, to be used for updating priorities.
    batch_eps = []
    for ep, ind in zip(which_eps, which_inds):
      batch_eps.append([f'{ep}:{i}' for i in range(ind + self.min_priority_step, ind + length)])

    # t00 = time.time()  # TODO: Optimize this.
    specific_dataset = self.specific_dataset(batch, length, which_eps, which_inds)
    specific_batch = next(iter(specific_dataset))
    # print(f"##PROFILE## specific_batch: {time.time() - t00:0.6f}")

    return specific_batch, batch_eps

  def update_batch_prioritization(self, outputs, batch_eps, do_ppriority=None, increment_counter=True):
    """
    This function is the heart of where you determine what is prioritized.
    May deserve more investigation (e.g. different vals, different use of running_min, etc).
    :param outputs: dict return by agnt.train
    :param batch_eps: list with keys into priority buffer (of form 'ep_filename:step')
    :param do_ppriority: None, False, or True. If None, then updates ppriority if it exists.
    :return:
    """
    if do_ppriority is None or not do_ppriority:
      metric = self._priority.metric
      if metric == 'CR' or metric == 'losses':  # See agent.loss(), computes loss from -likes.
        vals = sum(-self.config.loss_scales.get(k, 1.0) * v
                   for k, v in outputs['likes'].items())
      else:
        vals = outputs[metric]
      vals = vals[:, -len(batch_eps[0]):]  # Accounts for if outputs[metric] excludes step 0 (e.g. when computing TD error, don't have a value for the first step).

      if self._priority.running_min and metric != 'TD':
        self.running_min_value = min(self.running_min_value, vals.min())  ## XX This is potentially not robust.
        vals -= self.running_min_value  # Increase the dynamic range by subtracting baseline value.

      newvals = np.array(vals).flatten()
      epsteps = np.array(batch_eps).flatten()
      idxs = [self._priority.idx_from_ep[epstep] for epstep in epsteps]
      self._priority.update(idxs, newvals, increment_counter)

    if do_ppriority is None or do_ppriority:
      if self.pp:
        metric = self._ppriority.metric
        if metric == 'CR' or metric == 'losses':  # See agent.loss(), computes loss from -likes.
          vals = sum(-self.config.loss_scales.get(k, 1.0) * v
                     for k, v in outputs['likes'].items())
        else:
          vals = outputs[metric]
        vals = vals[:, -len(batch_eps[0]):]

        if self._ppriority.running_min and metric != 'TD':
          self.running_min_value_pp = min(self.running_min_value_pp, vals.min())  ## XX This is potentially not robust.
          vals -= self.running_min_value_pp  # Increase the dynamic range by subtracting baseline value.

        newvals = np.array(vals).flatten()
        epsteps = np.array(batch_eps).flatten()
        idxs = [self._ppriority.idx_from_ep[epstep] for epstep in epsteps]
        self._ppriority.update(idxs, newvals, increment_counter)
      else:
        if do_ppriority is not None:
          raise('Trying to use policy_priority without setting config.policy_priority=True')


  def save_priority(self, suffix=None):
    priority_fname = self.priority_fname
    if suffix is not None:
      priority_fname = f'{priority_fname}{suffix}'
    with open(priority_fname, 'wb') as f:
      np.savez_compressed(f, arr=self._priority.tree.array[:], counter=self._priority.counter)
      print(f'Saved buffer prioritization to {priority_fname}')
    with open(f'{priority_fname}.pkl', 'wb') as f:
      pickle.dump({'ep_from_idx': self._priority.ep_from_idx, 'idx_from_ep': self._priority.idx_from_ep}, f)

    # Save ppriority
    if self.pp:
      ppriority_fname = f'{priority_fname}p'
      with open(ppriority_fname, 'wb') as f:
        np.savez_compressed(f, arr=self._ppriority.tree.array[:], counter=self._ppriority.counter)
        print(f'Saved buffer pprioritization to {ppriority_fname}')
      with open(f'{ppriority_fname}.pkl', 'wb') as f:
        pickle.dump({'ep_from_idx': self._ppriority.ep_from_idx, 'idx_from_ep': self._ppriority.idx_from_ep}, f)

  def load_priority(self):
    with open(self.priority_fname, 'rb') as f:
      arr = np.load(f)['arr']
      self._priority.tree.update(np.arange(len(arr)), arr)
      print(f'Loaded buffer prioritization from {self.priority_fname}.pkl')
    with open(f'{self.priority_fname}.pkl', 'rb') as f:
      d = pickle.load(f)
      self._priority.ep_from_idx = d['ep_from_idx']
      self._priority.idx_from_ep = d['idx_from_ep']
    # Load ppriority
    if self.pp:
      ppriority_fname = f'{self.priority_fname}p'
      with open(ppriority_fname, 'rb') as f:
        arr = np.load(f)['arr']
        self._ppriority.tree.update(np.arange(len(arr)), arr)
        print(f'Loaded buffer prioritization from {ppriority_fname}.pkl')
      with open(f'{ppriority_fname}.pkl', 'rb') as f:
        d = pickle.load(f)
        self._ppriority.ep_from_idx = d['ep_from_idx']
        self._ppriority.idx_from_ep = d['idx_from_ep']

def update_priority_entire_buffer(agent, batch, length):
  raise('Not implemented.')


def count_episodes(directory):
  filenames = list(directory.glob('*.npz'))
  num_episodes = len(filenames)
  num_steps = sum(int(str(n).split('-')[-1][:-4]) - 1 for n in filenames)
  return num_episodes, num_steps


def save_episode(directory, episode):
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  identifier = str(uuid.uuid4().hex)
  length = eplen(episode)
  filename = directory / f'{timestamp}-{identifier}-{length}.npz'
  with io.BytesIO() as f1:
    np.savez_compressed(f1, **episode)
    f1.seek(0)
    with filename.open('wb') as f2:
      f2.write(f1.read())
  return filename


def load_episodes(directory, capacity=None, minlen=1):
  # The returned directory from filenames to episodes is guaranteed to be in
  # temporally sorted order.
  filenames = sorted(directory.glob('*.npz'))
  if capacity:
    num_steps = 0
    num_episodes = 0
    for filename in reversed(filenames):
      length = int(str(filename).split('-')[-1][:-4])
      num_steps += length
      num_episodes += 1
      if num_steps >= capacity:
        break
    filenames = filenames[-num_episodes:]
  episodes = {}
  for filename in filenames:
    try:
      with filename.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode {str(filename)}: {e}')
      continue
    episodes[str(filename)] = episode
  return episodes


def convert(value):
  value = np.array(value)
  if np.issubdtype(value.dtype, np.floating):
    return value.astype(np.float32)
  elif np.issubdtype(value.dtype, np.signedinteger):
    return value.astype(np.int32)
  elif np.issubdtype(value.dtype, np.uint8):
    return value.astype(np.uint8)
  return value


def eplen(episode):
  return len(episode['action']) - 1


def get_specific_eps(episodes, length, which_eps, which_inds):
  """
  Load specific episodes.
  :param which_eps: list. either strings specifying name of episode, or int, specifying order (0 is most recent)
  :param which_inds: list. index into each ep.
  :return:
  """
  eps = list(episodes.keys())
  eps.sort(reverse=True)

  # print(f"Get specific eps: {[f'{ep}, {index}, {index + length}' for ep, index in zip(which_eps, which_inds)]}")

  for i in range(len(which_eps)):
    if not isinstance(which_eps[i], str):
      ep = eps[int(which_eps[i])]
    else:
      ep = which_eps[i]
    episode = episodes[ep]
    index = which_inds[i]
    episode = {k: v[index: index + length] for k, v in episode.items()}

    yield episode