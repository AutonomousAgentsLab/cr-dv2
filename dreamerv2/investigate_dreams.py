"""
Load a model, investigate what it is dreaming.
Potentially use hand-crafted test episodes from envs.scripts.make_example_episodes.py

See download_ckpt.sh to get necessary files from remote server.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import time
import pickle

import elements
# from dreamerv2 import common
# from dreamerv2 import agent
import argparse
from tqdm import tqdm

from os.path import expanduser
import training.utils.loading_utils as lu
import training.utils.diagnosis_utils as du

tf.config.run_functions_eagerly(True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify episode arguments.')
  # parser.add_argument('--bufferid', default='DRF-416')
  parser.add_argument('--bufferid', default='CRAFTER_EXAMPLE_EPS')

  parser.add_argument('--expid',    default='DRF-416')
  # parser.add_argument('--expid',    default='DRF-80')
  parser.add_argument('--ckpt',    default='variables.pkl')
  parser.add_argument('--plot_intrinsic_reward', default='0')
  parser.add_argument('--burnin', default='30')
  parser.add_argument('--imagine_for', default='45')
  # parser.add_argument('--which_ep', default=None)

  ## An early episode 1
  # parser.add_argument('--which_ep', default='20230124T052747-8197b66266304fe89bb730ebb49890c5-270.npz') # An early episode (no iron?)
  # parser.add_argument('--start_imagining_at', default='170')

  ## An early episode 2
  parser.add_argument('--which_ep', default='20230124T053031-3301d31c6f124840ade2472ec841409f-240.npz') # An early episode (no iron?)
  parser.add_argument('--start_imagining_at', default='120')

  ## An early episode 3
  # parser.add_argument('--which_ep', default='20230124T131501-293c2ded03bb4425b1057606f2cbdce1-235.npz') # An early episode (no iron?)
  # parser.add_argument('--start_imagining_at', default='150')

  ## Example 1 of iron
  # parser.add_argument('--which_ep', default='20230126T044509-22798cceadf24b4f8b3335261347ec46-242.npz') # iron at 145
  # parser.add_argument('--start_imagining_at', default='142')

  ## Example 2 of iron (best)
  # parser.add_argument('--which_ep', default='20230125T214154-26ff020b41d1404489c9ab0f3a49fdc3-409.npz') # iron at 318
  # parser.add_argument('--start_imagining_at', default='315')

  ## Example 3 of iron (best)
  # parser.add_argument('--which_ep', default='20230126T013711-b8f271ded9684ced836ecab7970b78e0-387.npz') # iron at 166
  # parser.add_argument('--start_imagining_at', default='163')

  # parser.add_argument('--start_imagining_at', default=None)

  args = parser.parse_args()
  expid = args.expid # For Fig 1: 'DRE-354'   # 'DRE-377'
  bufferid = args.bufferid
  plot_intrinsic_reward = bool(int(args.plot_intrinsic_reward))
  burnin = int(args.burnin)
  imagine_for = int(args.imagine_for)
  print(f'plot_intrinsic_reward: {plot_intrinsic_reward}')

  if args.ckpt != 'None':
    checkpoint_name = args.ckpt
  else:
    # checkpoint_name = 'variables_train_agent_envindex0_final_000502500.pkl'
    checkpoint_name = 'variables_pretrained_env0.pkl'

  print(f'Ckpt: {checkpoint_name}')

  # which_eps = [[98], [99], [100],[101]]
  # which_eps = [[0], [1], [2], [3], [4]]
  # start_imagining_at = 220 #263   ### 30 FOR DEBUGGING
  which_eps = [[10]]
  start_imagining_at = 315 # Iron is at step 145
  if 'GEN-EXAMPLE_EPS' in bufferid:
    # which_eps = [[0], [1], [2], [3]]
    which_eps = [[1], [3]]
    start_imagining_at = 36

  else:
    if args.which_ep is not None:
      which_eps = [[args.which_ep]]
    if args.start_imagining_at is not None:
      start_imagining_at = int(args.start_imagining_at)

  print(which_eps)
  print(f'start_imagining_at {start_imagining_at}')

  # replay_buffer_name = 'train_replay_0'
  replay_buffer_name = 'train_episodes'


  print(f'which_eps: {which_eps}, start_imagining_at: {start_imagining_at}')

  ylim_intr_rew = None

  ylim_intr_rew=[-3.5, -1.5]
  # ylim_intr_rew=[0, 0.0005]
  # ylim_intr_rew = [0, 0.05]
  # ylim_intr_rew = [0, 0.001]


  home = expanduser("~")
  basedir = f"{home}/logs/{expid}"
  buffer_basedir = f"{home}/logs/{bufferid}"
  plot_lims=[-20,20]

  plot_dir = f'{basedir}/plots'
  os.makedirs(plot_dir, exist_ok=True)

  from tensorflow.keras.mixed_precision import experimental as prec
  prec.set_policy(prec.Policy('mixed_float16'))

  ## Load agent from checkpoint
  agnt = lu.load_agent_dv2(basedir,
                       checkpoint_name=checkpoint_name,
                       batch_size=5,
                       )

  do_visualize_dream = True


  if do_visualize_dream:
    all_loss = []
    all_imgs_w_burnin = []
    all_intr_reward = []
    all_img_mse_w_burnin = []
    all_img_abserr_w_burnin = []
    for batch_eps in which_eps:

      eps = lu.load_eps(buffer_basedir, replay_buffer_name, batch_eps=batch_eps)

      nt = agnt.config.dataset.length
      nt = eps[0]['action'].shape[0]

      # xyz = np.stack([ep['absolute_position_agent0'][:nt, :] for ep in eps], axis=0)
      # data = {key: np.vstack([np.expand_dims(ep[key].astype('float32'), axis=0)
      #                         for ep in eps])[:, :nt]
      #         for key in eps[0].keys()}
      data = {key: np.vstack([np.expand_dims(ep[key], axis=0)
                              for ep in eps])[:, :nt]
              for key in eps[0].keys()}
      data = {key: tf.cast(data[key], dtype=data[key].dtype) for key in data.keys()}

      # Optionally, overwrite actions here, by overwriting data['action']

      save_dir = f"{plot_dir}/{bufferid}/{checkpoint_name}/{batch_eps[0]}"
      os.makedirs(save_dir, exist_ok=True)

      actions_for_imagine = data['action']
      # key_t = np.array([-10, -3, 0, 5, 10, 15, 20])
      # key_t = np.array([-10, -3, 0, 5, 10, 15, 20, 25, 30])
      # key_t = np.array([0, 3, 6, 9, 12, 15, 18])  ## Use for figs
      key_t = np.array([-10, -3, 0, 3, 6, 9, 12])  #

      loss_imagine, loss_burnin, intr_reward, imgs_w_burnin, img_mse_w_burnin, img_abserr_w_burnin  = \
                                           du.dream_dv2(agnt, data,
                                           actions_for_imagine, save_dir,
                                           expid=expid,
                                           start_imagining_at=start_imagining_at,
                                           burnin=burnin,
                                           imagine_for=imagine_for,
                                           plot_intrinsic_reward=plot_intrinsic_reward,
                                           show_burnin_recon=False,
                                           deterministic_wm=True,
                                           do_save_gif=True,
                                           ylim_imgloss=[11200, 12300],
                                           ylim_intr_rew=ylim_intr_rew,
                                           key_t=key_t,
                                           do_return_imgs=True,
                                           do_include_error_img=True,
                                           fig_suffix='pdf',
                                           )
      all_loss.append(np.hstack((loss_burnin, loss_imagine)))
      all_intr_reward.append(intr_reward)
      all_imgs_w_burnin.append([imgs_w_burnin[t+burnin] for t in key_t])
      all_img_mse_w_burnin.append(img_mse_w_burnin)
      all_img_abserr_w_burnin.append(img_abserr_w_burnin)

  do_plot_img_comparison = False
  if do_plot_img_comparison:
    for i in range(len(all_imgs_w_burnin)):
      imgs_w_burnin = all_imgs_w_burnin[i]
      nimgs = len(np.where(key_t>0)[0])
      plt.figure()
      for ii in range(nimgs):
        plt.subplot(3, nimgs, ii+1)
        plt.imshow
        plt.subplot(3, nimgs, ii+nimgs)

        plt.subplot(3, nimgs, ii+2*nimgs)




  save_dir = f"{plot_dir}/{bufferid}/{checkpoint_name}"
  plt.figure()
  for i, ep  in enumerate(which_eps):
    plt.plot(np.arange(-burnin, imagine_for), all_loss[i][0])
  plt.legend(which_eps)
  plt.ylabel('Image Loss')
  plt.xlabel('Imagined step')
  plt.suptitle(':'.join([expid, bufferid, checkpoint_name]), fontsize=10)
  plt.savefig(os.path.join(save_dir, 'img_loss.png'))
  print(f"saved to {os.path.join(save_dir, f'img_loss_{args.which_ep}.png')}")
  plt.show()

  to_save = {}
  to_save['all_loss'] = all_loss
  to_save['bufferid'] = bufferid
  to_save['which_eps'] = which_eps
  to_save['start_imagining_at'] = start_imagining_at
  to_save['replay_buffer_name'] = replay_buffer_name
  to_save['burnin'] = burnin
  to_save['imagine_for'] = imagine_for
  to_save['imgs_t'] = key_t
  to_save['all_imgs_w_burnin'] = all_imgs_w_burnin
  to_save['all_intr_reward'] = all_intr_reward
  to_save['all_img_mse_w_burnin'] = all_img_mse_w_burnin
  to_save['all_img_abserr_w_burnin'] = all_img_abserr_w_burnin
  pickle.dump(to_save, open(os.path.join(save_dir, f'img_loss_{args.which_ep}.pkl'), 'wb'))
  print('saved to ', os.path.join(save_dir, f'img_loss_{args.which_ep}.pkl'))
  print('done')