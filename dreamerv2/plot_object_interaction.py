"""Plot summary of object interaction log."""
import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
try:
  matplotlib.use('module://backend_interagg')
except:
  pass

# Defaults
SAVERATE = 20  # record_every_k_timesteps param in logging_params in adaptgym.wrapped.ADMC
PREFILL = 1e3  # prefill param in config.yaml


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Path to the log directory',
                        default='~/logdir/admc_sphero_novel_object_unchanging/dreamerv2_cr/2')
    parser.add_argument('--outdir', type=str, help='Path to the output directory for plots',
                        default=None)
    return parser.parse_args()


def main():
  args = parse_arguments()
  logdir = args.logdir
  outdir = args.outdir
  os.makedirs(outdir, exist_ok=True)
  expid = '/'.join(logdir.split('/')[-3:])
  fn = f'{logdir}/log_train_env0.csv'
  df = pd.read_csv(fn)

  colors = {'object2': 'magenta'}  # Object name defined by admc_sphero_novel_object_unchanging environment.
  labels = {'object2': 'novel'}
  ball_ids = list(colors.keys())

  # Set timestep of object introduction into environment.
  if 'unchanging' in fn:
    tstart = int((0 + PREFILL) / SAVERATE)
    tend = int(1e6 / SAVERATE)
  else:
    tstart = int((5e5 + PREFILL) / SAVERATE)
    tend = int(1.5e6 / SAVERATE)

  #  Plot overall exploration trajectory and interactions.
  plot_indiv_playground_object_expt(df, tstart, tend, ball_ids, colors,
                                    labels, expid, SAVERATE,
                                    savepath=outdir)

  #  Trajectory snapshots at multiple timepoints after object introduction.
  tends = tstart + np.round(np.array([100e3, 200e3, 300e3])
                            / SAVERATE).astype(int)
  plot_indiv_trajectories(df, tstart, tends, ball_ids, colors,
                          titlestrs=['100k steps', '200k steps', '300k steps'],
                          savepath=outdir,
                          )


def plot_indiv_playground_object_expt(plot_df, tstart, tend,
                                      ball_ids, colors, labels,
                                      expid, saverate,
                                      savepath=None):
  """Plot map of agent and object locations over the session,
  and cumulative collisions vs time."""
  plt.figure(figsize=(8, 3))
  ax = plt.subplot(1, 2, 1)
  print(tstart)
  print(tend)
  print(plot_df['agent0_xloc'][tstart:tend])
  print(plot_df['agent0_yloc'][tstart:tend])

  plt.plot(plot_df['agent0_xloc'][tstart:tend],
           plot_df['agent0_yloc'][tstart:tend], 'k.',
           markersize=1, alpha=0.2)
  for b in ball_ids:
    plt.plot(plot_df[f'{b}_xloc'][tstart:tend],
             plot_df[f'{b}_yloc'][tstart:tend], '.',
             markersize=2, color=colors[b], alpha=0.5)
  ax.axis('equal')
  ax.set_xlim([-15, 15])
  ax.set_ylim([-15, 15])
  plt.axis('off')
  plt.suptitle(f'{expid}')

  plt.subplot(1, 2, 2)
  legend_str = []
  for b in ball_ids:
    y = np.cumsum(plot_df[f'collisions_{b}/shell'][tstart:tend])
    tt = saverate * np.arange(y.shape[0])
    plt.plot(tt, y,
             color=colors[b])
    legend_str.append(f'{b}:' + labels[f'{b}'])
    plt.xlabel('Steps in env')
    plt.ylabel('Cumulative collisions')
  plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
  plt.tight_layout()
  if savepath is not None:
    plt.savefig(f'{savepath}/interaction_summary.png')
    print(f'Saving to {savepath}')
  plt.show()

def plot_indiv_trajectories(plot_df, tstart, tends,
                            ball_ids, colors,  titlestrs,
                            savepath=None):
  """Plot map of agent and object locations accumulated
  to different timepoints."""
  fig, axs = plt.subplots(1, len(tends), figsize=(9, 3))
  for i in range(len(tends)):
    ax = axs[i]
    tend = tends[i]
    ax.plot(plot_df['agent0_xloc'][tstart:tend],
            plot_df['agent0_yloc'][tstart:tend], 'k.',
            markersize=1, alpha=0.2)
    for b in ball_ids:
      ax.plot(plot_df[f'{b}_xloc'][tstart:tend],
              plot_df[f'{b}_yloc'][tstart:tend], '.',
              markersize=2, color=colors[b], alpha=0.5)
    rect = patches.Rectangle((-12.76, -14.1713),
                             2 * 12.76, 2 * 12.76,
                             linewidth=1, edgecolor='None',
                             facecolor='#eeeeee')
    ax.add_patch(rect)
    ax.annotate(f'{titlestrs[i]}', (-4, 13), fontsize=12)
    ax.axis('equal')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-14.5, 12])
    ax.axis('off')
  plt.tight_layout()
  if savepath is not None:
    plt.savefig(f'{savepath}/interaction_snapshots.png')
    print(f'Saving to {savepath}')
  plt.show()


if __name__ == "__main__":
  main()
