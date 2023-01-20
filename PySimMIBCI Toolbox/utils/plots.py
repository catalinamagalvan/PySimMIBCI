# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:55:19 2021

@author: catal
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_2_channels(raw, save_name=None, save=False, start=30,
                        duration=20):
    """Plot a set of channels in raw data."""
    indices = [raw.ch_names.index('C3'), raw.ch_names.index('C4')]  # C3 y C4
    data_eeg = raw.get_data(picks='eeg')

    temporal_line = np.arange(int(start*raw.info['sfreq']),
                              int((start+duration)*raw.info['sfreq']),
                              dtype='float64')
    temporal_line /= raw.info['sfreq']
    channels_names = raw.info['ch_names']

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True,
                            sharex=True, sharey=True)
    # fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axs.flat):
        line, = ax.plot(temporal_line, data_eeg[indices[i],
                                                int(start*raw.info['sfreq']):
                                                int((start+duration)*raw.info
                                                    ['sfreq'])],
                        linewidth=0.3)
        ax.set_ylim([-100, 100])
        ax.set_title(channels_names[indices[i]])
    # Reserve space for axis labels
    axs[-1].set_xlabel('.', color=(0, 0, 0, 0))
    axs[-1].set_ylabel('.', color=(0, 0, 0, 0))
    # Make common axis labels
    fig.text(0.5, 0.02, 'Time [s]', va='center', ha='center', fontsize=16)
    fig.text(0.01, 0.5, 'Amplitude ' + r'$\mu$V', va='center', ha='center',
             rotation='vertical', fontsize=16)
    if save:
        fig.savefig(save_name, transparent=True, dpi=2000,
                    bbox_inches='tight', pad_inches=0)
    plt.show()
