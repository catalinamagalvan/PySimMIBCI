"""
# Author: Catalina M. Galvan <cgalvan@santafe-conicet.gov.ar>
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_2_channels(raw, save_name=None, save=False, start=30,
                        duration=20):
    """
    Plot C3 and C4 channels of an EEG segment.

    Parameters
    ----------
    raw : Instance of MNE Raw
        The corresponding MNE Raw.
    save_name : str, optional
        The name of the file where the plot will be saved if save=True. The
        default is None.
    save : bool, optional
        Whether to save the plot. The default is False.
    start : float, optional
        Time in seconds of the start of the segment to plot. The default is 30.
    duration : float, optional
        Time in seconds of the duration of the segment to plot. The default is
        20.

    Returns
    -------
    None.

    """
    indices = [raw.ch_names.index('C3'), raw.ch_names.index('C4')]  # C3 y C4
    data_eeg = raw.get_data(picks='eeg')

    temporal_line = np.arange(int(start*raw.info['sfreq']),
                              int((start+duration)*raw.info['sfreq']),
                              dtype='float64')
    temporal_line /= raw.info['sfreq']
    channels_names = raw.info['ch_names']

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True,
                            sharex=True, sharey=True)
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
