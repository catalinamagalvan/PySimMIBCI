"""
# Author: Catalina M. Galvan <cgalvan@santafe-conicet.gov.ar>
"""

import numpy as np
from fooof import FOOOF
from fooof.analysis import get_band_peak_fm
import matplotlib.pyplot as plt


def fit_user_params(epochs, epochs_right, epochs_left, f_min=1, f_max=40,
                    t_min=0, t_max=3, aperiodic_mode='fixed',
                    peak_width_limits=(1, 10), band=(7, 14),
                    select_highest=True):
    """
    Fit the user-specific aperiodic and periodic params for the right hand MI
    vs left hand MI scenario.

    Parameters
    ----------
    epochs : instance of MNE Epochs
        The real MI-EEG epochs.
    epochs_right : instance of MNE Epochs
        The real MI-EEG epochs corresponding to right hand MI class.
    epochs_left : instance of MNE Epochs
        The real MI-EEG epochs corresponding to left hand MI class.
    f_min : float, optional
        Min frequency of interest in the computation of the power spectral
        density (PSD) by multitaper method. The default is 1.
    f_max : float, optional
        Max frequency of interest in the computation of the PSD by multitaper
        method. The default is 40.
    t_min : float, optional
        Min time of interest in the computation of the PSD by multitaper
        method. The default is 0.
    t_max : float, optional
        Min time of interest in the computation of the PSD by multitaper
        method. The default is 3.
    aperiodic_mode : {'fixed', 'knee', None}
        The approach to use to parameterize the aperiodic component by FOOOF.
        If 'fixed', it fits with just an offset and a exponent, equivalent to
        a linear fit in log-log space.
        If ‘knee’, it includes a knee parameter, reflecting a fit with a bend,
        in log-log space
        The default is 'fixed'.
    peak_width_limits : tuple of (float, float), optional.
        Limits on possible peak width, in Hz, as (lower_bound, upper_bound).
        The default is (1, 10).
    band : tuple of (float, float), optional
        Frequency range for the band of interest in the peaks extraction.
        Defined as: (lower_frequency_bound, upper_frequency_bound).
        The default is (7, 14).
    select_highest : bool, optional
        Whether to return single peak (if True) or all peaks within the range
        found (if False). If True, returns the highest power peak within the
        search range. The default is True.

    Returns
    -------
    user_params : dict
        User-specific params dictionary.
        'aperiodic_params': 1d array
        Parameters that define the aperiodic fit as [Offset, (Knee), Exponent].
        The knee parameter is only included if aperiodic component is fitted
        with a knee.
        'peak_params': 2d array
        Fitted parameter values for the peaks. Each row is a peak, as
        [CF: center frequency, PW: power, BW: Bandwidth].
    """
    ch_names = epochs.ch_names
    # PSD and FOOOF for real data
    # Compute the power spectral density (PSD) using multitapers.
    # Calculates spectral density for orthogonal tapers, then averages them
    # together for each channel/epoch.
    PSD_right, freqs = epochs_right.compute_psd(method='multitaper',
                                                fmin=f_min, fmax=f_max,
                                                tmin=t_min,
                                                tmax=t_max).get_data(
                                                    return_freqs=True)
    PSD_left, freqs = epochs_left.compute_psd(method='multitaper',
                                              fmin=f_min, fmax=f_max,
                                              tmin=t_min, tmax=t_max).get_data(
                                                  return_freqs=True)
    PSD_both, freqs_both = epochs.compute_psd(method='multitaper',
                                              fmin=f_min, fmax=f_max).get_data(
                                                  return_freqs=True)
    output_both_full = fit_FOOOF(freqs_both, np.mean(np.mean(PSD_both, axis=1),
                                                     axis=0),
                                 aperiodic_mode=aperiodic_mode, band=band,
                                 select_highest=select_highest)
    output_right_C3 = fit_FOOOF(freqs, np.mean(PSD_right[:, ch_names.index(
        'C3')], axis=0), aperiodic_mode=aperiodic_mode, band=band,
        select_highest=select_highest)
    output_right_C4 = fit_FOOOF(freqs, np.mean(
        PSD_right[:, ch_names.index('C4')], axis=0),
        aperiodic_mode=aperiodic_mode, band=band,
        select_highest=select_highest)
    output_left_C3 = fit_FOOOF(freqs, np.mean(
        PSD_left[:, ch_names.index('C3')], axis=0),
        aperiodic_mode=aperiodic_mode, band=band,
        select_highest=select_highest)
    output_left_C4 = fit_FOOOF(freqs, np.mean(
        PSD_left[:, ch_names.index('C4')], axis=0),
        aperiodic_mode=aperiodic_mode, band=band,
        select_highest=select_highest)

    aperiodic_params = output_both_full['aperiodic_params']
    peak_params = {}
    if not np.isnan(output_left_C3['peak_params'][0]):
        peak_params['G_precentral-lh'] = list(output_left_C3['peak_params'])
    elif not np.isnan(output_right_C3['peak_params'][0]):
        peak_params['G_precentral-lh'] = list(output_right_C3['peak_params'])
    else:
        peak_params['G_precentral-lh'] = [11, 1, 2.5]

    if not np.isnan(output_right_C4['peak_params'][0]):
        peak_params['G_precentral-rh'] = list(output_right_C4['peak_params'])
    elif not np.isnan(output_left_C4['peak_params'][0]):
        peak_params['G_precentral-rh'] = list(output_left_C4['peak_params'])
    else:
        peak_params['G_precentral-rh'] = [11, 1, 2.5]

    user_params = {'aperiodic_params': aperiodic_params,
                   'peak_params': peak_params}
    return user_params


def fit_FOOOF(freqs, power_spectrum, aperiodic_mode='fixed',
              peak_width_limits=(1, 10), plot=False, savefig=False,
              filename=None, band=(7, 14), select_highest=True):
    """
    Initialize and fit FOOOF object.

    Parameters
    ----------
    freqs : 1d array
        Frequency values for the power spectrum, in linear space.
    power_spectrum : 1d array
        Power spectrum values, which must be input in linear space.
    aperiodic_mode : {'fixed', 'knee', None}
        The approach to use to parameterize the aperiodic component by FOOOF.
        If 'fixed', it fits with just an offset and a exponent, equivalent to
        a linear fit in log-log space.
        If ‘knee’, it includes a knee parameter, reflecting a fit with a bend,
        in log-log space
        The default is 'fixed'.
    peak_width_limits : tuple of (float, float), optional.
        Limits on possible peak width, in Hz, as (lower_bound, upper_bound).
        The default is (1, 10).
    plot : bool, optional
        Whether to plot the power spectrum and model fit results from the
        FOOOF object. The default is False.
    savefig : bool, optional
        Whether to save out a copy of the plot. The default is False.
    filename : str, optional
        Name to give the saved out file. The default is None.
    band : tuple of (float, float), optional
        Frequency range for the band of interest in the peaks extraction.
        Defined as: (lower_frequency_bound, upper_frequency_bound).
        The default is (7, 14).
    select_highest : bool, optional
        Whether to return single peak (if True) or all peaks within the range
        found (if False). If True, returns the highest power peak within the
        search range. The default is True.

    Returns
    -------
    output : dict
        FOOOF output dictionary.
        'aperiodic_mode': {'fixed', 'knee'}
            The approach used to parameterize the aperiodic component by FOOOF.
        'aperiodic_params': 1d array
            Parameters that define the aperiodic fit as
            [Offset, (Knee), Exponent]. The knee parameter is only included if
            aperiodic component is fitted with a knee.
        'peak_params': 2d array
            Fitted parameter values for the peaks. Each row is a peak, as
            [CF: center frequency, PW: power, BW: Bandwidth].
        'r_squared': float
            R-squared of the fit between the input power spectrum and the full
            model fit.
        'fm': instance of FOOOF.
            The fitted FOOOF object.

    """
    output = dict()
    if aperiodic_mode is not None:
        fm = FOOOF(aperiodic_mode=aperiodic_mode,
                   peak_width_limits=peak_width_limits)
        fm.fit(freqs, power_spectrum)
        output['aperiodic_mode'] = aperiodic_mode

    else:
        fm_fixed = FOOOF(aperiodic_mode='fixed',
                         peak_width_limits=peak_width_limits)
        fm_knee = FOOOF(aperiodic_mode='knee',
                        peak_width_limits=peak_width_limits)
        # Report: fit the model, print the resulting parameters,
        # and plot the reconstruction
        fm_fixed.fit(freqs, power_spectrum)
        fm_knee.fit(freqs, power_spectrum)
        if fm_knee.r_squared_ >= fm_fixed.r_squared_:
            fm = fm_knee
            output['aperiodic_mode'] = 'knee'
        else:
            fm = fm_fixed
            output['aperiodic_mode'] = 'fixed'

    output['aperiodic_params'] = fm.aperiodic_params_
    output['r_squared'] = fm.r_squared_
    output['peak_params'] = get_band_peak_fm(fm, band,
                                             select_highest=select_highest)
    output['fm'] = fm

    if plot:
        fm.plot()
        if savefig:
            plt.savefig(filename, format='png', transparent=True, dpi=2000,
                        bbox_inches='tight', pad_inches=0)
    return output


def plot_FOOOF(freqs, power_spectrum, aperiodic_mode='fixed',
               peak_width_limits=(1, 10), savefig=False,
               filename=None, band=(7, 14), select_highest=True,
               ax=None, add_legend=True):
    """
    Fit and plot FOOOF object.

    Parameters
    ----------
    freqs : 1d array
        Frequency values for the power spectrum, in linear space.
    power_spectrum : 1d array
        Power spectrum values, which must be input in linear space.
    aperiodic_mode : {'fixed', 'knee', None}
        The approach to use to parameterize the aperiodic component by FOOOF.
        If 'fixed', it fits with just an offset and a exponent, equivalent to
        a linear fit in log-log space.
        If ‘knee’, it includes a knee parameter, reflecting a fit with a bend,
        in log-log space
        The default is 'fixed'.
    peak_width_limits : tuple of (float, float), optional.
        Limits on possible peak width, in Hz, as (lower_bound, upper_bound).
        The default is (1, 10).
    savefig : bool, optional
        Whether to save out a copy of the plot. The default is False.
    filename : str, optional
        Name to give the saved out file. The default is None.
    band : tuple of (float, float), optional
        Frequency range for the band of interest in the peaks extraction.
        Defined as: (lower_frequency_bound, upper_frequency_bound).
        The default is (7, 14).
    select_highest : bool, optional
        Whether to return single peak (if True) or all peaks within the range
        found (if False). If True, returns the highest power peak within the
        search range. The default is True.
    ax: matplotlib.Axes, optional
        Figure axes upon which to plot.
    add_legend: boolean, optional, default: True
        Whether to add a legend describing the plot components.

    """
    if aperiodic_mode is not None:
        fm = FOOOF(aperiodic_mode=aperiodic_mode,
                   peak_width_limits=peak_width_limits)
        fm.fit(freqs, power_spectrum)
    else:
        fm_fixed = FOOOF(aperiodic_mode='fixed',
                         peak_width_limits=peak_width_limits)
        fm_knee = FOOOF(aperiodic_mode='knee',
                        peak_width_limits=peak_width_limits)
        # Report: fit the model, print the resulting parameters,
        # and plot the reconstruction
        fm_fixed.fit(freqs, power_spectrum)
        fm_knee.fit(freqs, power_spectrum)
        if fm_knee.r_squared_ >= fm_fixed.r_squared_:
            fm = fm_knee
        else:
            fm = fm_fixed

    fm.plot(ax=ax, add_legend=add_legend)
    if savefig:
        plt.savefig(filename, format='png', transparent=True, dpi=2000,
                    bbox_inches='tight', pad_inches=0)


def plot_PSD_FOOOF(epochs_right, epochs_left, f_min=1, f_max=40, t_min=0,
                   t_max=3, title=None):
    """
    Calculate and plot the power spectrum and the FOOOF model fit results.
    Parameters
    ----------
    epochs_right : instance of MNE Epochs
        The real MI-EEG epochs corresponding to right hand MI class.
    epochs_left : instance of MNE Epochs
        The real MI-EEG epochs corresponding to left hand MI class.
    f_min : float, optional
        Min frequency of interest in the computation of the power spectral
        density (PSD) by multitaper method. The default is 1.
    f_max : float, optional
        Max frequency of interest in the computation of the PSD by multitaper
        method. The default is 40.
    t_min : float, optional
        Min time of interest in the computation of the PSD by multitaper
        method. The default is 0.
    t_max : float, optional
        Min time of interest in the computation of the PSD by multitaper
        method. The default is 3.
    title : str, optional
        Text to use for the title.

    Returns
    -------
    None.

    """
    ch_names = epochs_right.ch_names
    PSD_right, freqs = epochs_right.compute_psd(method='multitaper',
                                                fmin=f_min, fmax=f_max,
                                                tmin=t_min,
                                                tmax=t_max).get_data(
                                                          return_freqs=True)
    PSD_left, freqs = epochs_left.compute_psd(method='multitaper',
                                              fmin=f_min, fmax=f_max,
                                              tmin=t_min, tmax=t_max).get_data(
                                                          return_freqs=True)
    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(20, 4.5))
    fig.subplots_adjust(top=0.82)
    axs[0].set_title('Right hand MI - C3', fontsize=18)
    plot_FOOOF(freqs, np.mean(PSD_right[:, ch_names.index('C3')], axis=0),
               ax=axs[0], add_legend=False)
    axs[0].grid(False)

    axs[1].set_title('Right hand MI - C4', fontsize=18)
    plot_FOOOF(freqs, np.mean(PSD_right[:, ch_names.index('C4')], axis=0),
               ax=axs[1], add_legend=False)
    axs[1].set_yticklabels([])
    axs[1].set_ylabel(None)
    axs[1].grid(False)

    axs[2].set_title('Left hand MI - C3', fontsize=18)
    plot_FOOOF(freqs, np.mean(PSD_left[:, ch_names.index('C3')], axis=0),
               ax=axs[2], add_legend=False)
    axs[2].set_yticklabels([])
    axs[2].set_ylabel(None)
    axs[2].grid(False)

    axs[3].set_title('Left hand MI - C4', fontsize=18)
    plot_FOOOF(freqs, np.mean(PSD_left[:, ch_names.index('C4')], axis=0),
               ax=axs[3])
    axs[3].set_yticklabels([])
    axs[3].set_ylabel(None)
    axs[3].grid(False)
    if title:
        fig.suptitle('PSD and FOOOF fitting - ' + title, fontsize=22, y=0.98)
