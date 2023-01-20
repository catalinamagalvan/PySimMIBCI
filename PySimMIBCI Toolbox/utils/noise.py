# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:10:38 2021

@author: catal
"""
import numpy as np
from numpy import einsum
from mne import make_ad_hoc_cov, Covariance
from mne.io import BaseRaw
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.simulation.evoked import _validate_type, pick_info
from mne.io.pick import (channel_indices_by_type, _DATA_CH_TYPES_SPLIT,
                         pick_types)
from mne.utils import (logger, _check_preload)
from mne._ola import _Interp2
from mne.bem import fit_sphere_to_headshape, make_sphere_model
from mne.source_space import setup_volume_source_space
from mne.utils import (check_random_state, _validate_type, _check_preload)
from mne.simulation.raw import (_check_head_pos, _SimForwards)
import colorednoise as cn


def make_noise_cov(info):
    noise_cov = make_ad_hoc_cov(info)
    names = noise_cov.ch_names
    bads = []
    projs = noise_cov.__getitem__('projs')
    nfree = noise_cov.nfree

    ch_indices = channel_indices_by_type(info)
    allowed_types = _DATA_CH_TYPES_SPLIT
    picks = list()
    for this_type in allowed_types:
        picks += ch_indices[this_type]

    chs = [info['chs'][pick] for pick in picks]
    n_channels = len(chs)
    pos = np.empty((n_channels, 3))
    for ci, ch in enumerate(chs):
        pos[ci] = ch['loc'][:3]

    cross_spec = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            d = np.linalg.norm(pos[i]-pos[j], 2)
            cross_spec[i, j] = np.exp(-d**2)
            cross_spec[j, i] = np.exp(-d**2)
    cov_cross_spec = Covariance(cross_spec, names, bads, projs, nfree,
                                verbose=None)
    return cov_cross_spec


def _generate_pink_noise(info, cov, n_samples, picks=None, exponent=1.7,
                         random_state=None):
    """Create spatially colored and temporally IIR-filtered noise."""
    dim = cov['dim']

    noise_matrix = np.empty((dim, n_samples))

    for n, noise_channel in enumerate(noise_matrix):
        noise_matrix[n] = cn.powerlaw_psd_gaussian(exponent, n_samples,
                                                   random_state=random_state)

    # in this case the cross spectrum is same for each freq, and we add an
    # 1/freq power law here.
    # n.b. In total, this probably resembles generating pink noise and
    # enforcing a spatial cov structure for it.
    # [E,D] = eig(spectralModel.crossSpectrum);
    # D: matriz diagonal que contiene los autovalores
    # E: matriz que contiene los autovalores
    D, E = np.linalg.eig(cov['data'])
    D = np.diag(D)

    # Filter the white noise to give it the desired covariance structure
    colorer_2 = np.real(E @ np.sqrt(D)).T  # igual
    noise_matrix = noise_matrix.T @ colorer_2
    return noise_matrix.T


def add_aperiodic_activity(inst, allow_subselection=True, exponent=1.7,
                           offset=(np.sqrt(1/0.005)), random_state=None):
    """Create noise as a multivariate Gaussian.
    The spatial covariance of the noise is given from the cov matrix.
    Parameters
    ----------
    inst : instance of Evoked, Epochs, or Raw
        Instance to which to add noise.
    cov : instance of Covariance
        The noise covariance.
    iir_filter : None | array-like
        IIR filter coefficients (denominator).
    %(random_state)s
    %(verbose)s
    Returns
    -------
    inst : instance of Evoked, Epochs, or Raw
        The instance, modified to have additional noise.
    Notes
    -----
    Only channels in both ``inst.info['ch_names']`` and
    ``cov['names']`` will have noise added to them.
    This function operates inplace on ``inst``.
    .. versionadded:: 0.18.0
    """
    cov = make_noise_cov(inst.info)
    offset = offset/2

    _validate_type(cov, Covariance, 'cov')
    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked),
                   'inst', 'Raw, Epochs, or Evoked')
    _check_preload(inst, 'Adding noise')
    data = inst._data
    assert data.ndim in (2, 3)
    if data.ndim == 2:
        data = data[np.newaxis]
    # Subselect if necessary
    info = inst.info
    info._check_consistency()
    if allow_subselection:
        use_chs = list(set(info['ch_names']) & set(cov['names']))
        picks = np.where(np.in1d(info['ch_names'], use_chs))[0]
        logger.info('Adding noise to %d/%d channels (%d channels in cov)'
                    % (len(picks), len(info['chs']), len(cov['names'])))
        info = pick_info(inst.info, picks)
        info._check_consistency()
    for epoch in data:
        noise_component = np.zeros_like(epoch[picks])
        noise_component_temp = _generate_pink_noise(info, cov, epoch.shape[1],
                                                    exponent=exponent,
                                                    random_state=random_state)
        noise_component += 1e-1*np.real(noise_component_temp)
        epoch[picks] += 10**offset*noise_component
    return inst


def add_eye_movement(raw, head_pos=None, interp='cos2', n_jobs=1,
                     random_state=None, verbose=None):
    """Add eye-movement artifact to raw data.

    Parameters
    ----------
    raw : instance of Raw
        The raw instance to modify.
    %(head_pos)s
    %(interp)s
    %(n_jobs)s
    %(random_state)s
        The random generator state used for blink, ECG, and sensor noise
        randomization.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The instance, modified in place.

    See Also
    --------
    add_chpi
    add_ecg
    add_eog
    add_noise
    simulate_raw

    Notes
    -----
    The blink artifacts are generated by:
    1. Random activation times are drawn from an inhomogeneous poisson
       process whose blink rate oscillates between 4.5 blinks/minute
       and 17 blinks/minute based on the low (reading) and high (resting)
       blink rates from [1]_.
    2. The activation kernel is a 250 ms Hanning window.
    3. Two activated dipoles are located in the z=0 plane (in head
       coordinates) at ±30 degrees away from the y axis (nasion).
    4. Activations affect MEG and EEG channels.

    The scale-factor of the activation function was chosen based on
    visual inspection to yield amplitudes generally consistent with those
    seen in experimental data. Noisy versions of the activation will be
    stored in the first EOG channel in the raw instance, if it exists.

    References
    ----------
    .. [1] Bentivoglio et al. "Analysis of blink rate patterns in normal
           subjects" Movement Disorders, 1997 Nov;12(6):1028-34.
    """
    return add_exg(raw, 'eye_movement', head_pos, interp, n_jobs,
                   random_state)


def add_exg(raw, kind, head_pos, interp, n_jobs, random_state):
    assert isinstance(kind, str) and kind in ('ecg', 'blink', 'eye_movement')
    _validate_type(raw, BaseRaw, 'raw')
    _check_preload(raw, 'Adding %s noise ' % (kind,))
    rng = check_random_state(random_state)
    info, times, first_samp = raw.info, raw.times, raw.first_samp
    data = raw._data
    meg_picks = pick_types(info, meg=True, eeg=False, exclude=())
    meeg_picks = pick_types(info, meg=True, eeg=True, exclude=())
    R, r0 = fit_sphere_to_headshape(info, units='m', verbose=False)[:2]
    bem = make_sphere_model(r0, head_radius=R,
                            relative_radii=(0.97, 0.98, 0.99, 1.),
                            sigmas=(0.33, 1.0, 0.004, 0.33), verbose=False)
    trans = None
    dev_head_ts, offsets = _check_head_pos(head_pos, info, first_samp, times)
    if kind == 'eye_movement':
        # place dipoles at 45 degree angles in z=0 plane
        exg_rr = np.array([[np.cos(np.pi / 3.), np.sin(np.pi / 3.), 0.],
                           [-np.cos(np.pi / 3.), np.sin(np.pi / 3), 0.]])
        exg_rr /= np.sqrt(np.sum(exg_rr * exg_rr, axis=1, keepdims=True))
        exg_rr *= 0.96 * R
        exg_rr += r0
        # oriented upward
        nn = np.array([[0., 0., 1.], [0., 0., 1.]])
        # Draw occurence times from a Poisson process
        timeNow = 0
        endTime = times[-1]
        eventFreq = 0.2
        randomMaxduration = 500  # in ms
        timeline = np.zeros(np.size(times))
        while timeNow < endTime:
            nextTime = timeNow + (-np.log(1-rng.uniform()))/eventFreq
            if nextTime < endTime:
                event_latency = np.int(nextTime*info['sfreq'])
                event_duration = np.int(np.random.uniform() * randomMaxduration
                                        / 1000 * info['sfreq'])
                timeline[event_latency:event_latency+event_duration] = 1
            timeNow = nextTime
        exg_data = np.random.randn(np.size(times)) * timeline[np.newaxis,
                                                              :] * 1e-8
        picks = meeg_picks
    del meg_picks, meeg_picks
    src = setup_volume_source_space(pos=dict(rr=exg_rr, nn=nn),
                                    sphere_units='mm')

    used = np.zeros(len(raw.times), bool)
    get_fwd = _SimForwards(
        dev_head_ts, offsets, info, trans, src, bem, 0.005, n_jobs, picks)
    interper = _Interp2(offsets, get_fwd, interp)
    proc_lims = np.concatenate([np.arange(0, len(used), 10000), [len(used)]])
    for start, stop in zip(proc_lims[:-1], proc_lims[1:]):
        fwd, _ = interper.feed(stop - start)
        data[picks, start:stop] += einsum(
            'svt,vt->st', fwd, exg_data[:, start:stop])
        assert not used[start:stop].any()
        used[start:stop] = True
    assert used.all()
