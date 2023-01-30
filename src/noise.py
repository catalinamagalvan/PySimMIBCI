"""
# Author: Catalina M. Galvan <cgalvan@santafe-conicet.gov.ar>
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
from mne.utils import (check_random_state)
from mne.simulation.raw import (_check_head_pos, _SimForwards)
import colorednoise as cn


def make_noise_cov(info):
    """
    Generate noise covariance matrix based on electrode distances: closer
    electrodes have stronger correlation (gaussian weighted distance).


    Parameters
    ----------
    info : instance of MNE Info
        Corresponding MNE Info object.

    Returns
    -------
    cov_cross_spec : instance of MNE Covariance
        DESCRIPTION.

    """
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
    # Weight the spectrum so that channels which are close by are more
    # correlated (gaussian weighted distance)
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
    """
    Create spatially colored and temporally Gaussian distributed noise with a
    power law spectrum with specified exponent.

    Parameters
    ----------
    info : instance of MNE Info
        Corresponding MNE Info object.
    cov : instance of MNE Covariance
        Corresponding MNE Covariance object.
    n_samples : int
        Number of samples to generate the noise.
    picks : str | array_like | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. In lists, channel type strings (e.g., ['meg',
        'eeg']) will pick channels of those types, channel name strings (e.g.,
        ['MEG0111', 'MEG2623'] will pick the given channels. Can also be the
        string values “all” to pick all channels, or “data” to pick data
        channels. None (default) will pick all channels. Note that channels in
        info['bads'] will be included if their names or indices are explicitly
        provided.The default is None.
    exponent : float, optional
        The exponent (k) of the 1/(f**k) noise. The default is 1.7.
    random_state : int | numpy.integer | numpy.random.Generator |
                   numpy.random.RandomState | None
        NumPy's random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly. The
        default is None.

    Returns
    -------
    noise_matrix: numpy.array
        The noise matrix, of dimension n channels x n samples.

    """
    dim = cov['dim']

    noise_matrix = np.empty((dim, n_samples))

    for n, noise_channel in enumerate(noise_matrix):
        noise_matrix[n] = cn.powerlaw_psd_gaussian(exponent, n_samples,
                                                   random_state=random_state)

    # in this case the cross spectrum is same for each freq, and we add an
    # 1/freq power law here.
    # n.b. In total, this probably resembles generating pink noise and
    # enforcing a spatial cov structure for it.
    D, E = np.linalg.eig(cov['data'])
    D = np.diag(D)

    # Filter the white noise to give it the desired covariance structure
    colorer_2 = np.real(E @ np.sqrt(D)).T  # igual
    noise_matrix = noise_matrix.T @ colorer_2
    return noise_matrix.T


def add_aperiodic_activity(inst, allow_subselection=True, exponent=1.7,
                           offset=0, random_state=None):
    """
    Applies spatially colored and temporally Gaussian distributed noise.

    Parameters
    ----------
    inst : instance of Evoked, Epochs, or Raw
        Instance to which add noise.
    allow_subselection : bool, optional
        Whether to allow channel subselection based on info. The default is
        True.
    exponent : float, optional
        The exponent (k) of the 10**b 1/(f**k) noise. The default is 1.7.
    offset : float, optional
        The offset (b) in the 10**b 1/(f**k) noise. The default is 0.
    random_state : int | numpy.integer | numpy.random.Generator |
                   numpy.random.RandomState | None
        NumPy's random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly. The
        default is None.

    Returns
    -------
    inst : instance of Evoked, Epochs, or Raw
        Instance modified in-place.

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
    head_pos: None | str | dict | tuple | array
        Name of the position estimates file. Should be in the format of the
        files produced by MaxFilter. If dict, keys should be the time points
        and entries should be 4x4 dev_head_t matrices. If None, the original
        head position (from info['dev_head_t']) will be used. If tuple, should
        have the same format as data returned by head_pos_to_trans_rot_t. If
        array, should be of the form returned by mne.chpi.read_head_pos().
    interp: str
        Either ‘hann’, ‘cos2’ (default), ‘linear’, or ‘zero’, the type of
        forward-solution interpolation to use between forward solutions at
        different head positions.
    n_jobs: int | None
        The number of jobs to run in parallel. If -1, it is set to the number
        of CPU cores. Requires the joblib package. None (default) is a marker
        for ‘unset’ that will be interpreted as n_jobs=1 (sequential execution)
        unless the call is performed under a joblib.parallel_backend() context
        manager that sets another value for n_jobs.
    random_state: None | int | instance of RandomState
        A seed for the NumPy random number generator (RNG). If None (default),
        the seed will be obtained from the operating system (see RandomState
        for details), meaning it will most likely produce different output
        every time this function or method is run. To achieve reproducible
        results, pass a value here to explicitly initialize the RNG with a
        defined state. The random generator state used for noise randomization.
    verbose: bool | str | int | None
        Control verbosity of the logging output. If None, use the default
        verbosity level. See the logging documentation and mne.verbose() for
        details. Should only be passed as a keyword argument.

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
    The eye-movement artifacts are generated by:
    1. Random activation times are drawn from an homogeneus poisson.
    2. Two activated dipoles are located in the z=0 plane (in head
       coordinates) at ±30 degrees away from the y axis (nasion), as in
       mne.simulation.add_eog function.
    3. Activations affect MEG and EEG channels.

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
    """
    Modification of add_exg function in mne.simulation.raw.
    """
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
