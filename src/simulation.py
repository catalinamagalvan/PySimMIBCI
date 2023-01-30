"""
# Author: Catalina M. Galvan <cgalvan@santafe-conicet.gov.ar>
"""

import numpy as np
from numpy.matlib import repmat
from scipy import signal
from scipy.io import savemat
from mne import read_labels_from_annot, make_forward_solution
from mne.label import select_sources
from mne.datasets import fetch_fsaverage
from mne.simulation import SourceSimulator
import os


def set_up_source_forward(subject, info):
    """
    Set up source space and compute forward solution

    Parameters
    ----------
    subject : str
        The FreeSurfer subject name.
    info : instance of MNE Info
        Corresponding MNE Info object.

    Returns
    -------
    fwd : Instance of MNE Forward
        The forward solution.
    source_simulator : Instance of MNE SourceSimulator
        Corresponding object to generate simulated Source Estimates.

    """
    # Download head model (fsaverage) files
    fs_dir = fetch_fsaverage(verbose=False)
    src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    fwd = make_forward_solution(info, trans=subject, src=src,
                                bem=bem, eeg=True, mindist=5.0)
    # Here, :class:`~mne.simulation.SourceSimulator` is used, which allows to
    # specify where (label), what (source_time_series), and when (events) an
    # event type will occur.
    src = fwd['src']
    source_simulator = SourceSimulator(src, tstep=1/info['sfreq'])
    return fwd, source_simulator


def set_peak_amplitudes(MI_tasks, user_peak_params, reduction=0.5):
    """
    Set up alpha peak amplitudes for right vs left hand MI tasks.

    Parameters
    ----------
    MI_tasks : list of str
        List of MI tasks names.
    user_peak_params : dict
        User-specific peak parameters.
    reduction : float, optional
        Float in the range (0, 1). The percentage of desynchronization for
        alpha ERDs. The default is 0.5.

    Returns
    -------
    simulation_peak_params : TYPE
        DESCRIPTION.

    """
    labels_names = user_peak_params.keys()
    simulation_peak_params = {}
    for label in labels_names:
        simulation_peak_params[label] = {}
        for task in MI_tasks:
            simulation_peak_params[label][task] = user_peak_params[label]
    # Fixed amplitude
    simulation_peak_params['G_precentral-lh']['MI/left'][1] = 0.4
    simulation_peak_params['G_precentral-rh']['MI/right'][1] = 0.4
    simulation_peak_params['G_precentral-lh']['MI/right'][1] = 0.4*(
        1-reduction)
    simulation_peak_params['G_precentral-rh']['MI/left'][1] = 0.4*(1-reduction)
    return simulation_peak_params


def generate_what_failed(MI_tasks, events, user_params, MI_duration, sfreq,
                         N_trials, reduction, p_failed):
    """
    Generate waveform (what) dictionary with the inclusion of a percentage of
    trials without ERD.

    Parameters
    ----------
    MI_tasks : list of str
        List of MI tasks names.
    events : array of int, shape (n_events, 3)
        Events associated to the waveform(s) to specify when the activity
        should occur.
    user_params :  dict
        User-specific parameters.
    MI_duration : int
        MI trials duration in ms.
    sfreq : int
        The sampling frequency.
    N_trials : int
        The number of trials to generate.
    reduction : float
        Float in the range (0, 1). The percentage of desynchronization for
        alpha ERDs.
    p_failed : float
        Float in the range (0, 1). The percentage of trials without ERD.

    Returns
    -------
    MI_activity_epoched : dict
        The keys of the dictionary are the different labels. Each element in
        the dictionary is another dictionary that has the different MI tasks
        as keys. For each label and MI task there is an array, of shape
        (n_events, n_times) that corresponds to the waveform describing the
        activity on that label for that MI task and for each of the events.

    """
    peak_params = set_peak_amplitudes(MI_tasks, user_params['peak_params'],
                                      reduction=reduction)
    aperiodic_params = user_params['aperiodic_params']
    # Generate bad trials vector
    bad_trials_vector = np.zeros(np.size(events, 0))
    rng = np.random.default_rng()
    # Select only MI positions (not rest)
    MI_positions = np.logical_or(events[:, 2] == 1, events[:, 2] == 2).nonzero(
        )[0]
    MI_positions_failed = rng.choice(MI_positions, int(p_failed*N_trials/2))
    bad_trials_vector[MI_positions_failed] = 1

    # Load the 4 necessary label names.
    labels_names = list(peak_params.keys())
    N_samples_trial = int(np.round(MI_duration/1000*sfreq))
    N_samples = int(events[-1, 0]) + N_samples_trial
    N_samples_class = int(N_samples//len(MI_tasks))
    N_trials_class = N_trials//len(MI_tasks)

    MI_activity = dict()
    # Create raw
    for label in labels_names:
        MI_activity[label] = dict()
        for task in MI_tasks:
            # Right activity
            peak = peak_params[label][task]
            MI_activity[label][task] = np.zeros(N_samples_class)
            non_filtered_activity = np.random.randn(1, N_samples_class)
            # aperiodic component in linear space
            offset = aperiodic_params[0]/2
            exponent = aperiodic_params[1]
            cf = peak[0]
            aperiodic_f = 10**offset/(cf**exponent)
            pw = aperiodic_f*10**(peak[1])
            bw = peak[2]
            # Filter
            sos = signal.butter(2, (cf-bw/2, cf+bw/2),
                                'bandpass', fs=sfreq, output='sos')
            aux = signal.sosfilt(sos, non_filtered_activity[0])
            MI_activity[label][task] += 1e-4*pw*aux
            # No modulation activity
            if peak[1] == 0.4*(1-reduction):
                MI_activity[label][task+'_wrong'] = np.zeros(
                    N_samples_class)
                non_filtered_activity = np.random.randn(1,
                                                        N_samples_class)
                # aperiodic component in linear space
                aperiodic_f = 10**offset/(cf**exponent)
                pw = aperiodic_f*10**(peak[1]/(1-reduction))
                bw = peak[2]
                # Filter
                sos = signal.butter(2, (cf-bw/2, cf+bw/2),
                                    'bandpass', fs=sfreq, output='sos')
                aux = signal.sosfilt(sos, non_filtered_activity[0])
                MI_activity[label][task+'_wrong'] += 1e-4*pw*aux

    MI_activity_epoched = dict()
    for label in labels_names:
        MI_activity_epoched[label] = dict()
        for task_ID, task in enumerate(MI_tasks, 1):
            peak = peak_params[label][task]
            bad_trials_vector_tmp = bad_trials_vector[np.where(events[:, 2] ==
                                                               task_ID)[0]]
            MI_activity_epoched[label][task] = np.empty((N_trials_class,
                                                         N_samples_trial))
            for t, _ in enumerate(bad_trials_vector_tmp):
                if bad_trials_vector_tmp[t] == 1 and peak[1] == 0.4*(
                        1-reduction):
                    MI_activity_epoched[label][task][t] = MI_activity[label][
                        task+'_wrong'][t*N_samples_trial:(t+1)*N_samples_trial]
                else:
                    MI_activity_epoched[label][task][t] = MI_activity[label][
                        task][t*N_samples_trial:(t+1)*N_samples_trial]
    return MI_activity_epoched


def generate_what(MI_tasks, events, user_params, MI_duration, sfreq, N_trials,
                  reduction):
    """
    Generate waveform (what) dictionary.

    Parameters
    ----------
    MI_tasks : list of str
        List of MI tasks names.
    events : array of int, shape (n_events, 3)
        Events associated to the waveform(s) to specify when the activity
        should occur.
    user_params :  dict
        User-specific parameters.
    MI_duration : int
        MI trials duration in ms.
    sfreq : int
        The sampling frequency.
    N_trials : int
        The number of trials to generate.
    reduction : float
        Float in the range (0, 1). The percentage of desynchronization for
        alpha ERDs.

    Returns
    -------
    MI_activity_epoched : dict
        The keys of the dictionary are the different labels. Each element in
        the dictionary is another dictionary that has the different MI tasks
        as keys. For each label and MI task there is an array, of shape
        (n_events, n_times) that corresponds to the waveform describing the
        activity on that label for that MI task and for each of the events.

    """
    peak_params = set_peak_amplitudes(MI_tasks, user_params['peak_params'],
                                      reduction=reduction)
    aperiodic_params = user_params['aperiodic_params']
    labels_names = peak_params.keys()
    N_samples_trial = int(np.round(MI_duration/1000*sfreq))
    N_samples = int(events[-1, 0]) + N_samples_trial
    N_samples_class = int(N_samples//len(MI_tasks))
    N_trials_class = N_trials//len(MI_tasks)
    MI_activity = dict()

    # Create raw
    for label in labels_names:
        MI_activity[label] = dict()
        for task in MI_tasks:
            # Only one peak
            peak = peak_params[label][task]
            MI_activity[label][task] = np.zeros(N_samples_class)
            non_filtered_activity = np.random.randn(1, N_samples_class)
            cf = peak[0]
            # aperiodic component in linear space
            offset = aperiodic_params[0]/2
            exponent = aperiodic_params[1]
            aperiodic_f = 10**offset/(cf**exponent)
            # Filter in alpha band
            pw = aperiodic_f*10**(peak[1])
            bw = peak[2]
            sos = signal.butter(2, (cf-bw/2, cf+bw/2),
                                'bandpass', fs=sfreq, output='sos')
            aux = signal.sosfilt(sos, non_filtered_activity[0])
            MI_activity[label][task] += 1e-4*pw*aux

    MI_activity_epoched = dict()
    for label in labels_names:
        MI_activity_epoched[label] = dict()
        for task in MI_tasks:
            MI_activity_epoched[label][task] = np.empty((N_trials_class,
                                                         N_samples_trial))
            for t in range(N_trials_class):
                MI_activity_epoched[label][task][t] = MI_activity[label][
                    task][t*N_samples_trial:(t+1)*N_samples_trial]
    return MI_activity_epoched


def generate_when(events_info, N_trials, sfreq, include_rest=False,
                  rest_duration=2000, include_baseline=False,
                  baseline_duration=5000):
    """
    Create events matrix (when).

    Parameters
    ----------
    events_info : dict
        label: events label
        duration: events duration in ms.
    N_trials : int
        number of trials.
    sfreq : int
        the sampling frequency in Hz.
    include_rest : bool, optional
        whether to include or not rest segments. The default is False.
    rest_duration : int
        rest segments duration in ms if include_rest=True.
    include_baseline : bool, optional
        whether to include or not baseline segment. The default is False.
   baseline_duration : int
        baseline segment duration in ms if include_baseline=True.

    Returns
    -------
    events : array of int, shape (n_events, 3)
        Events associated to the waveform that specify when the activity
        should occur.
        first column: sample numbers corresponding to each event
        last column: event ID

    """
    N_classes = len(events_info)
    trials_ID = repmat(np.arange(start=1, stop=N_classes+1),
                       1, int(N_trials/N_classes))
    trials_ID = np.transpose(trials_ID)[:, 0]
    trials_ID = np.random.permutation(trials_ID)

    cnt = 0
    current_time = 0

    if include_baseline:
        if include_rest:
            events = np.zeros((2*N_trials+1, 3), dtype=np.int32)
        else:
            events = np.zeros((N_trials+1, 3), dtype=np.int32)
    else:
        if include_rest:
            events = np.zeros((2*N_trials, 3), dtype=np.int32)
        else:
            events = np.zeros((N_trials, 3), dtype=np.int32)

    if include_baseline:
        events[cnt, 0] = int(current_time)
        events[cnt, 2] = int(N_classes+1)  # rest ID
        duration_tmp = round(baseline_duration/1000*sfreq)
        current_time += duration_tmp
        cnt += 1

    for i, event_ID in enumerate(trials_ID):
        events[cnt, 0] = int(current_time)
        events[cnt, 2] = int(event_ID)
        number = events_info[event_ID-1]['duration']/1000*sfreq
        duration_tmp = round(number)

        current_time += duration_tmp
        cnt += 1

        if include_rest:
            events[cnt, 0] = int(current_time)
            events[cnt, 2] = int(N_classes+1)
            duration_tmp = round(rest_duration/1000*sfreq)
            current_time += duration_tmp
            cnt += 1
    return events


def generate_where(subject):
    """
    Generate spatial labels to hand MI tasks.

    Parameters
    ----------
    subject : str
        The FreeSurfer subject name.

    Returns
    -------
    where : list of mne.Label
        List of labels that contains the selected sources in the hand motor
        cortex at left and right hemispheres.

    """
    labels_names = ['G_precentral-lh', 'G_precentral-rh']
    annot = 'aparc.a2009s'
    where = dict()
    for label in labels_names:
        label_tmp = read_labels_from_annot(subject, annot,
                                           regexp=label, verbose=False)
        label_tmp = label_tmp[0]
        label_tmp = select_sources(subject, label_tmp, location='center',
                                   extent=30)
        where[label] = label_tmp
    return where


def theta_activity(N_samples, sfreq, increase=1):
    """
    Generate theta band activity.

    Parameters
    ----------
    N_samples : int
        Number of samples to generate.
    sfreq : int
        Sampling frequency in Hertz.
    increase : float, optional
        Multiplier of theta activity. The default is 1.

    Returns
    -------
    source_theta_activity : numpy array, (N_samples,)
        The source activity in theta band.

    """
    # All sources are  gaussian
    rng = np.random.default_rng()
    non_filtered_activity = rng.normal(loc=3.5, size=N_samples)
    # Filter to a narrow band
    sos = signal.butter(2, (4, 8),
                        'bandpass', fs=sfreq, output='sos')
    source_theta_activity = signal.sosfilt(sos, non_filtered_activity)
    # Increase theta activity in the fatigue condition
    if increase != 1:
        mask = np.linspace(0, increase, len(source_theta_activity))
        source_theta_activity *= mask
    source_theta_activity = 1e-8*source_theta_activity
    return source_theta_activity


def alpha_activity(N_samples, sfreq, increase=1):
    """
    Generate alpha band activity.

    Parameters
    ----------
    N_samples : int
        Number of samples to generate.
    sfreq : int
        Sampling frequency in Hertz.
    increase : float, optional
        Multiplier of theta activity. The default is 1.

    Returns
    -------
    source_alpha_activity : numpy array, (N_samples,)
        The source activity in alpha band.

    """
    # All sources are gaussian
    rng = np.random.default_rng()
    non_filtered_activity = rng.normal(loc=4, size=N_samples)
    # Filter to a narrow band
    sos = signal.butter(2, (8, 13),
                        'bandpass', fs=sfreq, output='sos')
    source_alpha_activity = signal.sosfilt(sos, non_filtered_activity)
    # Increase alpha activity in the fatigue condition
    if increase != 1:
        mask = np.linspace(0, increase, len(source_alpha_activity))
        source_alpha_activity *= mask
    return 1e-8*source_alpha_activity


def add_basal_theta_alpha(source_simulator, fatigue_start, subject):
    """
    Add basal theta band activity to source_simulator object.
    Mental fatigue is associated with increased power in frontal theta (θ) and
    parietal alpha (α) EEG rhythms. Returns modified SourceSimulator object.

    Parameters
    ----------
    source_simulator : Instance of MNE SourceSimulator object
        The mne.simulation.SourceSimulator object to modify.
    fatigue_start : int
        Time fatigue starts in % of the total length of the session.

    Returns
    -------
    source_simulator : Instance of MNE SourceSimulator object.
        The mne.simulation.SourceSimulator instance modified in-place.

    """
    annot = 'aparc.a2009s'
    event = np.array([0, 0, 4])[np.newaxis]
    N_samples_alert = int(fatigue_start*source_simulator.n_times)

    # Add alpha activiy in alert condition
    what = alpha_activity(N_samples_alert,
                          sfreq=int(1/source_simulator._tstep))
    label_tmp = read_labels_from_annot(subject, annot, hemi='rh',
                                       regexp='G_and_S_paracentral',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=5)
    source_simulator.add_data(label_tmp, what, event)

    label_tmp = read_labels_from_annot(subject, annot, hemi='lh',
                                       regexp='G_and_S_paracentral',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=5)
    source_simulator.add_data(label_tmp, what, event)

    # Add theta activiy in alert condition
    what = theta_activity(N_samples_alert,
                          sfreq=int(1/source_simulator._tstep))
    label_tmp = read_labels_from_annot(subject, annot, hemi='rh',
                                       regexp='G_front_sup',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10)
    source_simulator.add_data(label_tmp, what, event)

    label_tmp = read_labels_from_annot(subject, annot, hemi='lh',
                                       regexp='G_front_sup',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10)
    source_simulator.add_data(label_tmp, what, event)

    return source_simulator


def add_fatigue_effect(source_simulator, fatigue_start, subject,
                       annot='aparc.a2009s'):
    """
    Add fatigue effects to source_simulator object.
    Mental fatigue is associated with increased power in frontal theta (θ) and
    parietal alpha (α) EEG rhythms. Returns modified SourceSimulator object.

    Parameters
    ----------
    source_simulator : Instance of MNE SourceSimulator object
        The mne.simulation.SourceSimulator object to modify.
    fatigue_start : int
        Time fatigue starts in % of the total length of the session.

    Returns
    -------
    source_simulator : Instance of MNE SourceSimulator object.
        The mne.simulation.SourceSimulator instance modified in-place.

    """

    N_samples_alert = int(fatigue_start*source_simulator.n_times)
    event = np.array([N_samples_alert, 0, 5])[np.newaxis]
    # Add alpha activiy in fatigue condition
    N_samples_fatigue = source_simulator.n_times - N_samples_alert

    what = alpha_activity(N_samples_fatigue,
                          sfreq=int(1/source_simulator._tstep),
                          increase=30*1.5)
    label_tmp = read_labels_from_annot(subject, annot, hemi='rh',
                                       regexp='G_and_S_paracentral',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10)
    source_simulator.add_data(label_tmp, what, event)

    label_tmp = read_labels_from_annot(subject, annot, hemi='lh',
                                       regexp='G_and_S_paracentral',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10)
    source_simulator.add_data(label_tmp, what, event)

    # Add theta activiy in fatigue condition
    what = theta_activity(N_samples_fatigue,
                          sfreq=int(1/source_simulator._tstep),
                          increase=40*1.3)
    label_tmp = read_labels_from_annot(subject, annot,
                                       regexp='G_front_sup',
                                       hemi='lh', verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10)
    source_simulator.add_data(label_tmp, what, event)

    label_tmp = read_labels_from_annot(subject, annot,
                                       regexp='G_front_sup',
                                       hemi='rh', verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10)
    source_simulator.add_data(label_tmp, what, event)

    return source_simulator


def save_mat_simulated_data(raw, events, spath, fname):
    """
    Save simulated data in a .mat file compatible with FBCNet Toolbox
    functions.

    Parameters
    ----------
    raw : Instance of MNE.io.Raw
        Raw data to save.
    events : array of int, shape (n_events, 3)
        The array of events. The first column contains the event time in
        samples, with first_samp included. The third column contains the event
        id.
    spath : str
        Saving path.
    fname : str
        Saving filename.

    Returns
    -------
    None.

    """
    # Save signals in .mat
    raw_data = raw.get_data().T
    pos = events[:, 0]
    pos = pos[events[:, -1] != 3]
    mrk = {"pos": pos, "y": events[:, -1][events[:, -1] != 3]}
    nfo = {"fs": raw.info['sfreq'], "clab": raw.ch_names}
    mdict = {"cnt": raw_data, "mrk": mrk, "nfo": nfo}
    if not os.path.exists(spath):
        os.makedirs(spath)
    savemat(os.path.join(spath, fname), mdict)
