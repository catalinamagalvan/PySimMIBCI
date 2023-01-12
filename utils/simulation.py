# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 19:31:54 2021

@author: catal
"""

import numpy as np
from numpy.matlib import repmat
from scipy import signal
from mne import read_labels_from_annot
from mne.label import select_sources


def set_peak_amplitudes(MI_tasks, user_peak_params, reduction=0.5):
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


def generate_what(MI_tasks, events, user_params, MI_duration, sfreq, N_trials,
                  reduction):

    peak_params = set_peak_amplitudes(MI_tasks, user_params['peak_params'],
                                      reduction=reduction)
    aperiodic_params = user_params['aperiodic_params']
    labels_names = ['G_precentral-lh', 'G_precentral-rh']
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
    Create events matrix.

    Parameters
    ----------
    events_info : dict
        label: events label
        duration: events duration in ms
    n_trials : int
        number of trials.
    include_rest : bool, optional
        include or not rest segments. The default is True.
    rest_duration: int
        rest segment duration in ms
    sample_freq:
        sample frequency in Hz

    Returns
    -------
    events : array [n_events x 3]
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


def generate_where(subject, subjects_dir):
    """
    Generate 

    Parameters
    ----------
    subject : TYPE
        DESCRIPTION.
    subjects_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    where : list of mne.Label
        DESCRIPTION.

    """
    labels_names = ['G_precentral-lh', 'G_precentral-rh']
    annot = 'aparc.a2009s'
    where = dict()
    for label in labels_names:
        label_tmp = read_labels_from_annot(subject, annot,
                                           subjects_dir=subjects_dir,
                                           regexp=label, verbose=False)
        label_tmp = label_tmp[0]
        label_tmp = select_sources(subject, label_tmp, location='center',
                                   extent=30, subjects_dir=subjects_dir)
        where[label] = label_tmp
    return where


def theta_activity(N_samples, sfreq, increase=1):
    f_theta_min = 4
    f_theta_max = 8
    # All sources are independent gaussian
    rng = np.random.default_rng()
    non_filtered_activity = rng.normal(loc=7, size=N_samples)
    # Filter to a narrow band
    filter_order = 2
    sos = signal.butter(filter_order, (f_theta_min, f_theta_max),
                        'bandpass', fs=sfreq, output='sos')
    source_activity = signal.sosfilt(sos, non_filtered_activity)
    # Increase theta activity in the fatigue condition
    if increase != 1:
        mask = np.linspace(0, increase, len(source_activity))
        source_activity *= mask
    return 1e-8*source_activity


def alpha_activity(N_samples, sfreq, increase=1):
    f_alpha_min = 8
    f_alpha_max = 13
    # All sources are independent gaussian
    rng = np.random.default_rng()
    non_filtered_activity = rng.normal(loc=7.8, size=N_samples)
    # Filter to a narrow band
    filter_order = 2
    sos = signal.butter(filter_order, (f_alpha_min, f_alpha_max),
                        'bandpass', fs=sfreq, output='sos')
    source_activity = signal.sosfilt(sos, non_filtered_activity)
    # Increase theta activity in the fatigue condition
    if increase != 1:
        mask = np.linspace(0, increase, len(source_activity))
        source_activity *= mask
    return 1e-8*source_activity


def add_basal_theta_alpha(source_simulator, fatigue_start, subject,
                          subjects_dir):
    """
    Function to add fatigue effects to source_simulator object.

    Mental fatigue is associated with increased power in frontal theta (θ) and
    parietal alpha (α) EEG rhythms. Returns modified SourceSimulator object.

    Parameters
    ----------
    source_simulator : mne.simulation.SourceSimulator object
        DESCRIPTION.
    fatigue_start : int
        Time fatigue starts in % of the total length of the session.

    Returns
    -------
    None.

    """
    annot = 'aparc.a2009s'

    # Add alpha and theta activiy in alert condition
    event = np.array([0, 0, 4])[np.newaxis]
    N_samples_alert = int(fatigue_start*source_simulator.n_times)

    # Add alpha activity
    what = alpha_activity(N_samples_alert,
                          sfreq=int(1/source_simulator._tstep))
    label_tmp = read_labels_from_annot(subject, annot,
                                       subjects_dir=subjects_dir, hemi='rh',
                                       regexp='G_and_S_paracentral',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=5, subjects_dir=subjects_dir)
    source_simulator.add_data(label_tmp, what, event)

    label_tmp = read_labels_from_annot(subject, annot,
                                       subjects_dir=subjects_dir, hemi='lh',
                                       regexp='G_and_S_paracentral',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=5, subjects_dir=subjects_dir)
    source_simulator.add_data(label_tmp, what, event)

    # Add theta activity
    what = theta_activity(N_samples_alert,
                          sfreq=int(1/source_simulator._tstep))
    label_tmp = read_labels_from_annot(subject, annot,
                                       subjects_dir=subjects_dir, hemi='rh',
                                       regexp='G_front_sup',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10, subjects_dir=subjects_dir)
    source_simulator.add_data(label_tmp, what, event)

    label_tmp = read_labels_from_annot(subject, annot,
                                       subjects_dir=subjects_dir, hemi='lh',
                                       regexp='G_front_sup',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10, subjects_dir=subjects_dir)
    source_simulator.add_data(label_tmp, what, event)

    return source_simulator


def add_fatigue_effect(source_simulator, fatigue_start, subject, subjects_dir,
                       annot='aparc.a2009s'):
    """
    Function to add fatigue effects to source_simulator object.

    Mental fatigue is associated with increased power in frontal theta (θ) and
    parietal alpha (α) EEG rhythms. Returns modified SourceSimulator object.

    Parameters
    ----------
    source_simulator : mne.simulation.SourceSimulator object
        DESCRIPTION.
    fatigue_start : int
        Time fatigue starts in % of the total length of the session.

    Returns
    -------
    None.

    """

    # Add alpha and theta activiy in fatigue condition
    N_samples_alert = int(fatigue_start*source_simulator.n_times)
    event = np.array([N_samples_alert, 0, 5])[np.newaxis]
    N_samples_fatigue = source_simulator.n_times - N_samples_alert

    what = alpha_activity(N_samples_fatigue,
                          sfreq=int(1/source_simulator._tstep),
                          increase=30*1.5)
    label_tmp = read_labels_from_annot(subject, annot,
                                       subjects_dir=subjects_dir, hemi='rh',
                                       regexp='G_and_S_paracentral',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10, subjects_dir=subjects_dir)
    source_simulator.add_data(label_tmp, what, event)

    label_tmp = read_labels_from_annot(subject, annot,
                                       subjects_dir=subjects_dir, hemi='lh',
                                       regexp='G_and_S_paracentral',
                                       verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10, subjects_dir=subjects_dir)
    source_simulator.add_data(label_tmp, what, event)

    # Add theta activity
    what = theta_activity(N_samples_fatigue,
                          sfreq=int(1/source_simulator._tstep),
                          increase=40*1.3)
    label_tmp = read_labels_from_annot(subject, annot,
                                       subjects_dir=subjects_dir,
                                       regexp='G_front_sup',
                                       hemi='lh', verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10, subjects_dir=subjects_dir)
    source_simulator.add_data(label_tmp, what, event)

    label_tmp = read_labels_from_annot(subject, annot,
                                       subjects_dir=subjects_dir,
                                       regexp='G_front_sup',
                                       hemi='rh', verbose=False)
    label_tmp = label_tmp[0]
    label_tmp = select_sources(subject, label_tmp, location='center',
                               extent=10, subjects_dir=subjects_dir)
    source_simulator.add_data(label_tmp, what, event)

    return source_simulator
