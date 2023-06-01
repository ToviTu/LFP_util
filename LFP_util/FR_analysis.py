import seaborn as sns
import numpy as np
import sys
from datetime import datetime
from datetime import time
import glob, os
import os.path as op
import sahara_work as sw
import pickle
import pandas as pd
import traceback
from datetime import date
from sys import platform
import sklearn
import re
import tqdm


def load_spkTime(files: list) -> list:
    ALL_CELLS_spk_times = []
    for idx, file in enumerate(files):
        neurons = np.load(file, allow_pickle=True)

        if idx == 0:
            start_time = dtify(neurons[0].rstart_time)

        fr_cutoff = 50
        good_cells = [
            neurons
            for neurons in neurons
            if neurons.quality < 3
            and neurons.plotFR(binsz=neurons.end_time, lplot=0, lonoff=0)[0][0]
            < fr_cutoff
            and neurons.presence_ratio() > 0.99
        ]
        for a in range(len(good_cells)):
            ALL_CELLS_spk_times.append(
                good_cells[a].spike_time_sec
                + ((dtify(neurons[0].rstart_time) - start_time).total_seconds())
            )
    return ALL_CELLS_spk_times


def n_getspikes(neuron_list, start=False, end=False, lonoff=1):

    """
    Extracts spiketimes to a list from neuron_list
    Unless otherwise specified start, end are in seconds
    n_getspikes(neuron_list, start=False, end=False)
    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    start : Start time (default self.start_time)
    end : End time (default self.end_time)
    lonoff : Apply on off times (default on, 1)
    Returns
    -------
    spiketimes_allcells : List of all spiketimes
    Raises
    ------
    ValueError if neuron list is empty
    See Also
    --------
    Notes
    -----
    Examples
    --------
    n_getspikes(neuron_list, start=False, end=False)
    """

    logger.info("Extracting spiketimes to a list from neuron_list")
    # check neuron_list is not empty
    if len(neuron_list) == 0:
        raise ValueError("Neuron list is empty")

    if start is False:
        start = neuron_list[0].start_time
    if end is False:
        end = neuron_list[0].end_time
    logger.info("start and end is %s and %s", start, end)

    if start < neuron_list[0].start_time:
        raise ValueError("start is less than neuron_list[0].start_time")
    if end > neuron_list[0].end_time:
        raise ValueError("end is greater than neuron_list[0].end_time")

    # Create empty list
    spiketimes_allcells = []

    # Loop through and get spike times
    for idx, neuron_l in enumerate(neuron_list):
        logger.debug("Getting spiketimes for cell %s", str(idx))

        # get spiketimes for each cell and append
        if lonoff:
            spiketimes = neuron_l.spike_time_sec_onoff
        else:
            spiketimes = neuron_l.spike_time_sec
        spiketimes = spiketimes[(spiketimes >= start) & (spiketimes <= end)]
        spiketimes_allcells.append(spiketimes)

    return spiketimes_allcells


def n_spiketimes_to_spikewords(
    neuron_list, binsz=0.02, start=False, end=False, binarize=0, lonoff=1
):
    """
        This function converts spiketimes to spikewords
        Unless otherwise specified binsz, start, end are in seconds
        n_spiketimes_to_spikewords(neuron_list, binsz=0.02, binarize=0)


    Examples
        --------
        n_spiketimes_to_spikewords(neuron_list, binsz=0.02, binarize=0)
    """

    # Constants
    conv_mills = 1000.0

    logger.info("Converting spiketime to spikewords")

    # check neuron_list is not empty
    if len(neuron_list) == 0:
        raise ValueError("Neuron list is empty")
    # check binsize is not less than 1ms
    if binsz < 0.001:
        raise ValueError("Bin size is less than 1millisecond")
    # binarize is only 0 or 1
    if binarize not in [0, 1]:
        raise ValueError("Binarize takes only values 0 or 1")

    # Get time
    if start is False:
        start = neuron_list[0].start_time
    if end is False:
        end = neuron_list[0].end_time
    logger.debug("start and end is %s and %s", start, end)

    # Get spiketime list
    spiketimes = n_getspikes(neuron_list, start=start, end=end, lonoff=lonoff)

    # convert time to milli seconds
    start = start * conv_mills
    end = end * conv_mills
    binsz = binsz * conv_mills

    # startime in bins
    binrange = np.arange(start, (end + binsz), binsz)
    n_cells = len(spiketimes)

    # initialize array
    spikewords_array = np.zeros([n_cells, binrange.shape[0] - 1])

    # loop over cells and find counts/binarize
    for i in range(n_cells):

        # spiketimes in seconds to ms
        spiketimes_cell = np.asarray(spiketimes)[i] * conv_mills
        counts, bins = np.histogram(spiketimes_cell, bins=binrange)

        # binarize the counts
        if binarize == 1:
            counts[counts > 0] = 1
        spikewords_array[i, :] = counts

    if binarize == 1:
        if (spikewords_array < 128).all():
            return spikewords_array.astype(np.int8)
        else:
            return spikewords_array.astype(np.int32)
    elif binarize == 0:
        if (spikewords_array < 2147483647).all():
            return spikewords_array.astype(np.int32)
        else:
            return spikewords_array.astype(np.int64)


def meanFR_by_interval(spk_time: list, SW_array: np.array, OFFSET=0) -> tuple:
    for neuron in spk_time:
        neuron -= OFFSET
        neuron = neuron[neuron >= 0]

    SW_changes = np.where(np.diff(SW_array) != 0)[0]
    all_meanFR = np.empty((len(spk_time), SW_changes.shape[0])).fill(np.nan)

    for idx, neuron in enumerate(spk_time):
        prev_time = 0
        count = 0
        for idy, end_time in enumerate(SW_changes):
            delta_time = end_time - prev_time
            prev_time = end_time
            while neuron[count] <= end_time:
                count += 1
            all_meanFR[idx, idy] = count / delta_time

    all_meanFR[all_meanFR == 0] = np.nan
    result = np.nanmean(all_meanFR, axis=1)

    return (result, SW_changes)


def meanFR_by_interval(spk_time: list, SW_array: np.array, OFFSET=0) -> tuple:
    for neuron in spk_time:
        neuron -= OFFSET
        neuron = neuron[neuron >= 0]

    SW_changes = np.where(np.diff(SW_array) != 0)[0]
    all_meanFR = np.empty((len(spk_time), SW_changes.shape[0]))
    all_meanFR.fill(np.nan)

    for idx, neuron in enumerate(spk_time):
        prev_time = 0
        prev_count = 0
        count = 0
        for idy, end_time in enumerate(SW_changes):
            delta_time = end_time - prev_time
            prev_time = end_time
            while count < len(neuron) and neuron[count] <= end_time:
                count += 1
            all_meanFR[idx, idy] = (count - prev_count) / delta_time
            prev_count = count

    all_meanFR[all_meanFR == 0] = np.nan
    result = np.nanmean(all_meanFR, axis=0)

    return (result, SW_changes)


def get_time_till_light_change(restart_time: datetime) -> tuple:
    if restart_time.time() < time(hour=7, minute=30):
        return (
            (
                datetime(
                    restart_time.year,
                    restart_time.month,
                    restart_time.day,
                    hour=7,
                    minute=30,
                )
                - restart_time
            ).total_seconds(),
            "dark",
        )
    elif restart_time.time() < time(hour=19, minute=30):
        return (
            (
                datetime(
                    restart_time.year,
                    restart_time.month,
                    restart_time.day,
                    hour=19,
                    minute=30,
                )
                - restart_time
            ).total_seconds(),
            "light",
        )
    else:
        return (
            (
                datetime(
                    restart_time.year,
                    restart_time.month,
                    restart_time.day + 1,
                    hour=7,
                    minute=30,
                )
                - restart_time
            ).total_seconds(),
            "dark",
        )


from LFP_util import general_tools as gt
from LFP_util import data_manage as dm

RAW_LFP_LOC = "/media/HlabShare/Clustering_Data/"


def get_FLPfilez(animal: str, restart_time: datetime) -> list:
    path = name_short2long(animal) + "/" + format_pathname(animal, str(restart_time))
    return gt.search_file("neurons_group0", RAW_LFP_LOC + "/" + path + "/*")


def sort_FLPfilez(filez: list) -> list:
    new_OS = gt.OS_deco(lambda x, y: int(x) < int(y), r"/(\d{1,3})_\d{1,3}/")
    filez_sorted = list(map(new_OS, filez))
    filez_sorted.sort()
    return list(map(lambda x: x.string, filez_sorted))


def get_FilesOfFirst24Hours(filez_sorted: list) -> list:
    idx = 0
    while "/24_" not in filez_sorted[idx]:
        idx += 1
    return filez_sorted[:idx]


def meanFRs_pipeline(animal: str, date: datetime) -> pd.DataFrame:
    all_animal, SW_arrays = dm.agg("Animal", "Restart_date", "SW_array")
    all_animal = all_animal[["Genotype", "RestartTime", "SW_id"]].dropna()
    animal_info = all_animal.query(f'Animal == "{animal}"')
    SW_arr_id = animal_info["SW_id"]
    SW_array = SW_arrays[int(SW_arr_id)]

    filez = get_FLPfilez(animal, date)
    filez = [file for file in filez if "probe1" in file]
    filez = sort_FLPfilez(filez)
    filez = get_FilesOfFirst24Hours(filez)

    neurons = load_spkTime(filez)
    offset = (
        date - dtify(re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", filez[0])[0])
    ).total_seconds()
    meanFRs, stateChanges = meanFR_by_interval(neurons, SW_array)

    firing_rates = pd.DataFrame(
        {"meanFR": meanFRs, "time": stateChanges, "state": SW_array[stateChanges]}
    )

    light_conditions = np.ones(firing_rates.shape[0])
    time_interval, light_condition = get_time_till_light_change(date)
    time_interval = int(time_interval)
    light_condition = 1 if light_condition == "light" else 0
    firing_rates["light_condition"] = light_conditions
    firing_rates.loc[
        firing_rates["time"] < time_interval, "light_condition"
    ] = light_condition
    firing_rates.loc[
        (firing_rates["time"] > time_interval)
        & (firing_rates["time"] < time_interval + 12 * 3600),
        "light_condition",
    ] = (
        1 - light_condition
    )
    firing_rates.loc[
        firing_rates["time"] > time_interval + 12 * 3600, "light_condition"
    ] = light_condition
    firing_rates["animal"] = [animal] * firing_rates.shape[0]

    return firing_rates


def meanFRs_pipeline_short(animal: str, filez: list, date: datetime) -> pd.DataFrame:
    all_animal, SW_arrays = dm.agg("Animal", "Restart_date", "SW_array")
    all_animal = all_animal[["Genotype", "RestartTime", "SW_id"]].dropna()
    animal_info = all_animal.query(f'Animal == "{animal}"')
    SW_arr_id = animal_info[(animal_info["RestartTime"] == date)]["SW_id"][0]
    SW_array = SW_arrays[int(SW_arr_id)]
    neurons = load_spkTime(filez)
    offset = (
        date - dtify(re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", filez[0])[0])
    ).total_seconds()
    meanFRs, stateChanges = meanFR_by_interval(neurons, SW_array)

    firing_rates = pd.DataFrame(
        {"meanFR": meanFRs, "time": stateChanges, "state": SW_array[stateChanges]}
    )

    light_conditions = np.ones(firing_rates.shape[0])
    time_interval, light_condition = get_time_till_light_change(date)
    time_interval = int(time_interval)
    light_condition = 1 if light_condition == "light" else 0
    firing_rates["light_condition"] = light_conditions
    firing_rates.loc[
        firing_rates["time"] < time_interval, "light_condition"
    ] = light_condition
    firing_rates.loc[
        (firing_rates["time"] > time_interval)
        & (firing_rates["time"] < time_interval + 12 * 3600),
        "light_condition",
    ] = (
        1 - light_condition
    )
    firing_rates.loc[
        firing_rates["time"] > time_interval + 12 * 3600, "light_condition"
    ] = light_condition
    firing_rates["animal"] = [animal] * firing_rates.shape[0]

    return firing_rates
