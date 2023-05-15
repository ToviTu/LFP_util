import pandas as pd
import numpy as np
import pickle
import sahara_work as sw
import glob
import statistics as st
import re
from datetime import datetime
import copy


def bout_duration(loc: str) -> list:
    # load files
    file_loc = loc
    files = np.sort(glob.glob(file_loc))
    sleep_states = np.array([])

    for a in files:
        sleep_states = np.append(sleep_states, np.load(a))

    # change QWake to AWake
    sleep_states[sleep_states == 5] = 1
    sleep_states[sleep_states == 4] = 2

    # get the index at which there is a state change
    diff = np.diff(sleep_states)
    index = np.nonzero(diff)[0]

    # modify the index array for duration calculation duration = index[i] - index[i-1]
    # add -1 to the start so that: first index - 0 + 1 = first index + 1 = index[1] - (-1) = index[1] + 1
    # add the length of the sleep_state array - 1 to the index array, this is the index of the last block
    index = np.insert(index, 0, -1)
    index = np.insert(index, len(index), len(sleep_states) - 1)

    # this give the difference calculation
    # use this to calculate how the states change
    extracted = np.extract(diff != 0, diff)
    arr = []
    arr.insert(len(arr), sleep_states[0])
    for i in extracted:
        arr.insert(len(arr), arr[len(arr) - 1] + i)

    # calculate the duration from the index array
    # and place the durating in the array of the corresponding state
    Wake_bout = []
    REM_bout = []
    NREM_bout = []
    # MA_bout = []

    for i in range(len(arr)):
        if arr[i] == 1:
            block = index[i + 1] - index[i]
            time = block * 4 / 60
            Wake_bout = np.insert(Wake_bout, len(Wake_bout), time)
        elif arr[i] == 2:
            block = index[i + 1] - index[i]
            time = block * 4 / 60
            NREM_bout = np.insert(NREM_bout, len(NREM_bout), time)

        elif arr[i] == 3:
            block = index[i + 1] - index[i]
            time = block * 4 / 60
            REM_bout = np.insert(REM_bout, len(REM_bout), time)

    average_Wake_bout = st.mean(Wake_bout)
    average_REM_bout = st.mean(REM_bout)
    average_NREM_bout = st.mean(NREM_bout)
    average = [average_NREM_bout, average_REM_bout, average_Wake_bout]
    return average


def microarousal_count(loc: str) -> float:
    file_loc = loc
    files = np.sort(glob.glob(file_loc))
    microarousal = []
    for a in files:
        sleep_states = np.array([])
        sleep_states = np.append(sleep_states, np.load(a))

        for i in range(len(sleep_states)):
            if sleep_states[i] == 5:
                if (
                    (2 in sleep_states[(i - 4) : i])
                    and (2 in sleep_states[i : (i + 4)])
                    and (3 not in sleep_states[(i - 4) : i])
                ):
                    sleep_states[i] = 4

        diff = np.diff(sleep_states)
        extracted = np.extract(diff != 0, diff)
        arr = []
        arr.insert(len(arr), sleep_states[0])
        for i in extracted:
            arr.insert(len(arr), arr[len(arr) - 1] + i)

        microarousal.insert(len(microarousal), arr.count(4))
    microarousal_avg = st.mean(microarousal)
    return microarousal_avg


def sw_states(file_loc: str) -> np.array:
    files = np.sort(glob.glob(file_loc))
    sleep_states = np.array([])
    for a in files:
        sleep_states = np.append(sleep_states, np.load(a))
    return sleep_states


def time_in_states(file_loc: str) -> list:
    """
    files = np.sort(glob.glob(file_loc))
    sleep_states = np.array([])

    for a in files:
        sleep_states = np.append(sleep_states, np.load(a))
    """

    sleep_states = sw_states(file_loc)
    REM = np.count_nonzero(sleep_states == 3) * 4 / 3600
    sleep_states[sleep_states == 5] = 1
    sleep_states[sleep_states == 4] = 2
    Wake = np.count_nonzero(sleep_states == 1) * 4 / 3600
    NREM = np.count_nonzero(sleep_states == 2) * 4 / 3600
    times = [NREM, REM, Wake]

    return times


def percentage_of_states_individual(loc: str) -> list:
    times = time_in_states(loc)
    sum = 0
    for i in times:
        sum += i
    percent_NREM = times[0] / sum * 100
    percent_REM = times[1] / sum * 100
    percent_Wake = times[2] / sum * 100
    percentages = [percent_NREM, percent_REM, percent_Wake]
    return percentages


def get_genotype(animal: str) -> str:
    return sw.get_genotype(animal.lower()).upper()


def parse_animalName(string: str) -> str:
    name_pattern = r"([A-Z]{3})0{2,3}(\d+)"
    return re.search(name_pattern, string).groups()


def parse_datetime(string: str) -> str:
    time_pattern = r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})"
    time = re.search(time_pattern, string).groups()
    mapper = lambda x: int(x)
    time = list(map(mapper, time))
    return datetime(*time)


def load_SWarray(loc: str) -> np.array:
    sleep_filez = glob.glob(loc)
    Sleep_states = []
    for a in sleep_filez:
        Sleep_states = np.append(Sleep_states, (np.load(a)))
    Sleep_states_by_sec = np.repeat(Sleep_states, 4)
    return Sleep_states_by_sec


def SW_analyze(loc: str, eFlag=False) -> pd.DataFrame:
    filez = glob.glob(loc)
    curTime = parse_datetime(filez[0])
    name = "".join(parse_animalName(loc))
    birthday = sw.get_birthday(name.lower())
    age = (curTime - birthday).days
    genotype = get_genotype(name.lower())

    pNREM, pREM, pWake = tuple(percentage_of_states_individual(loc))
    bNREM, bREM, bWake = tuple(bout_duration(loc))
    nMA = microarousal_count(loc)
    SWarray = load_SWarray(loc)
    return (
        pd.DataFrame(
            {
                "Animal": [name],
                "Age": [age],
                "Genotype": [genotype],
                "Birthday": [birthday],
                "%Wake": [pWake],
                "%NREM": [pNREM],
                "%REM": [pREM],
                "Wake Bout": [bWake],
                "NREM Bout": [bNREM],
                "REM Bout": [bREM],
                "#MA": [nMA],
                "Location": [loc],
                "RestartTime": [curTime],
                "hasError": [eFlag],
            }
        ),
        SWarray,
    )
