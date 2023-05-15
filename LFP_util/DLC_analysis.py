import pandas as pd
import numpy as np
import glob
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


def load_dlc(animal: str) -> pd.DataFrame:
    """
    Nirali's function to load DLC data with filtering

    Arguments:
        animal (str): The animal's name

    Returns:
        pd.Dataframe
            A dataframe of various statistics about the animal
    """

    lstream = 0
    overallDataFrame = pd.DataFrame()
    video_dir = "/media/bs001r/Nirali_Project/" + animal + "/"
    video_dir2 = "/media/bs001r/watchtower/" + animal + "/"

    # defining cage coordinates and area (or these)
    pathUL = "/media/bs001r/Nirali_Project/" + animal + "/" + animal + "UL.npy"
    pathUR = "/media/bs001r/Nirali_Project/" + animal + "/" + animal + "UR.npy"
    pathLR = "/media/bs001r/Nirali_Project/" + animal + "/" + animal + "LR.npy"
    pathLL = "/media/bs001r/Nirali_Project/" + animal + "/" + animal + "LL.npy"
    UL = np.load(pathUL)
    UR = np.load(pathUR)
    LR = np.load(pathLR)
    LL = np.load(pathLL)
    XLenUpper = UR[0] - UL[0]
    XLenLower = LR[0] - LL[0]
    YLenLeft = LL[1] - UL[1]
    YLenRight = LR[1] - UR[1]
    area1 = (LL[0] - UL[0]) * YLenLeft * 0.5
    area2 = ((YLenRight + YLenLeft) / 2) * XLenUpper
    area3 = (LR[0] - UR[0]) * YLenRight * 0.5
    areaCage = area1 + area2 + area3
    areasAndCoords = pd.read_pickle(
        "/media/bs001r/Nirali_Project/" + animal + "areasAndCoords" + ".pickle"
    )

    # this loads our lovely little main data frame and filters out the points with a likelihood below 0.9 and points that are not within cage boundaries
    bp4 = pd.read_pickle(
        "/media/bs001r/Nirali_Project/" + animal + "overallBody" + ".pickle"
    )
    bp4_filtered = bp4[(bp4.likelihood.values > 0.9)]
    bp4_filtered = bp4_filtered[(bp4_filtered.xIncm.values <= 17)]
    bp4_filtered = bp4_filtered[(bp4_filtered.yIncm.values <= 19)]
    bp4_filtered = bp4_filtered[(bp4_filtered.xIncm.values >= 0)]
    bp4_filtered = bp4_filtered[(bp4_filtered.yIncm.values >= 0)]
    bp4_xf = bp4_filtered.x.values
    bp4_yf = bp4_filtered.y.values
    bp4_tf = bp4_filtered.time.values
    rev_bp4_tf = bp4_tf[::-1]
    return bp4_filtered


def load_SW(dir: str) -> np.ndarray:
    sleep_filez = np.sort(glob.glob(f"{dir}/*SleepStates*MA*"))
    Sleep_states = []
    for a in sleep_filez:
        Sleep_states = np.append(Sleep_states, (np.load(a)))

    Sleep_states_by_sec = np.repeat(Sleep_states, 4)
    print(f"The SW array has a shape of {str(Sleep_states_by_sec.shape[0])}")
    return Sleep_states_by_sec


def load_raw_DLC(animal: str, date: str) -> pd.DataFrame:
    """
    James's function to load raw dlc data that are newly generated.

    Arguments:
        animal (str): Name of the animal. e.g. CAF52
        date (str): The specific restart date. e.g. 20201123

    Returns:
        pd.DataFrame
            A dataframe of the dlc data including various body parts
    """
    video_dir = "/media/bs001r/James_AD_Project/CAF00052/20201123/"
    dlcfiles = np.sort(glob.glob(video_dir + "*.h5"))
    dlcdata_list = []

    for ind, dlcfile in enumerate(dlcfiles):
        # print('Working on video ' + str(dlcfile.split('/')[-1]))
        dlcdata_hour = pd.read_hdf(dlcfile)
        dlcdata_list.append(dlcdata_hour)

    # concatenates all the features into dlcdata_block
    dlcdata_block = pd.concat(dlcdata_list, axis=0)
    return dlcdata_block


def likelihood_clean(data: pd.DataFrame, threshold=0.95) -> pd.DataFrame:
    """
    Replace (x,y) with low likelihood with NAN and then fill the NANs by interpolating
    """
    likelihood_mask = data.likelihood > 0.95
    data.loc[~likelihood_mask, "x"] = np.NAN
    data.loc[~likelihood_mask, "y"] = np.NAN
    return data.interpolate()


def get_speed(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the speed from a dataframe of (x,y) coordinates
    """
    data["Speed"] = np.sqrt((data.x.diff() ** 2 + data.y.diff() ** 2)) * 30
    return data


def moving_mean(data: pd.DataFrame, window=30) -> pd.DataFrame:
    """
    Calculate the rolling mean
    """
    data["Speed_filt"] = data.Speed.rolling(window).mean()
    return data


pipe = Pipeline(
    steps=[
        (
            "Interpolation",
            FunctionTransformer(likelihood_clean),
        ),
        ("Speed", FunctionTransformer(get_speed)),
        (
            "mean_filt",
            FunctionTransformer(moving_mean),
        ),
    ]
)
