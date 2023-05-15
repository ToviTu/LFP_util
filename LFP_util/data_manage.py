import pandas as pd
import numpy as np
import pickle
from LFP_util import all_data

cache = None


def view_new():
    if cache is None:
        print("There is no cached changes")
    else:
        return cache


def save_new():
    if cache is None:
        print("There is no cached changes")
    else:
        with open(
            "/hlabhome/wg-mjames/git/LFP_util/LFP_util/SW_data.pickle", "wb"
        ) as f:
            pickle.dump(cache, f)


def format(formatted_frame):
    pa = pd.melt(
        formatted_frame,
        id_vars=["Animal", "Age"],
        value_vars=["%Wake", "%NREM", "%REM"],
        value_name="percent",
        var_name=["State"],
        ignore_index=False,
    )
    pa.loc[:, "State"] = pa.loc[:, "State"].apply(lambda x: x[1:])
    ba = pd.melt(
        formatted_frame,
        id_vars=["Animal", "Age"],
        value_vars=["Wake Bout", "NREM Bout", "REM Bout"],
        value_name="Bout",
        var_name=["State"],
        ignore_index=False,
    )
    ba.loc[:, "State"] = ba.loc[:, "State"].apply(lambda x: x[:-5])
    pa.set_index(["Animal", "Age", "State"], inplace=True)
    ba.set_index(["Animal", "Age", "State"], inplace=True)
    sa = pa.join(ba)

    ra = formatted_frame.drop(
        columns=["%Wake", "%NREM", "%REM"]
        + ["Wake Bout", "NREM Bout", "REM Bout"]
        + ["Birthday", "Genotype"]
    ).set_index(["Animal", "Age"])
    aa = formatted_frame[["Animal", "Genotype", "Birthday"]].set_index(["Animal"])

    return (sa, ra, aa)


def append(entry: pd.DataFrame, new_array: np.array) -> dict:
    """
    Append one one entry at a time. The function automatically formats the dataframe and save it to a temporary cache.

    Parameter:
    entry: A dataframe with only one row that cannot has nan in it
    new_array: the sw_scoring array for this entry

    Return:
    the new dataframe
    """
    SW_stat = accessor("SW_stat")
    animal = accessor("Animal")
    restart_date = accessor("Restart_date")
    sw_array = accessor("SW_array")

    format_frame = pd.DataFrame(
        {
            "Animal": [],
            "Age": [],
            "Genotype": [],
            "Birthday": [],
            "%Wake": [],
            "%NREM": [],
            "%REM": [],
            "Wake Bout": [],
            "NREM Bout": [],
            "REM Bout": [],
            "#MA": [],
            "Location": [],
            "RestartTime": [],
            "hasError": [],
            "SW_id": [],
        }
    )
    formatted_frame = pd.concat([format_frame, entry])
    formatted_frame.iloc[0, -1] = len(sw_array)
    sw_array.append(new_array)
    if np.any(formatted_frame.isnull()):
        raise Exception("Missing value detected")
    if formatted_frame.shape[1] != 15:
        raise Exception("Info column does not match")
    if formatted_frame.shape[0] != 1:
        raise Exception("Appending more than one entry is currently not supported")

    sa, ra, aa = format(formatted_frame)

    animal_new = pd.concat([animal, aa])
    restart_date_new = pd.concat([restart_date, ra])
    SW_stat_new = pd.concat([SW_stat, sa])

    if not (
        animal_new.index.is_unique
        and restart_date_new.index.is_unique
        and SW_stat_new.index.is_unique
    ):
        raise Exception("Duplicate index")

    new_dic = dict()

    new_dic["SW_array"] = sw_array
    new_dic["SW_stat"] = SW_stat_new
    new_dic["Animal"] = animal_new
    new_dic["Restart_date"] = restart_date_new

    global cache
    cache = new_dic
    return cache


def agg(*fields) -> pd.DataFrame:
    """
    This function pull data from the database and aggregate them into a dataframe in long form.
    Notice that there might be missing data if the record itself is imcomplete.

    Tip: If a user friendy wide form datafame is desired, use the following example command:
        #to convert SW_stat from long form to wide from
        df.reset_index().pivot(index=['Animal', 'Age'], columns=['State'], values=['percent', 'Bout'])

    Parameter:
    fields: names of the interested fields
            (select from ['Animal', 'SW_stat', 'Restart_date', 'SW_array'])
            If "all" is passed in, all fields would be selected
            Not a list! Strings plz

    Return:
    result: A dataframe of the selected fields;
            None if no fields is selected
    sw_array: (Optional) A list of sleep_wake_state arrays for all restart day;
              Only returned when 'SW_array' is passed as an argument
    """
    if "all" in fields:
        fields = ["Animal", "SW_stat", "Restart_date", "SW_array"]

    result = None
    if not fields:
        return None
    else:
        result = accessor(fields[0])
        for each in fields[1:]:
            if each != "SW_array":
                result = result.join(accessor(each))

    if "SW_array" in fields:
        return result, accessor("SW_array")
    return result


def accessor(field):
    if cache is None:
        return all_data[field]
    return cache[field]
