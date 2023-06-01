import re
import numpy as np
import pandas as pd
import glob


def search_file(kw: str, path: str) -> list:
    search_result = []
    stack = glob.glob(path + "/*")
    while len(stack) > 0:
        cur = stack.pop()
        cur_stuff = glob.glob(cur + "/*")
        for each in cur_stuff:
            if kw in each:
                search_result.append(each)
            else:
                stack.append(each)
    return search_result


class ordered_str:
    string = None
    rule = None
    to_compare = None

    def __init__(self, s: str, rule, reg):
        self.string = s
        self.to_compare = re.search(reg, s).groups()[0]
        self.rule = rule

    def __lt__(self, other):
        return self.rule(self.to_compare, other.to_compare)

    def __hash___(self) -> int:
        return hash(self.to_compare)


def OS_deco(rule, reg):
    def new_func(string):
        return ordered_str(string, rule, reg)

    return new_func


def format_pathname(name: str, date: str) -> str:
    """
    Concatenate an animal name and a date with certain format
    Example: (CAF50, 2020-12-02) -> caf50_12022020
    """
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date)
    return (
        name.lower()
        + "_"
        + date_match.groups()[1]
        + date_match.groups()[2]
        + date_match.groups()[0]
    )


def name_short2long(name: str) -> str:
    """
    Convert a short animal name to its long form.
    Example: CAF050 -> CAF00050
    """
    match = re.match(r"([A-Z]{3})(\d{2,3})", name)
    return (
        "000".join(match.groups())
        if len(match.groups()[1]) == 2
        else "00".join(match.groups())
    )


def dtify(datetimestring):
    """
    Usage:
    dtify('2020-12-28_11-16-26') --> returns datetime.datetime(2020, 12, 28, 11, 16, 26)
    """
    import datetime

    try:
        dtobj = datetime.datetime.strptime(datetimestring, "%Y%m%dT%H%M%S")
    except:
        dtobj = datetime.datetime.strptime(datetimestring, "%Y-%m-%d_%H-%M-%S")
    return dtobj
