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
