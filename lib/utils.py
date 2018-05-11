import pandas as  pd
import numpy as np


def avg_importance(sa, sb):
    sc = sa.add(sb, fill_value=None).dropna() / 2
    sd = sa.add(sb, fill_value=0).drop(sc.index)
    return sc.append(sd)
