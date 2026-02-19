# This file is part of minlp-toolbox/misc
# Copyright (C) 2026  Andrea Ghezzi
# SPDX-License-Identifier: GPL-3.0-or-later


from sys import argv
from matplotlib import pyplot as plt
from matplotlib import lines
from matplotlib import colors
import matplotlib
import pandas as pd
import numpy as np
import os
from datetime import datetime


def to_float(val):
    """Convert to float."""
    if type(val) == str:
        if (
            ("Objective" in val)
            or ("feasible" in val)
            or ("Error" in val)
            or ("Calling" in val)
            or ("g_val" in val)
            or ("CRASH" in val)
            or ("FAILED" in val)
            or ("empty" in val)
            or ("basic_string" in val)
            or ("No objective" in val)
            or ("Suffix values" in val)
            or ("for indices" in val)
            or ("has no attribute" in val)
        ):
            return np.inf
        elif val == "-inf":
            return np.inf
        elif val == "NAN":
            return np.inf
        else:
            val = float(val)
            if val > 1e20:
                return np.inf
            else:
                return float(val)
    else:
        return float(val)


if __name__ == "__main__":

    if len(argv) != 3:
        print("Usage: python create_plot.py <data_file.csv> <key>")
        print("key: cvx or noncvx")
        exit(1)

    data = pd.read_csv(argv[1])
    SAVE_DIRECTORY = os.path.dirname(argv[1])
    key = argv[2]
    assert key == "cvx" or key == "noncvx"
    total_entries = data.shape[0]

    col_name = f"{key}_sbmiqp.calc_time"
    data.set_index("name", inplace=True)
    sbmiqp_wall_time = data[[col_name]]
    sbmiqp_wall_time[col_name] = sbmiqp_wall_time[col_name].map(to_float)
    sbmiqp_wall_time[col_name] = sbmiqp_wall_time[col_name].clip(lower=0, upper=300)

    sbmiqp_wall_time.to_json(
        os.path.join(
            os.path.dirname(os.path.abspath(argv[0])), f"wall_time_{key}_sbmiqp.json"
        )
    )
