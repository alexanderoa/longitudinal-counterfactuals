import sparse
import json
import numpy as np
import pandas as pd
import scipy as sp
import sys
import io
import argparse

from ordinal import *
from evaluate import *

datasets = ["eicu", "mimic3"]
problems = ["ARF_4h", "ARF_12h", "Shock_4h", "Shock_12h"]

for d in datasets:
    for p in problems:
        s = sparse.load_npz(
            "../data/fiddle/FIDDLE_{dataset}/features/{problem}/s.npz".format(
                problem=p, dataset=d
            )
        ).todense()
        x = sparse.load_npz(
            "../data/fiddle/FIDDLE_{dataset}/features/{problem}/X.npz".format(
                problem=p, dataset=d
            )
        ).todense()

        s_feats = json.load(
            open(
                "../data/fiddle/FIDDLE_{dataset}/features/{problem}/s.feature_names.json".format(
                    problem=p, dataset=d
                ),
                "r",
            )
        )
        x_feats = json.load(
            open(
                "../data/fiddle/FIDDLE_{dataset}/features/{problem}/X.feature_names.json".format(
                    problem=p, dataset=d
                ),
                "r",
            )
        )

        features = get_feature_idx(dataset=dataset, x_feats=x_feats, s_feats=s_feats)

        output = open("../data/fiddle/preprocessed/{dataset}/{problem}_features.pkl")
        pickle.dump(features, output)
        output.close()
