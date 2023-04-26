import sparse
import json
import numpy as np
import pandas as pd
import scipy as sp
import sys
import io
import os
import argparse

from ordinal import *
from evaluate import *

datasets = ["eicu", "mimic3"]
problems = ["ARF_4h", "ARF_12h", "Shock_4h", "Shock_12h"]

cwd = os.getcwd()

for d in datasets:
    for p in problems:
        s = sparse.load_npz(
            os.path.join(cwd,"/data/fiddle/FIDDLE_{dataset}/features/{problem}/s.npz".format(
                problem=p, dataset=d
            ))
        ).todense()
        x = sparse.load_npz(
            os.path.join(cwd,"/data/fiddle/FIDDLE_{dataset}/features/{problem}/X.npz".format(
                problem=p, dataset=d
            ))
        ).todense()

        s_feats = json.load(
            open(
                os.path.join(cwd,"/data/fiddle/FIDDLE_{dataset}/features/{problem}/s.feature_names.json".format(
                    problem=p, dataset=d
                )),
                "r",
            )
        )
        x_feats = json.load(
            open(
                os.path.join(cwd,+"/data/fiddle/FIDDLE_{dataset}/features/{problem}/X.feature_names.json".format(
                    problem=p, dataset=d
                )),
                "r",
            )
        )

        feat_idx, sets, freq_stats = get_feature_idx(dataset=d, x_feats=x_feats, s_feats=s_feats)
        end = x.shape[1]-1
        x_start = np.hstack([s,x[:,0,:]])
        x_end = np.hstack([s,x[:,end,:]])
        
        df_start = pd.DataFrame(ohe_to_ordinal(
            feature_sets=sets,
            feat_idx=feat_idx,
            x=x_start
        ))

        df_end = pd.DataFrame(ohe_to_ordinal(
            feature_sets=sets,
            feat_idx=feat_idx,
            x=x_end
        ))

        na_start = get_na_mask(x_start, freq_stats, feat_idx)
        na_end = get_na_mask(x_end, freq_stats, feat_idx)
        na_both = np.logical_and(na_start, na_end)
        
        results = {}
        results['df_start'] = df_start
        results['df_end'] = df_end
        results['na_mask'] = na_both
        results['feat_idx'] = feat_idx
        results['sets'] = sets
        results['freq_stats'] = freq_stats
        
        save_dir = os.path.join(cwd,"/data/fiddle/preprocessed/{dataset}/{problem}/".format(
                    problem=p, dataset=d
                ))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        output = open(
            os.path.join(cwd,"/data/fiddle/preprocessed/{dataset}/{problem}/results.pkl".format(
                    problem=p, dataset=d
                )), 
            'w')
        pickle.dump(results, output)
        output.close()
