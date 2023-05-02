import numpy as np
import dice_ml
from tqdm import tqdm, trange
import pandas as pd
import scipy as sp

def compare_cf(d, diff, mad, ftv, tol=1e-5, n_avg=5):
   d = d[ftv]
   comp = diff[ftv]
   distance = []
   for i in range(comp.shape[0]):
      distance.append(np.sum(np.abs(d - comp.iloc[i,:])/(mad+tol)))
   distance = np.sort(distance)
   return np.sum(distance[:n_avg])/n_avg

def evaluate_rank(df, model, diff, ftv, cont=[], n_cfs=3, n_avg=5):
    m = dice_ml.Model(model=model, backend="sklearn")
    d = dice_ml.Data(dataframe=df, continuous_features=cont, outcome_name='label')
    exp = dice_ml.Dice(d, m, method="random")
    all_comparisons = []
    top_comparisons = []
    top_cfs = []
    bot_cfs = []
    failed = []
    few = []
    few_cfs = []
    changes = pd.DataFrame(columns = ['idx', 'cf_idx'] + ftv)
    mad = sp.stats.median_abs_deviation(diff[ftv])
    for i in trange(df.shape[0]):
        query = df.iloc[i:(i+1),:]
        query = query.drop(columns=['label'])
        try:
            e = exp.generate_counterfactuals(
                query,
                total_CFs=n_cfs, 
                desired_class="opposite",
                features_to_vary=ftv)
        except:
            failed.append(i)
            continue
        cfs = e.cf_examples_list[0].final_cfs_df
        if cfs is None:
            failed.append(i)
            continue
        elif cfs.shape[0] < n_cfs:
            few.append(i)
            few_cfs.append(cfs)
            continue
        comparisons = []
        for j in trange(n_cfs):
            d = cfs.iloc[j,:-1].apply(pd.to_numeric) - query.iloc[0,:].apply(pd.to_numeric)
            c = compare_cf(
                d=d,
                diff=diff,
                mad=mad,
                ftv=ftv,
                n_avg=n_avg
            )
            row = []
            changes.loc[len(changes)-1] = [int(i), int(j)] + list(np.zeros(len(ftv)))
            # changes.iloc[changes.shape[0]-1,:][ftv] = d
            all_comparisons.append(c)
            comparisons.append(c)
        top = np.argmin(comparisons)
        bot = np.argmax(comparisons)
        top_comparisons.append(comparisons[top])

        top_cfs.append(cfs.iloc[top,:])
        bot_cfs.append(cfs.iloc[bot,:])
    results = {
        'all' : all_comparisons,
        'top' : top_comparisons,
        'top_cfs' : top_cfs,
        'bot_cfs' : bot_cfs,
        'failed' : failed,
        'few' : few,
        'few_cfs' : few_cfs,
        'changes' : changes
    }
    return results