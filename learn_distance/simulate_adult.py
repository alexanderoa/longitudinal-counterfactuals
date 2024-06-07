import pandas as pd
import numpy as np
from tqdm import tqdm
import secrets

def simulate_adult(data, time=1, thres=[0.5, 0.9], seed=None):
    res = data.copy()
    ed_table = data.groupby(['education', 'education-num']).size()
    ed_table = ed_table.index.to_frame()
    if seed is None:
        seed = secrets.randbits(128)
    rng = np.random.default_rng(seed)
    res['age'] = res['age'] + time
    changed = []
    higher_ed = []
    for i in range(len(res)):
        rand = rng.random()
        if rand < thres[0]:
            continue
        if (rand > thres[1]) and (res.iloc[i,:]['education'] != 'Doctorate'):
            ed_num = res.iloc[i,:]['education-num'] + 1
            idx = np.where(ed_table['education-num']==ed_num)
            res.loc[i,'education'] = ed_table['education'].iloc[idx].values[0]
            res.loc[i,'education-num'] = ed_num
            higher_ed.append(i)
        ed_idx = data.index[np.where(data['education'] == res.iloc[i,:]['education'])[0]]
        new = rng.choice(ed_idx)
        shift = ['hours-per-week', 'occupation', 'workclass', 'capital-gain', 'capital-loss']
        res.loc[i, shift] = data.loc[new, shift]
        changed.append(i)
    return res, changed, higher_ed

def simulate_adult_range(data, time=10, thres=[0.5, 0.8], seed=None):
    res = data.copy()
    ed_table = data.groupby(['education', 'education-num']).size()
    ed_table = ed_table.index.to_frame()
    if seed is None:
        seed = secrets.randbits(128)
    rng = np.random.default_rng(seed)
    changed = []
    higher_ed = []
    for i in range(len(res)):
        rand = rng.random()
        if rand < thres[0]:
            res.loc[i,'age'] = res.loc[i,'age'] + rng.integers(time)
            continue
        if (rand > thres[1]) and (res.iloc[i,:]['education'] != 'Doctorate'):
            ed_num = res.iloc[i,:]['education-num'] + 1
            idx = np.where(ed_table['education-num']==ed_num)
            res.loc[i,'education'] = ed_table['education'].iloc[idx].values[0]
            res.loc[i,'education-num'] = ed_num
            higher_ed.append(i)
        check = np.all(np.stack([
            (data['education'] == res.iloc[i,:]['education']),
            (data['age'] >= res.iloc[i,:]['age']),
            (data['age'] <= res.iloc[i,:]['age'] + time)
        ]), axis=0)
        ed_idx = data.index[check]
        if ed_idx.size == 0:
            continue
        new = rng.choice(ed_idx)
        shift = ['age', 'hours-per-week', 'occupation', 'workclass', 'capital-gain', 'capital-loss']
        res.loc[i, shift] = data.loc[new, shift]
        changed.append(i)
    return res, changed, higher_ed