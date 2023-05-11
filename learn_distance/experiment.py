import sparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import scipy as sp
import sys
import io
import os
import argparse
from tqdm import tqdm, trange

import dice_ml
from dice_ml.utils import helpers
import matplotlib.pyplot as plt
import seaborn as sns

from ordinal import *
from evaluate import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='Dataset for experiment', choices=['eicu', 'mimic3'], type=str)
parser.add_argument('-p', '--problem', help='Prediction problem for experiment', choices=['ARF_4h', 'ARF_12h', 'Shock_4h', 'Shock_12h', 'mortality_48h'], type=str)
parser.add_argument('-m', '--model', help='sklearn model for experiment', choices=['rf', 'logistic', 'boosting'], type=str)
parser.add_argument('-l', '--load', action='store_true', help='load preprocessed data')

args = parser.parse_args()

problem = args.problem
dataset = args.dataset
model = args.model

print('Loading data...')

cwd = os.getcwd()

if args.load:
    '''
    # file path is not correct, need to update
    
    with open(cwd+"/data/fiddle/preprocessed/{dataset}/{problem}_features.pkl", 'rb') as file:
        features = pickle.load(file)
    '''

else:
    s = sparse.load_npz(cwd+'/data/fiddle/FIDDLE_{dataset}/features/{problem}/s.npz'.format(problem=problem, dataset=dataset)).todense()
    x = sparse.load_npz(cwd+'/data/fiddle/FIDDLE_{dataset}/features/{problem}/X.npz'.format(problem=problem, dataset=dataset)).todense()

    s_feats = json.load(open(cwd+'/data/fiddle/FIDDLE_{dataset}/features/{problem}/s.feature_names.json'.format(problem=problem, dataset=dataset), 'r'))
    x_feats = json.load(open(cwd+'/data/fiddle/FIDDLE_{dataset}/features/{problem}/X.feature_names.json'.format(problem=problem, dataset=dataset), 'r'))
    
    end = x.shape[1]-1
    x_start = np.hstack([s,x[:,0,:]])
    x_end = np.hstack([s,x[:,end,:]])
    feats = s_feats + x_feats
    
    feat_idx, sets, freq_stats = get_feature_idx(
        dataset=dataset,
        x_feats=x_feats,
        s_feats=s_feats)
    
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


y = pd.read_csv(cwd+'/data/fiddle/FIDDLE_{dataset}/population/{problem}.csv'.format(problem=problem, dataset=dataset))

df_start = df_start.iloc[na_both,:]
df_end = df_end.iloc[na_both,:]
label = y[problem[:problem.find('_')+'_LABEL']]

print('Training model...')

xtr, xte, ytr, yte = train_test_split(df_start, y_both, stratify=y_both, random_state=123)
if model == 'rf':
    boundary = RandonForestClassifier()
elif model == 'logistic':
    boundary = LogisticRegression(solver='liblinear', penalty='l1')
elif model == 'boosting':
    boundary = GradientBoostingClassifier()

boundary.fit(xtr, ytr)

train = pd.DataFrame(xtr, columns = df_start.columns)
train['label'] = ytr

start = train.drop(columns=['label'])

test = pd.DataFrame(xte, columns=df_start.columns)
test['label'] = yte

pos = np.where(yte==1)
diff = df_end - df_start
test_pos = test.iloc[pos]
test_pos[freq_stats] = test_pos[freq_stats].apply(pd.to_numeric)

print('Evaluating counterfactuals...')

eval_df = test_pos
results = evaluate_rank(
    df = eval_df,
    model = boundary,
    diff = diff,
    n_cfs=10,
    n_avg=3,
    ftv = freq_stats,
    cont = freq_stats
)

print('Saving results...')

savename = dataset + '_' +  problem + '_' + 'result.pkl'
with open(savename, 'rb') as file:
    results = pickle.load(file)
