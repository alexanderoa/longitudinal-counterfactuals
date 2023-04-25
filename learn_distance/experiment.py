import sparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import scipy as sp
import sys
import io
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

s = sparse.load_npz('../data/fiddle/FIDDLE_{dataset}/features/{problem}/s.npz'.format(problem=problem, dataset=dataset)).todense()
x = sparse.load_npz('../data/fiddle/FIDDLE_{dataset}/features/{problem}/X.npz'.format(problem=problem, dataset=dataset)).todense()

s_feats = json.load(open('../data/fiddle/FIDDLE_{dataset}/features/{problem}/s.feature_names.json'.format(problem=problem, dataset=dataset), 'r'))
x_feats = json.load(open('../data/fiddle/FIDDLE_{dataset}/features/{problem}/X.feature_names.json'.format(problem=problem, dataset=dataset), 'r'))

y = pd.read_csv('../data/fiddle/FIDDLE_{dataset}/population/{problem}.csv'.format(problem=problem, dataset=dataset))

