import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import scipy as sp
from statsmodels.distributions.empirical_distribution import ECDF
import random
import secrets
from tqdm import tqdm
import sys
import argparse
import pickle
import time

from geco import *
from simulate_adult import *

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threshold", help="Threshold for prediction target")
parser.add_argument(
    "-l", "--longitudinal", help="Do longitudinal counterfactuals", action="store_true"
)
parser.add_argument(
    "-r", "--range", help="Use simulate_adult_range", action="store_true"
)
parser.add_argument("-y", "--year", help="Length of time for simulation")

parser.add_argument("-s", "--size", help="Test size")

args = parser.parse_args()

thres = int(args.threshold) if args.threshold is not None else 50000

if args.longitudinal:
    if args.range:
        year = args.year if args.year is not None else 10
    else:
        year = args.year if args.year is not None else 1


def compute_proximity(cfs, query, continuous, categorical, mad, tol=1e-5):
    diff = np.abs(cfs[continuous] - query.loc[0,continuous]) / (mad + tol)
    cont_loss = np.sum(diff, axis=1) / np.sum(mad)
    cat_loss = np.sum(cfs[categorical] != query.loc[0, categorical], axis=1) / len(
        categorical
    )
    return cont_loss + cat_loss


adult = pd.read_csv(
    "https://raw.githubusercontent.com/socialfoundations/folktables/main/adult_reconstruction.csv"
)
data = adult.loc[adult["workclass"] != "?", :]
data = data.loc[data["native-country"] != "?", :].reset_index()
data = data.loc[data["occupation"] != "?", :].reset_index()

target = (data["income"] > thres).astype(int)
data = data.drop(columns=["income", "index"])

savename = "geco" + "_" + "thres" + str(int(thres/1000))

xtr, xte, ytr, yte = train_test_split(data, target, test_size=0.25, random_state=123)
xtr = xtr.reset_index().drop(columns=["index", "level_0"])
xte = xte.reset_index().drop(columns=["index", "level_0"])
size = int(args.size) if args.size is not None else len(xte)

continuous = ["hours-per-week", "age", "capital-gain", "capital-loss"]
categorical = []
label_encoders = {}

if args.longitudinal:
    print("Simulating data...")
    if args.range:
        savename = savename + "_range"
        sim, _, _ = simulate_adult_range(xtr, time=year)
        sim = sim.drop(columns=['education-num'])
    else:
        savename = savename + "_year"
        sim, _, _ = simulate_adult(xtr, time=year)
        sim = sim.drop(columns=['education-num'])
        
xtr = xtr.drop(columns=['education-num'])
xte = xte.drop(columns=['education-num'])

for col in xtr.columns:
    if col in continuous:
        continue

    categorical.append(col)
    enc = LabelEncoder()
    label_encoders[col] = enc.fit(xtr[col])
    xtr[col] = enc.transform(xtr[col])
    xte[col] = enc.transform(xte[col])
    if args.longitudinal:
        sim[col] = enc.transform(sim[col])

model = RandomForestClassifier()
model = model.fit(xtr, ytr)
save_cfs = pd.DataFrame()

if not args.longitudinal:
    g = Geco(model.predict, model.predict_proba, xtr, continuous=continuous)
    idx = []
    validity = []
    prox_mean = []
    prox_max = []
    prox_min = []
    prox_std = []

    print("Generating longitudinal counterfactuals...")
    for i in tqdm(range(size)):
        query = xte.iloc[i : (i + 1), :].reset_index().drop(columns=["index"])
        pred = model.predict(query)
        if pred != 0:
            continue
        res = g.get_counterfactuals(query, features_to_vary=xtr.columns, n_cfs=10)
        prox = compute_proximity(
            cfs = res[0].iloc[:10, :], 
            query = query, 
            categorical = categorical, 
            continuous = continuous, 
            mad = g.mad
        )
        preds = model.predict(res[0])
        prox_mean.append(np.mean(prox))
        prox_max.append(np.max(prox))
        prox_min.append(np.min(prox))
        prox_std.append(np.std(prox))
        validity.append(np.sum(preds[:10]) / 10)
        idx.append(i)
        df_idx = i * np.ones(10)
        cf_idx = np.arange(10)
        save_res = res[0].iloc[:10,:].copy()
        save_res['idx'] = df_idx
        save_res['cf_idx'] = cf_idx
        save_cfs = pd.concat([save_cfs, save_res])
    res_df = pd.DataFrame(
        {
            "idx": idx,
            "validity": validity,
            "prox_mean": prox_mean,
            "prox_max": prox_max,
            "prox_min": prox_min,
            "prox_std": prox_std,
        }
    )
    all_res = {
        'summary' : res_df,
        'cfs' : save_cfs, 
        'model' : model
    }
    savename = savename + "_results.pkl"
    with open(savename, "wb") as file:
        results = pickle.dump(all_res, file)


else:
    g = Geco(
        model.predict,
        model.predict_proba,
        xtr,
        continuous=continuous,
        long=True,
        start=xtr,
        end=sim,
    )
    idx = []
    validity = []
    prox_mean = []
    prox_max = []
    prox_min = []
    prox_std = []
    
    print("Generating longitudinal counterfactuals...")
    for i in tqdm(range(size)):
        query = xte.iloc[i : (i + 1), :].reset_index().drop(columns=["index"])
        pred = model.predict(query)
        if pred != 0:
            continue
        res = g.get_counterfactuals(query, features_to_vary=xtr.columns, n_cfs=10)
        prox = compute_proximity(
            cfs = res[0].iloc[:10, :], 
            query = query, 
            categorical = categorical, 
            continuous = continuous, 
            mad = g.mad
        )
        
        preds = model.predict(res[0])
        prox_mean.append(np.mean(prox))
        prox_max.append(np.max(prox))
        prox_min.append(np.min(prox))
        prox_std.append(np.std(prox))
        validity.append(np.sum(preds[:10]) / 10)
        idx.append(i)
        df_idx = i * np.ones(10)
        cf_idx = np.arange(10)
        save_res = res[0].iloc[:10,:].copy()
        save_res['idx'] = df_idx
        save_res['cf_idx'] = cf_idx
        save_cfs = pd.concat([save_cfs, save_res])
    res_df = pd.DataFrame(
        {
            "idx": idx,
            "validity": validity,
            "prox_mean": prox_mean,
            "prox_max": prox_max,
            "prox_min": prox_min,
            "prox_std": prox_std,
        }
    )
    all_res = {
        'summary' : res_df,
        'cfs' : save_cfs,
        'model' : model
    }
    savename = savename + "_" + time.strftime("%Y%m%d") + "_results.pkl"
    with open(savename, "wb") as file:
        results = pickle.dump(all_res, file)
