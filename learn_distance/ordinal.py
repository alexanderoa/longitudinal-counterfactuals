import numpy as np

def get_feature_idx(frequent, stats, s_feats, x_feats):
    """
    Obtains indices of feature buckets for conversion to ordinal data

    Args --- 
    frequent : list of 'frequent' variables under FIDDLE framework
    stats : list of statistics to be calculated on frequent variables
    s_feats : list of features that do not change with time
    x_feats : list of features that do change with time

    Returns --- 
    sets : list of features 
    feat_idx : dictionary mapping feature names to their indices
    """
    feats = list(s_feats) + list(x_feats)
    freq_stats = []
    freq_mask = []
    for f in frequent:
        for s in stats:
            freq_stats.append(f + '_' + s)
        freq_mask.append(f + '_' + 'mask')
    feature_prefix = freq_mask + freq_stats
    for f in s_feats:
        prefix = f[:(f.find('value')-1)]
        feature_prefix.append(prefix)
    for f in x_feats:
        rate = f.find('_Rate')
        amount = f.find('_Amount')
        input = f.find('_Input')
        dose = f.find('_Dose')
        bar = f.find('_')
        colon = f.find(':')
        if rate > 0:
            feature_prefix.append(f[:(rate+5)]) #adding length of _Rate
        elif amount > 0:
            feature_prefix.append(f[:(amount+7)]) # adding length of _Amount
        elif dose > 0:
            feature_prefix.append(f[:(dose+5)])
        elif input > 0:
            feature_prefix.append(f) # for drug inputs, save entire variable name
        elif bar > 0 and bar <= 6:
            if f[bar:(bar+3)] == '_va':
                feature_prefix.append(f[:bar]+'_value')
            else:
                feature_prefix.append(f[:bar])
        elif colon > 0 and colon <= 6:
            if f[bar:(bar+3)] == '_va':
                feature_prefix.append(f[:colon]+'_value')
            else:
                feature_prefix.append(f[:colon])

    sets = np.unique(feature_prefix)
    for f in frequent:
        idx = np.where(sets == f)
        sets = np.delete(sets, idx)
    feat_idx = {}
    for var in sets:
        feat_idx[var] = []
        for idx, feature in enumerate(feats):
            if var in feature:
                feat_idx[var].append(idx)

    age_values = [] # AGE_value also captures LANGUAGE_value
    for idx in feat_idx['AGE']:
        if 'LANGUAGE' not in feats[idx]:
            age_values.append(idx)
    feat_idx['AGE'] = age_values
    
    return feat_idx, sets, freq_stats

def ohe_to_ordinal(feature_sets, feat_idx, x):
    ordinal_dict = {}
    for feat in feature_sets:
        idx = feat_idx[feat]
        not_observed = np.sum(x[:,idx], axis=1) < 1
        t = np.argmax(x[:,idx], axis=1)
        t[not_observed] = -1
        ordinal_dict[feat] = t
    return ordinal_dict

def check_feature_idx(x, feat_idx, sets, feats):
    for feat in sets:
        idx = feat_idx[feat]
        test = np.sum(x[:, idx], axis=1) <= 1
        passed = np.all(test)
        if not passed:
            print("Failed :(")
            print(np.array(feats)[idx])
            break
    print("Success :)")

def get_na_mask(x, feats, feat_idx):
    ''' 
    Returns rows that have an observation (i.e. a one in some bucket) for all feats
    '''
    na_rows = []
    for i in range(x.shape[0]):
        for feat in feats:
            idx = np.array(feat_idx[feat])
            if np.sum(x[i,idx]) == 0:
                na_rows.append(i)
                break
    na_mask = np.ones(x.shape[0])
    na_mask[na_rows] = False
    return np.array(na_mask, dtype=np.bool8)