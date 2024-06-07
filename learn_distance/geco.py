import pandas as pd
import numpy as np
import random
import secrets
import scipy as sp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

class Geco():
    def __init__(self, pred_fn, score_fn, data, continuous, long=False, start=None, end=None):
        self.data = data
        self.pred_fn = pred_fn
        self.continuous = continuous
        self.score_fn = score_fn
        self.long = long
        if self.long:
            if (start is None) or (end is None):
                raise ValuerError('start and/or end data frames not defined')
            self.start = start
            self.start=start
            self.end=end
    
    
    def get_data_params(self, prec=1, tol=1e-4):
        # precision, num_features, feature_range, etc.
        self.num_features = self.data.shape[1]
        self.feature_names = self.data.columns
        self.categorical = []
        self.mad = sp.stats.median_abs_deviation(self.data[self.continuous])
        self.feature_range = {}
        self.precision = {}
        
        for feature in self.data.columns:
            if feature in self.continuous:
                self.feature_range[feature] = [np.min(self.data[feature]), np.max(self.data[feature])]
                if self.data[feature].dtype == 'int':
                    self.precision[feature] = 0
                else:
                    self.precision[feature] = prec
                    
            else:
                self.categorical.append(feature)
                self.feature_range[feature] = np.unique(self.data[feature])
        if self.long:
            self.get_long()
                
    def get_long(self):
        self.cont_diff = self.end[self.continuous] - self.start[self.continuous]
        self.scale = MinMaxScaler().fit(self.cont_diff)
        #self.cont_diff = self.scale.transform(self.cont_diff)
        self.change_freq = np.sum(self.start[self.categorical] != self.end[self.categorical], axis=0) / len(self.start)
        self.ohe = OneHotEncoder().fit(self.data[self.categorical])
        start_cat = self.ohe.transform(self.start[self.categorical])
        end_cat = self.ohe.transform(self.end[self.categorical])
        self.cat_diff = end_cat - start_cat
        self.diff_mad = np.zeros(len(self.continuous))
        self.cont_change_freq = np.zeros(len(self.continuous))
        for i in range(len(self.continuous)):
            nonzero = self.cont_diff[self.cont_diff[self.continuous[i]] != 0].iloc[:,i]
            self.cont_change_freq[i] = len(nonzero) / len(self.cont_diff)
            self.diff_mad[i] = sp.stats.median_abs_deviation(nonzero)
        
    
    def do_random_init_random(self, num_inits, features_to_vary, query, desired_class):
        
        init = np.zeros((num_inits, self.num_features))
        kx = 0
        
        while kx < num_inits:
            one_init = np.zeros(self.num_features)
            for jx, feature in enumerate(self.feature_names):
                if feature not in features_to_vary:
                    one_init[jx] = query.iloc[0,jx]
                    continue
                one_init[jx] = np.random.choice(self.data[feature])
            test = pd.DataFrame(one_init.reshape(1,-1), columns=self.feature_names)
            pred = self.pred_fn(test)
            if pred == desired_class:
                init[kx] = one_init
                kx += 1
        return (pd.DataFrame(init, columns=self.feature_names))
    
    def do_random_init_uniform(self, num_inits, features_to_vary, query, desired_class):
        
        init = np.zeros((num_inits, self.num_features))
        kx = 0
        
        while kx < num_inits:
            one_init = np.zeros(self.num_features)
            for jx, feature in enumerate(self.feature_names):
                if feature not in features_to_vary:
                    one_init[jx] = query.iloc[0,jx]
                    continue
                
                if feature in self.continuous:
                    one_init[jx] = np.round(
                        np.random.uniform(
                            self.feature_range[feature][0],
                            self.feature_range[feature][1]),
                            decimals = self.precision[feature]
                    )
                else:
                    one_init[jx] = np.random.choice(
                        self.feature_range[feature]
                    )
            test = pd.DataFrame(one_init.reshape(1,-1), columns=self.feature_names)
            pred = self.pred_fn(test)
            if pred == desired_class:
                init[kx] = one_init
                kx += 1
        return (pd.DataFrame(init, columns=self.feature_names)) 
    
    
    def condense_ranges(self, query):
        for i in range(len(self.continuous)):
            feat = self.continuous[i]
            new_feats = np.array(query[feat]) + self.cont_diff[feat]
            self.feature_range[feat] = [ 
                max(np.min(new_feats), np.min(self.data[feat])),
                min(np.max(new_feats), np.max(self.data[feat]))
            ] # constrain features to be intersection of possible changes and observed values
        print(self.feature_range)
        
    
    def compute_proximity(self, cfs, query, tol=1e-5):
        diff = np.abs(cfs[self.continuous] - query.loc[0,self.continuous]) / (self.mad+tol)
        cont_loss = np.sum(diff, axis=1) / np.sum(self.mad)
        '''
        Finish implementation of categorical distance 
        '''
        if len(self.categorical) < 1:
            return cont_loss
        if self.long:
            cat_loss = np.sum((cfs[self.categorical] != query.loc[0,self.categorical]) / (self.change_freq+tol), axis=1)
            cat_loss = cat_loss / len(self.categorical)
        else:
            cat_loss = np.sum(cfs[self.categorical] != query.loc[0, self.categorical], axis=1) / len(self.categorical)
        if len(self.continuous) < 1:
            return cat_loss
        return cont_loss + cat_loss
        
    def compute_sparsity(self, cfs, query):
        cont_sparsity = np.count_nonzero(cfs[self.continuous] - np.array(query[self.continuous]).reshape(1,-1), axis=1)
        '''
        Finish implementation of categorical distance
        Instead of ohe transform, check equality in entries
        '''
        cat_sparsity = np.sum(cfs[self.categorical] != query.loc[0,self.categorical], axis=1)
        if len(self.categorical) < 1:
            return cont_sparsity / len(self.feature_names)
        elif len(self.continuous) < 1:
            return cat_sparsity / len(self.feature_names)
        return (cont_sparsity + cat_sparsity) / len(self.feature_names)
    
    def compute_yloss(self, cfs, desired_class):
        preds = self.score_fn(cfs)[:,desired_class]
        loss = -preds
        loss[np.where(preds < 0.5)] += 1
        loss[np.where(preds >= 0.5)] = 0
        return loss.flatten()
 
    def compute_loss(self, cfs, query, desired_class, longitudinal=False):
        prox = self.compute_proximity(cfs, query)
        sparse = self.compute_sparsity(cfs, query)
        yloss = self.compute_yloss(cfs, desired_class)
        loss = np.array(prox + sparse + yloss).reshape(-1,1)
        index = np.arange(len(cfs)).reshape(-1,1)
        return np.concatenate([index, loss], axis=1)
    
    def compare_cf(self, cfs, query, desired_class, n_comp=5, alpha=5):
        loss = np.zeros((cfs.shape[0],))
        prox = self.compute_proximity(cfs, query)
        yloss = self.compute_yloss(cfs, desired_class)
        for i in range(cfs.shape[0]):
            cf = cfs.iloc[i:(i+1),:]
            cont_cf = cf[self.continuous] - np.array(query[self.continuous])
            cont_cf = self.scale.transform(cont_cf)
            cat_cf = self.ohe.transform(cf[self.categorical]) - self.ohe.transform(query[self.categorical]).toarray().flatten()
            cont_comp = np.sum(np.abs(cont_cf - self.cont_diff) / len(self.continuous), axis=1)
            cat_comp = np.sum(cat_cf != self.cat_diff.toarray(), axis=1) / (len(self.categorical))
            if len(self.categorical) < 1:
                comp = cont_comp
            elif len(self.continuous) < 1:
                comp = cat_comp.flatten()
            else:
                comp = cat_comp.flatten() + cont_comp
            comp_sort = np.sort(comp)
            top = comp_sort[:n_comp]
            avg = np.mean(top)
            loss[i] = avg
        index = np.arange(len(cfs)).reshape(-1,1)
        loss = np.array(loss + alpha*yloss + prox).reshape(-1,1)
        return np.concatenate([index, loss], axis=1)
    
    def compare_cf_mad(self, cfs, query, desired_class, n_comp=5, tol=1e-5, alpha=5):
        loss = np.zeros((cfs.shape[0],))
        prox = self.compute_proximity(cfs, query)
        yloss = self.compute_yloss(cfs, desired_class)
        for i in range(cfs.shape[0]):
            cf = cfs.iloc[i:(i+1),:]
            cont_cf = cf[self.continuous] - np.array(query[self.continuous])
            cat_cf = self.ohe.transform(cf[self.categorical]) - self.ohe.transform(query[self.categorical]).toarray().flatten()
            cont_comp = np.sum((np.abs(cont_cf - self.cont_diff) / (self.diff_mad+tol))/(self.cont_change_freq+tol), axis=1)
            cat_comp = np.sum(cat_cf != self.cat_diff.toarray(), axis=1) / (len(self.categorical))
            if len(self.categorical) < 1:
                comp = cont_comp
            elif len(self.continuous) < 1:
                comp = cat_comp.flatten()
            else:
                comp = cat_comp.flatten() + np.array(cont_comp)
            comp_sort = np.sort(comp)
            top = comp_sort[:n_comp]
            avg = np.mean(top)
            loss[i] = avg
        index = np.arange(len(cfs)).reshape(-1,1)
        loss = np.array(loss + alpha*yloss + prox).reshape(-1,1)
        return np.concatenate([index, loss], axis=1)
    
    def mate(self, parent1, parent2, features_to_vary, query):
        one_init = np.zeros(self.num_features)
        for i in range(self.num_features):
            feat = self.feature_names[i]
            if feat not in features_to_vary:
                one_init[i] = query.iloc[0][feat]
                continue
                
            prob = random.random()
            
            if prob < 0.40:
                one_init[i] = parent1[feat]
                
            elif prob < 0.80:
                one_init[i] = parent2[feat]
            
            else:
                if feat in self.continuous:
                    one_init[i] = np.round(
                        np.random.uniform(
                            self.feature_range[feat][0],
                            self.feature_range[feat][1]),
                            decimals = self.precision[feat]
                    )
                else:
                    one_init[i] = np.random.choice(
                        self.feature_range[self.feature_names[i]]
                    )
        return one_init
    
    def get_counterfactuals(self, query, features_to_vary, n_cfs, method = 'random', init_multiplier = 10, maxiter = 500, thres=1e-2):
        self.get_data_params()
        num_inits = init_multiplier * n_cfs
        desired_class = (self.pred_fn(query) + 1) % 2
        print('Initliaizing counterfactuals...')
        if self.long:
            self.condense_ranges(query)
        if method == 'uniform':
            initial = self.do_random_init_uniform(num_inits, features_to_vary, query, desired_class)
        elif method == 'random':
            initial = self.do_random_init_random(num_inits, features_to_vary, query, desired_class)
        population = initial.copy()
        iterations = 0
        prev_best = -np.inf
        cur_best = np.inf
        stop_count = 0
        cfs_pred = [np.inf] * n_cfs
        proportion_true = [num_inits]
        avg_loss = []
        print('Searching...')
        while iterations < maxiter:
            if (abs(prev_best - cur_best) <= thres) and (i == desired_class for i in cfs_pred):
                stop_count += 1
            else:
                stop_count = 0
            if stop_count >= 10:
                break
            
            prev_best = cur_best
            
            if self.long:
                fitness = self.compare_cf_mad(population, query, desired_class)
            else:
                fitness = self.compute_loss(population, query, desired_class)
            fitness = fitness[fitness[:,1].argsort()]
            cur_best = fitness[0][1]
            
            new_generation_1 = population.iloc[fitness[:n_cfs,0], :]
            cfs_pred = self.pred_fn(new_generation_1)
            
            remaining_pop = num_inits - n_cfs
            new_generation_2 = np.zeros((remaining_pop, self.num_features))
            top_half = fitness[:int(fitness.shape[0]/2),0].astype(np.int32)
            for idx in range(remaining_pop):
                parent1 = population.iloc[random.choice(top_half),:]
                parent2 = population.iloc[random.choice(top_half),:]
                child = self.mate(parent1, parent2, features_to_vary, query)
                new_generation_2[idx] = child
            new_generation_2 = pd.DataFrame(new_generation_2, columns=self.feature_names)
            population = pd.concat([new_generation_1, new_generation_2])
            iterations += 1
            proportion_true.append(np.sum(self.pred_fn(population) == desired_class))
            avg_loss.append(np.mean(fitness[:10,1]))
        
        return population, fitness, initial, iterations, proportion_true, avg_loss