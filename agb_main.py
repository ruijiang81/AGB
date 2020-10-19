import os
import random 
import pickle
import matplotlib 
matplotlib.use('Agg')
os.environ['QT_QPA_PLATFORM']='offscreen'
from alipy.toolbox import ToolBox
from alipy.oracle import Oracle, Oracles
from alipy.utils.misc import randperm
from alipy.query_strategy.noisy_oracles import QueryNoisyOraclesCEAL, QueryNoisyOraclesAll, \
    QueryNoisyOraclesIEthresh, QueryNoisyOraclesRandom, get_majority_vote, BaseNoisyOracleQuery, \
    get_query_results,QueryNoisyOraclesSelectInstanceUncertainty
from sklearn.datasets import make_classification
import copy
import collections
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import numpy as np
from itertools import combinations_with_replacement, combinations
import pandas as pd 
from alipy.index.index_collections import IndexCollection
from itertools import combinations_with_replacement
from scipy.special import comb, perm
from sklearn.preprocessing import LabelEncoder
from alipy.query_strategy.query_labels import _get_proba_pred
from alipy.query_strategy.noisy_oracles import majority_vote
from sklearn import cross_validation
from cealredundancy import QueryNoisyOraclesExpM, QueryNoisyOraclesCEAL, QueryNoisyOraclesCurEM
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from scipy.special import comb
import scipy 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--option", help="price setting", default = 'linear')
parser.add_argument("--dataset", help="dataset name", default = 'spambase')
parser.add_argument("--repl", help="repetition start", default = 0)
parser.add_argument("--repr", help="repetition end", default = 5)
parser.add_argument("--split", help="alibox split setting save", default = False)
parser.add_argument("--metric", help="performance metric chosen", default = 'accuracy')
args = parser.parse_args()
option = args.option
dataset = args.dataset
low_ind = int(args.repl)
high_ind = int(args.repr)
metric = args.metric

def pickle_save(file_name, result):
    with open(file_name, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

if dataset == 'spambase':
    data = pd.read_csv('../data/Spam/spambase.data',header = None)
    X, y = data.iloc[:,:-1].values, data.iloc[:,-1].values

if not args.split:
    alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')
    # Split data
    alibox.split_AL(test_ratio=0.3, initial_label_rate=0.05, split_count=30)
    pickle_save('./pkl/aliboxmv_' + dataset + '_' + option + '.pkl', alibox)
else:
    with open(args.split, 'rb') as f:
        alibox = pickle.load(f)

# Use the default Logistic Regression classifier
model = alibox.get_default_model()


stopping_criterion = alibox.get_stopping_criterion('cost_limit', 200)

def acc_price(price, option, fixP = 0.6, custom = {0.02:0.5,0.14:0.8,0.25:0.9}):
    if option == 'concave':
        return 0.48 + 0.066 * (100 * price) - 0.0022 * (100 * price) ** 2
    if option == 'fix':
        return fixP
    if option == 'asymptotic':
        return 1 - 1 / (price * 110)
    if option == 'linear':
        return 0.6 + (0.8-0.6)/(0.25-0.02)*(price - 0.02)
    if option == 'custom':
        return custom[price]

def create_oracles_redundancy_kmeans(X, y, num_price, min_p = 0.02, max_p = 0.25, option = 'concave', fixP = 0.6, custom = {0.02:0.5,0.14:0.8,0.25:0.9}, other_acc = 0.55):
    n_samples = len(y)
    oracles = Oracles()
    num_c = 50 
    cluster = KMeans(n_clusters=num_c, random_state=0).fit(X)
    price_levels = [5, 4, 3, 2, 1]
    h_acc =  [0.9,0.8,0.7,0.6,0.55]
    p = 0.2
    target_acc = (np.array(h_acc) - p * other_acc) / (1 - p)
    #cluster = DBSCAN(eps=0.3, min_samples= int(len(y)/5)).fit(X)
    # labeler expertise on some cluster 
    #rm_cluster = list(range(num_c))
    prob = np.ones([num_c, num_price]) * other_acc
    exp_his = []
    for i in range(num_price):
        index = np.random.choice(range(num_c), round( (1 - p) * num_c), replace = False)
        exp_his.append(index)
        prob[index, i] = target_acc[i]
    cluster_num = [list(cluster.labels_).count(i) for i in range(num_c)]
    #prob[np.array(exp),np.array(range(len(exp)))] += np.random.uniform(0.2, 0.45, len(exp)) 
    avg_prob = np.average(prob, 0, weights=[list(cluster.labels_).count(i) for i in range(num_c)])
    '''
    for i, p in enumerate(avg_prob):
        def f(x):
            return(acc_price_kmeans(x, option, fixP, custom) - p)
        sol = scipy.optimize.fsolve(f, [min_p, max_p])
        sol1 = sol[sol >= min_p]
        sol1 = sol1[sol1 <= max_p]
        price = np.random.choice(sol1)
        price_lab.append(price)
    '''
    y_his = []
    exp_his = np.array(exp_his)
    for i, price in enumerate(price_levels):
        y_ = y.copy()
        for j in exp_his[i,:]:
            target_acc = prob[j,i]
            num_s = sum(cluster.labels_== j)
            perms = randperm(num_s - 1)
            subset = y_[cluster.labels_== j]
            subset[perms[0:int(round(num_s*(1-target_acc)))]] = 1 - subset[perms[0:int(round(num_s*(1-target_acc)))]]
            y_[cluster.labels_== j] = subset
        num_ns = sum([ite not in exp_his[i,:] for ite in cluster.labels_])
        perms = randperm(num_ns - 1)
        subset = y_[[ite not in exp_his[i,:] for ite in cluster.labels_]]
        subset[perms[0:int(round(num_ns*(1-other_acc)))]] = 1 - subset[perms[0:int(round(num_ns*(1-other_acc)))]]
        y_[[ite not in exp_his[i,:] for ite in cluster.labels_]] = subset
        oracle_ = Oracle(labels=y_, cost=np.zeros(y.shape)+price)
        name_ = 'o' + str(i)
        oracles.add_oracle(oracle_name=name_, oracle_object=oracle_)
        y_his.append(y_)
    oracles1 = copy.deepcopy(oracles)
    y_his = np.array(y_his)
    for combination in combinations(range(len(price_levels)),3):
        y_ = (y_his[combination,:].sum(0) > len(combination) / 2) * 1
        oracle_ = Oracle(labels=y_, cost=np.zeros(y.shape)+sum([price_levels[price] for price in combination]))
        name_ = 'o' + str(combination)
        oracles.add_oracle(oracle_name=name_, oracle_object=oracle_)
    return oracles1, oracles, price_levels, avg_prob, cluster_num, prob 

num_price = 5
oracles, oracles1, price_levels, accuracy, cluster_num, prob = create_oracles_redundancy_kmeans(X, y, num_price, min_p = 0.02, max_p = 0.25, option = 'concave', fixP = 0.6, custom = {0.02:0.5,0.14:0.8,0.25:0.9}, other_acc = 0.55)

def al_loop(strategy, alibox, round, oracles):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)
    # Get repository to store noisy labels
    repo = alibox.get_repository(round)
    _, y_lab, indexes_lab = repo.get_training_data()
    model = alibox.get_default_model()
    model.fit(X=X[indexes_lab], y=y_lab)
    pred = model.predict(X[test_idx])
    perf = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=pred)
    # save
    cost = 0
    st = alibox.State(select_index=[], performance=perf, cost=cost)
    saver.add_state(st)
    stopping_criterion.update_information(saver)   
    while not stopping_criterion.is_stop():
        # Query
        select_ind, select_ora = strategy.select(label_ind, unlab_ind, repo = repo, model = model, n_neighbors = 30)
        oracles1 = copy.deepcopy(oracles)
        vote_count, vote_result, cost = get_majority_vote(selected_instance=select_ind, oracles=oracles1, names=select_ora)
        repo.update_query(labels=vote_result, indexes=select_ind)
        # update ind
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)
        # Train/test
        _, y_lab, indexes_lab = repo.get_training_data()
        model.fit(X=X[indexes_lab], y=y_lab)
        if metric == 'accuracy':
            pred = model.predict(X[test_idx])
            perf = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=pred)
        elif metric == 'auc':
            perf = roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:,1])
        # save
        st = alibox.State(select_index=select_ind, performance=perf, cost=cost)
        saver.add_state(st)
        stopping_criterion.update_information(saver)
    stopping_criterion.reset()
    return saver

ceal_result = []
cealmv_result = []
alpcv_result = []
iet_result = []
all_result = []
rand_result = []
alp_result = []
alc_result = []
bmtr_result = []
alpcv_result = []
bmtrcv_result = []
alpub_result = []
bmtrub_result = []
bmtrmvub_result = [] 
expm_result = []
expmr_result = []
curem_result = []

for r in range(low_ind, high_ind):
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(r)
    # init strategies
    ceal = QueryNoisyOraclesCEAL(X, y, oracles=oracles, initial_labeled_indexes=label_ind)
    expm = QueryNoisyOraclesExpM(X, y, oracles=oracles, initial_labeled_indexes=label_ind)
    curem = QueryNoisyOraclesCurEM(X = X, y=y, oracles=oracles, initial_labeled_indexes=label_ind, if_budget = 0, unit_b = 3, unit_num = 10)
    iet = QueryNoisyOraclesIEthresh(X=X, y=y, oracles=oracles, initial_labeled_indexes=label_ind)
    all = QueryNoisyOraclesAll(X=X, y=y, oracles=oracles)
    rand = QueryNoisyOraclesRandom(X=X, y=y, oracles=oracles)
    curem_result.append(copy.deepcopy(al_loop(curem, alibox, r, oracles)))
    ceal_result.append(copy.deepcopy(al_loop(ceal, alibox, r, oracles)))
    expm_result.append(copy.deepcopy(al_loop(expm, alibox, r, oracles)))
    iet_result.append(copy.deepcopy(al_loop(iet, alibox, r, oracles)))
    all_result.append(copy.deepcopy(al_loop(all, alibox, r, oracles)))
    rand_result.append(copy.deepcopy(al_loop(rand, alibox, r, oracles)))
    pickle_save('./result/mv/mv_' + dataset + '_' + option + '_all_' + str(r) + '.pickle', all_result[r])
    pickle_save('./result/mv/mv_' + dataset + '_' + option + '_iet_'  + str(r) + '.pickle', iet_result[r])
    pickle_save('./result/mv/mv_' + dataset + '_' + option + '_rand_' + str(r) + '.pickle', rand_result[r])
    pickle_save('./result/mv/mv_' + dataset + '_' + option + '_ceal_' + str(r) + '.pickle', ceal_result[r])
    pickle_save('./result/mv/mv_' + dataset + '_' + option + '_expm_' + str(r) + '.pickle', expm_result[r])
    pickle_save('./result/mv/mv_' + dataset + '_' + option + '_curem_' + str(r) + '.pickle', curem_result[r])

print(oracles.full_history())
analyser = alibox.get_experiment_analyser(x_axis='cost')
analyser.add_method(method_results=iet_result, method_name='iet')
analyser.add_method(method_results=all_result, method_name='all')
analyser.add_method(method_results=rand_result, method_name='rand')
analyser.add_method(method_results=ceal_result, method_name='ceal')
analyser.add_method(method_results=expm_result, method_name='expm')
analyser.add_method(method_results=curem_result, method_name='curem')
analyser.plot_learning_curves(saving_path = './alipy_plotting_mv_'+ option + '.jpg', show = False)
