import os
import random 
import matplotlib 
matplotlib.use('Agg')
#os.environ['QT_QPA_PLATFORM']='offscreen'
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
from itertools import combinations_with_replacement
import pandas as pd 
from alipy.index.index_collections import IndexCollection
from itertools import combinations_with_replacement
from scipy.special import comb, perm
from alipy.query_strategy.query_labels import _get_proba_pred
from alipy.query_strategy.noisy_oracles import majority_vote
from sklearn import cross_validation

class QueryNoisyOraclesCEAL(BaseNoisyOracleQuery):
    """Cost-Effective Active Learning from Diverse Labelers (CEAL) method assumes
    that different oracles have different expertise. Even the very noisy oracle
    may perform well on some kind of examples. The cost of a labeler is proportional
    to its overall labeling quality and it is thus necessary to query from the right oracle
    according to the selected instance.
    This method will select an instance-labeler pair (x, a), and queries the label of x
    from a, where the selection of both the instance and labeler is based on a
    evaluation function Q(x, a).
    The selection of instance is depend on its uncertainty. The selection of oracle is
    depend on the oracle's performance on the nearest neighbors of selected instance.
    The cost of each oracle is proportional to its overall labeling quality.
    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.
    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.
    oracles: {list, alipy.oracle.Oracles}
        An alipy.oracle.Oracle object that contains all the
        available oracles or a list of oracles.
        Each oracle should be a alipy.oracle.Oracle object.
    initial_labeled_indexes: {list, np.ndarray, IndexCollection}
            The indexes of initially labeled samples. Used for initializing the scores of each oracle.
    References
    ----------
    [1] Sheng-Jun Huang, Jia-Lve Chen, Xin Mu, Zhi-Hua Zhou. 2017.
        Cost-Effective Active Learning from Diverse Labelers. In The
        Proceedings of the 26th International Joint Conference
        on Artificial Intelligence (IJCAI-17), 1879-1885.
    """

    def __init__(self, X, y, oracles, initial_labeled_indexes):
        super(QueryNoisyOraclesCEAL, self).__init__(X, y, oracles=oracles)
        # ytype = type_of_target(self.y)
        # if 'multilabel' in ytype:
        #     warnings.warn("This query strategy does not support multi-label.",
        #                   category=FunctionWarning)
        assert (isinstance(initial_labeled_indexes, collections.Iterable))
        self._ini_ind = np.asarray(initial_labeled_indexes)
        # construct a nearest neighbor object implemented by scikit-learn
        self._nntree = NearestNeighbors(metric='euclidean')
        self._nntree.fit(self.X[self._ini_ind])

    def select(self, label_index, unlabel_index, eval_cost=False, model=None, **kwargs):
        """Query from oracles. Return the index of selected instance and oracle.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        eval_cost: bool, optional (default=False)
            To evaluate the cost of oracles or use the cost provided by oracles.
        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.
        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.
        Returns
        -------
        selected_instance: int
            The index of selected instance.
        selected_oracle: int or str
            The index of selected oracle.
            If a list is given, the index of oracle will be returned.
            If a Oracles object is given, the oracle name will be returned.
        """

        if model is None:
            model = LogisticRegression(solver='liblinear')
            model.fit(self.X[label_index], self.y[label_index])
        pred_unlab, _ = _get_proba_pred(self.X[unlabel_index], model)

        n_neighbors = min(kwargs.pop('n_neighbors', 10), len(self._ini_ind) - 1)
        return self.select_by_prediction_mat(label_index, unlabel_index, pred_unlab,
                                             n_neighbors=n_neighbors, eval_cost=eval_cost)

    def select_by_prediction_mat(self, label_index, unlabel_index, predict, **kwargs):
        """Query from oracles. Return the index of selected instance and oracle.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        predict: : 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.
        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.
        eval_cost: bool, optional (default=False)
            To evaluate the cost of oracles or use the cost provided by oracles.
        Returns
        -------
        selected_instance: int
            The index of selected instance.
        selected_oracle: int or str
            The index of selected oracle.
            If a list is given, the index of oracle will be returned.
            If a Oracles object is given, the oracle name will be returned.
        """
        n_neighbors = min(kwargs.pop('n_neighbors', 10), len(self._ini_ind)-1)
        eval_cost = kwargs.pop('eval_cost', False)
        Q_table, oracle_ind_name_dict = self._calc_Q_table(label_index, unlabel_index, self._oracles, predict,
                                                           n_neighbors=n_neighbors, eval_cost=eval_cost)
        # get the instance-oracle pair
        selected_pair = np.unravel_index(np.argmax(Q_table, axis=None), Q_table.shape)
        sel_ora = oracle_ind_name_dict[selected_pair[0]]
        if not isinstance(sel_ora, list):
            sel_ora = [sel_ora]
        return [unlabel_index[selected_pair[1]]], sel_ora

    def _calc_Q_table(self, label_index, unlabel_index, oracles, pred_unlab, n_neighbors=10, eval_cost=False):
        """Query from oracles. Return the Q table and the oracle name/index of each row of Q_table.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        oracles: {list, alipy.oracle.Oracles}
            An alipy.oracle.Oracle object that contains all the
            available oracles or a list of oracles.
            Each oracle should be a alipy.oracle.Oracle object.
        predict: : 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.
        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.
        eval_cost: bool, optional (default=False)
            To evaluate the cost of oracles or use the cost provided by oracles.
        Returns
        -------
        Q_table: 2D array
            The Q table.
        oracle_ind_name_dict: dict
            The oracle name/index of each row of Q_table.
        """
        # Check parameter and initialize variables
        if self.X is None or self.y is None:
            raise Exception('Data matrix is not provided, use select_by_prediction_mat() instead.')
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        num_of_neighbors = n_neighbors
        if len(unlabel_index) <= 1:
            return unlabel_index

        Q_table = np.zeros((len(oracles), len(unlabel_index)))  # row:oracle, col:ins
        spv = np.shape(pred_unlab)
        # calc least_confident
        rx = np.partition(pred_unlab, spv[1] - 1, axis=1)
        rx = 1 - rx[:, spv[1] - 1]
        for unlab_ind, unlab_ins_ind in enumerate(unlabel_index):
            # evaluate oracles for each instance
            nn_dist, nn_of_selected_ins = self._nntree.kneighbors(X=self.X[unlab_ins_ind].reshape(1, -1),
                                                                  n_neighbors=num_of_neighbors,
                                                                  return_distance=True)
            nn_dist = nn_dist[0]
            nn_of_selected_ins = nn_of_selected_ins[0]
            nn_of_selected_ins = self._ini_ind[nn_of_selected_ins]  # map to the original population
            oracles_score = []
            nn_dist[nn_dist == 0] = 1e-6
            nn_sim = 1. / nn_dist
            nn_sim = nn_sim / sum(nn_sim)
            for ora_ind, ora_name in enumerate(self._oracles_iterset):
                # calc q_i(x), expertise of this instance
                oracle = oracles[ora_name]
                #labels, cost = oracle.query_by_index(self._ini_ind)
                labels, cost = oracle.query_by_index(nn_of_selected_ins)
                oracles_score.append(sum([nn_sim[i] * (labels[i] == self.y[nn_of_selected_ins[i]]) for i in
                                          range(num_of_neighbors)]))
                #oracles_score.append(sum([(labels[i] == self.y[self._ini_ind[i]]) for i in
                #                          range(len(self._ini_ind))])/len(self._ini_ind))
                #oracles_score.append(sum([(labels[i] == self.y[nn_of_selected_ins[i]]) for i in
                #                          range(num_of_neighbors)]) / num_of_neighbors)                
                # calc c_i, cost of each labeler
                labels, cost = oracle.query_by_index(label_index)
                if eval_cost:
                    oracles_cost = sum([labels[i] == self.y[label_index[i]] for i in range(len(label_index))]) / len(label_index)
                else:
                    oracles_cost = cost[0]
                Q_table[ora_ind, unlab_ind] = oracles_score[ora_ind] * rx[unlab_ind] / max(oracles_cost, 0.0001)
        return Q_table, self._oracle_ind_name_dict


def accuracy_mv(oracles_score_init, comblist):
    l = len(comblist)
    acc = [oracles_score_init[i] for i in comblist]
    f_acc = np.prod(acc)    
    for i in range(l):
        f_acc += (1 - acc[i]) * np.prod([k for j,k in enumerate(acc) if i!=j])
    return f_acc


class QueryNoisyOraclesRR(BaseNoisyOracleQuery):
    def __init__(self, X, y, oracles, initial_labeled_indexes,price_levels):
        super(QueryNoisyOraclesCEALRedundancy, self).__init__(X, y, oracles=oracles)
        # ytype = type_of_target(self.y)
        # if 'multilabel' in ytype:
        #     warnings.warn("This query strategy does not support multi-label.",
        #                   category=FunctionWarning)
        assert (isinstance(initial_labeled_indexes, collections.Iterable))
        self._ini_ind = np.asarray(initial_labeled_indexes)
        # construct a nearest neighbor object implemented by scikit-learn
    def select(self, label_index, unlabel_index, eval_cost=False, model=None, **kwargs):
        return [np.random.choice(unlabel_index)], [self._oracle_ind_name_dict[np.random.randint(0, len(self._oracles), 1)[0]]]


class QueryNoisyOraclesExpM(BaseNoisyOracleQuery):
    def __init__(self, X, y, oracles, initial_labeled_indexes):
        super(QueryNoisyOraclesExpM, self).__init__(X, y, oracles=oracles)
        # ytype = type_of_target(self.y)
        # if 'multilabel' in ytype:
        #     warnings.warn("This query strategy does not support multi-label.",
        #                   category=FunctionWarning)
        assert (isinstance(initial_labeled_indexes, collections.Iterable))
        self._ini_ind = np.asarray(initial_labeled_indexes)
        # construct a nearest neighbor object implemented by scikit-learn
        self._nntree = NearestNeighbors(metric='euclidean')
        self._nntree.fit(self.X[self._ini_ind])

    def select(self, label_index, unlabel_index, eval_cost=False, model=None, **kwargs):
        """Query from oracles. Return the index of selected instance and oracle.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        eval_cost: bool, optional (default=False)
            To evaluate the cost of oracles or use the cost provided by oracles.
        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.
        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.
        Returns
        -------
        selected_instance: int
            The index of selected instance.
        selected_oracle: int or str
            The index of selected oracle.
            If a list is given, the index of oracle will be returned.
            If a Oracles object is given, the oracle name will be returned.
        """

        if model is None:
            model = LogisticRegression(solver='liblinear')
            model.fit(self.X[label_index], self.y[label_index])
        pred_unlab, _ = _get_proba_pred(self.X[unlabel_index], model)

        n_neighbors = min(kwargs.pop('n_neighbors', 30), len(self._ini_ind) - 1)
        return self.select_by_prediction_mat(label_index, unlabel_index, pred_unlab,
                                             n_neighbors=n_neighbors, eval_cost=eval_cost)

    def select_by_prediction_mat(self, label_index, unlabel_index, predict, **kwargs):
        """Query from oracles. Return the index of selected instance and oracle.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        predict: : 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.
        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.
        eval_cost: bool, optional (default=False)
            To evaluate the cost of oracles or use the cost provided by oracles.
        Returns
        -------
        selected_instance: int
            The index of selected instance.
        selected_oracle: int or str
            The index of selected oracle.
            If a list is given, the index of oracle will be returned.
            If a Oracles object is given, the oracle name will be returned.
        """
        n_neighbors = min(kwargs.pop('n_neighbors', 30), len(self._ini_ind)-1)
        eval_cost = kwargs.pop('eval_cost', False)
        Q_table, oracle_ind_name_dict = self._calc_Q_table(label_index, unlabel_index, self._oracles, predict,
                                                           n_neighbors=n_neighbors, eval_cost=eval_cost)
        # get the instance-oracle pair
        selected_pair = np.unravel_index(np.argmax(Q_table, axis=None), Q_table.shape)
        sel_ora = oracle_ind_name_dict[selected_pair[0]]
        if not isinstance(sel_ora, list):
            sel_ora = [sel_ora]
        return [unlabel_index[selected_pair[1]]], sel_ora

    def _calc_Q_table(self, label_index, unlabel_index, oracles, pred_unlab, n_neighbors=30, eval_cost=False):
        """Query from oracles. Return the Q table and the oracle name/index of each row of Q_table.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        oracles: {list, alipy.oracle.Oracles}
            An alipy.oracle.Oracle object that contains all the
            available oracles or a list of oracles.
            Each oracle should be a alipy.oracle.Oracle object.
        predict: : 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.
        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.
        eval_cost: bool, optional (default=False)
            To evaluate the cost of oracles or use the cost provided by oracles.
        Returns
        -------
        Q_table: 2D array
            The Q table.
        oracle_ind_name_dict: dict
            The oracle name/index of each row of Q_table.
        """
        # Check parameter and initialize variables
        if self.X is None or self.y is None:
            raise Exception('Data matrix is not provided, use select_by_prediction_mat() instead.')
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        num_of_neighbors = n_neighbors
        if len(unlabel_index) <= 1:
            return unlabel_index
        #Q_table = np.zeros((len(oracles) + len(list(combinations_with_replacement(range(len(self.price_levels)),3))), len(unlabel_index)))  # row:oracle, col:ins
        Q_table = np.zeros((len(oracles), len(unlabel_index)))  # row:oracle, col:ins
        spv = np.shape(pred_unlab)
        # calc least_confident
        rx = np.partition(pred_unlab, spv[1] - 1, axis=1)
        rx = 1 - rx[:, spv[1] - 1]
        for unlab_ind, unlab_ins_ind in enumerate(unlabel_index):
            # evaluate oracles for each instance
            nn_dist, nn_of_selected_ins = self._nntree.kneighbors(X=self.X[unlab_ins_ind].reshape(1, -1),
                                                                  n_neighbors=num_of_neighbors,
                                                                  return_distance=True)
            nn_dist = nn_dist[0]
            nn_of_selected_ins = nn_of_selected_ins[0]
            nn_of_selected_ins = self._ini_ind[nn_of_selected_ins]  # map to the original population
            oracles_score = []
            nn_dist[nn_dist == 0] = 1e-6
            nn_sim = 1. / nn_dist
            nn_sim = nn_sim / sum(nn_sim)
            for ora_ind, ora_name in enumerate(self._oracles_iterset):
                # calc q_i(x), expertise of this instance
                oracle = oracles[ora_name]
                labels, cost = oracle.query_by_index(nn_of_selected_ins)
                #labels, cost = oracle.query_by_index(self._ini_ind)
                #oracles_score.append(sum([nn_dist[i] * (labels[i] == self.y[nn_of_selected_ins[i]]) for i in
                #                          range(num_of_neighbors)]) / num_of_neighbors)
                #oracles_score.append(sum([(labels[i] == self.y[self._ini_ind[i]]) for i in
                #                          range(len(self._ini_ind))])/len(self._ini_ind))                  
                oracles_score.append(sum([nn_sim[i] * (labels[i] == self.y[nn_of_selected_ins[i]]) for i in
                                          range(num_of_neighbors)]))
                # calc c_i, cost of each labeler
                labels, cost = oracle.query_by_index(label_index)
                if eval_cost:
                    oracles_cost = sum([labels[i] == self.y[label_index[i]] for i in range(len(label_index))]) / len(label_index)
                else:
                    oracles_cost = cost[0]
                Q_table[ora_ind, unlab_ind] = (2 * oracles_score[ora_ind] - 1) * rx[unlab_ind] / np.sqrt(max(oracles_cost, 0.0001))
        return Q_table, self._oracle_ind_name_dict




class QueryNoisyOraclesCurEM(BaseNoisyOracleQuery):
    def __init__(self, X, y, oracles, initial_labeled_indexes, if_budget = 1, unit_b = 3, unit_num = 3):
        super(QueryNoisyOraclesCurEM, self).__init__(X, y, oracles=oracles)
        # ytype = type_of_target(self.y)
        # if 'multilabel' in ytype:
        #     warnings.warn("This query strategy does not support multi-label.",
        #                   category=FunctionWarning)
        assert (isinstance(initial_labeled_indexes, collections.Iterable))
        self._ini_ind = np.asarray(initial_labeled_indexes)
        # construct a nearest neighbor object implemented by scikit-learn
        self._nntree = NearestNeighbors(metric='euclidean')
        self._nntree.fit(self.X[self._ini_ind])
        self.rho = 1
        self.num_label = len(self._ini_ind)
        oracles_score = []
        for ora_ind, ora_name in enumerate(self._oracles_iterset):
            # calc q_i(x), expertise of this instance
            oracle = oracles[ora_name]
            labels, cost = oracle.query_by_index(self._ini_ind)
            #oracles_score.append(sum([nn_dist[i] * (labels[i] == self.y[nn_of_selected_ins[i]]) for i in
            #                          range(num_of_neighbors)]) / num_of_neighbors)
            oracles_score.append(sum([labels[i] == self.y[self._ini_ind[i]] for i in
                                      range(len(self._ini_ind))]) / self.num_label)
        self.if_budget = if_budget
        self.oracles_score = oracles_score
        self.unit_b = unit_b
        self.unit_num = unit_num

    def select(self, label_index, unlabel_index, eval_cost=False, model=None, **kwargs):
        """Query from oracles. Return the index of selected instance and oracle.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        eval_cost: bool, optional (default=False)
            To evaluate the cost of oracles or use the cost provided by oracles.
        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.
        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.
        Returns
        -------
        selected_instance: int
            The index of selected instance.
        selected_oracle: int or str
            The index of selected oracle.
            If a list is given, the index of oracle will be returned.
            If a Oracles object is given, the oracle name will be returned.
        """

        if model is None:
            model = LogisticRegression(solver='liblinear')
            model.fit(self.X[label_index], self.y[label_index])
        pred_unlab, _ = _get_proba_pred(self.X[unlabel_index], model)

        n_neighbors = min(kwargs.pop('n_neighbors', 30), len(self._ini_ind) - 1)
        select_ind, select_ora = self.select_by_prediction_mat(label_index, unlabel_index, pred_unlab,
                                             n_neighbors=n_neighbors, eval_cost=eval_cost)
        self.rho = (self.rho * self.num_label + self.oracles_score[list(self._oracle_ind_name_dict.values()).index(select_ora[0])])/(self.num_label + 1)
        self.num_label = self.num_label + 1
        return select_ind, select_ora

    def select_by_prediction_mat(self, label_index, unlabel_index, predict, **kwargs):
        """Query from oracles. Return the index of selected instance and oracle.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        predict: : 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.
        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.
        eval_cost: bool, optional (default=False)
            To evaluate the cost of oracles or use the cost provided by oracles.
        Returns
        -------
        selected_instance: int
            The index of selected instance.
        selected_oracle: int or str
            The index of selected oracle.
            If a list is given, the index of oracle will be returned.
            If a Oracles object is given, the oracle name will be returned.
        """
        n_neighbors = min(kwargs.pop('n_neighbors', 30), len(self._ini_ind)-1)
        eval_cost = kwargs.pop('eval_cost', False)
        Q_table, oracle_ind_name_dict = self._calc_Q_table(label_index, unlabel_index, self._oracles, predict,
                                                           n_neighbors=n_neighbors, eval_cost=eval_cost)
        # get the instance-oracle pair
        selected_pair = np.unravel_index(np.argmax(Q_table, axis=None), Q_table.shape)
        sel_ora = oracle_ind_name_dict[selected_pair[0]]
        if not isinstance(sel_ora, list):
            sel_ora = [sel_ora]
        return [unlabel_index[selected_pair[1]]], sel_ora

    def _calc_Q_table(self, label_index, unlabel_index, oracles, pred_unlab, n_neighbors=30, eval_cost=False):
        """Query from oracles. Return the Q table and the oracle name/index of each row of Q_table.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.
        oracles: {list, alipy.oracle.Oracles}
            An alipy.oracle.Oracle object that contains all the
            available oracles or a list of oracles.
            Each oracle should be a alipy.oracle.Oracle object.
        predict: : 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.
        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.
        eval_cost: bool, optional (default=False)
            To evaluate the cost of oracles or use the cost provided by oracles.
        Returns
        -------
        Q_table: 2D array
            The Q table.
        oracle_ind_name_dict: dict
            The oracle name/index of each row of Q_table.
        """
        # Check parameter and initialize variables
        if self.X is None or self.y is None:
            raise Exception('Data matrix is not provided, use select_by_prediction_mat() instead.')
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        num_of_neighbors = n_neighbors
        if len(unlabel_index) <= 1:
            return unlabel_index
        #Q_table = np.zeros((len(oracles) + len(list(combinations_with_replacement(range(len(self.price_levels)),3))), len(unlabel_index)))  # row:oracle, col:ins
        Q_table = np.zeros((len(oracles), len(unlabel_index)))  # row:oracle, col:ins
        spv = np.shape(pred_unlab)
        # calc least_confident
        rx = np.partition(pred_unlab, spv[1] - 1, axis=1)
        rx = 1 - rx[:, spv[1] - 1]
        for unlab_ind, unlab_ins_ind in enumerate(unlabel_index):
            # evaluate oracles for each instance
            nn_dist, nn_of_selected_ins = self._nntree.kneighbors(X=self.X[unlab_ins_ind].reshape(1, -1),
                                                                  n_neighbors=num_of_neighbors,
                                                                  return_distance=True)
            nn_dist = nn_dist[0]
            nn_of_selected_ins = nn_of_selected_ins[0]
            nn_of_selected_ins = self._ini_ind[nn_of_selected_ins]  # map to the original population
            oracles_score = []
            nn_dist[nn_dist == 0] = 1e-6
            nn_sim = 1. / nn_dist
            nn_sim = nn_sim / sum(nn_sim)
            for ora_ind, ora_name in enumerate(self._oracles_iterset):
                # calc q_i(x), expertise of this instance
                oracle = oracles[ora_name]
                labels, cost = oracle.query_by_index(nn_of_selected_ins)
                oracles_score.append(sum([nn_sim[i] * (labels[i] == self.y[nn_of_selected_ins[i]]) for i in
                                          range(num_of_neighbors)]))
                #labels, cost = oracle.query_by_index(self._ini_ind)
                #oracles_score.append(sum([labels[i] == self.y[nn_of_selected_ins[i]] for i in
                #                          range(num_of_neighbors)]) / num_of_neighbors)
                #score = sum([(labels[i] == self.y[self._ini_ind[i]]) for i in
                #                          range(len(self._ini_ind))])/len(self._ini_ind)
                #oracles_score.append(score + 1.96 * np.sqrt(score*(1-score)/len(self._ini_ind)))          
                # calc c_i, cost of each labeler
                #labels, cost = oracle.query_by_index(label_index)
                if eval_cost:
                    oracles_cost = sum([labels[i] == self.y[label_index[i]] for i in range(len(label_index))]) / len(label_index)
                else:
                    oracles_cost = cost[0]
                if self.if_budget:
                    Q_table[ora_ind, unlab_ind] = (2 * (self.rho * self.num_label + oracles_score[ora_ind] * np.floor(self.unit_b / max(oracles_cost, 0.0001)))/(self.num_label + np.floor(self.unit_b / max(oracles_cost, 0.0001))) - 1) * rx[unlab_ind] * np.sqrt(self.num_label + np.floor(self.unit_b / max(oracles_cost, 0.0001)))
                else:
                    Q_table[ora_ind, unlab_ind] = (2 * (self.rho * self.num_label + oracles_score[ora_ind] * self.unit_num)/(self.num_label + self.unit_num) - 1) * rx[unlab_ind] * np.sqrt(self.num_label + self.unit_num)
        return Q_table, self._oracle_ind_name_dict



