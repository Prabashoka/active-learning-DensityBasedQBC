# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:15:41 2023
@author: smash
"""
from statistics import mean
import math
from river import evaluate
from river import metrics
from river import tree
from river import datasets
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd


class FixedUncertainty:
    def __init__(self, df, model, budget, s):
        self.model = model
        self.df = df
        self.thresh = 1
        self.budget = budget
        self.s = s

    def create_uncertainty_accuracy_list(self):
        column_names = []
        for col in self.df.columns:
            column_names.append(col)
        z = []
        for i in range(self.df.shape[0]):
            z.append((list(self.df.loc[i][0:self.df.shape[1]-1]), self.df.loc[i][self.df.shape[1]-1]))
        prob1 = []
        prob0 = []
        diff = []
        acc = []
        correct_cnt = 0
        total_data = 0
        cost = 0
        for x in range(len(z)):
            a = {}
            data = z[x][0]
            for p in range(len(column_names)-1):
                a[column_names[p]] = data[p]
            b = z[x][1]
            total_data += 1
            if x == 0:
                self.model = self.model.learn_one(a, b)
            else:
                pred = self.model.predict_one(a)
                if pred == b:
                    acc.append(1)
                    correct_cnt += 1
                else:
                    acc.append(0)
                prob_0 = self.model.predict_proba_one(a)[0]
                prob_1 = self.model.predict_proba_one(a)[1]
                prob1.append(prob_1)
                prob0.append(prob_0)
                diff.append(prob_1 - prob_0)
                if abs(prob_0 - prob_1) <= self.thresh:
                    self.model = self.model.learn_one(a, b)
                    cost += 1

        accuracy_measure = []
        for p in range(math.floor(len(z) / 1000)):
            accuracy_measure.append(mean(acc[1:(p + 1) * 1000]))

        return accuracy_measure
