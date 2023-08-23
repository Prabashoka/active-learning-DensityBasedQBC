# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 08:57:24 2023
@author: smash
"""

from statistics import mean
import math
import numpy as np


class RandomizedVariableUncertainty:
    def __init__(self, df, model, budget, s, mean, delta):
        self.model = model
        self.df = df
        self.thresh = 1
        self.budget = budget
        self.s = s
        self.mean = mean
        self.delta = delta

    def create_randomized_variable_uncertainty_accuracy_list(self):
        column_names = []
        for col in self.df.columns:
            column_names.append(col)
        z = []
        for i in range(self.df.shape[0]):
            z.append((list(self.df.loc[i][0:self.df.shape[1] - 1]), self.df.loc[i][self.df.shape[1] - 1]))
        prob1 = []
        prob0 = []
        diff = []
        acc = []
        correct_cnt = 0
        total_data = 0
        cost = 0
        mu = self.mean  # mean
        sigma = self.delta  # standard deviation

        for x in range(len(z)):
            a = {}
            data = z[x][0]
            for p in range(len(column_names) - 1):
                a[column_names[p]] = data[p]
            b = z[x][1]
            total_data += 1

            if cost / len(z) <= self.budget:
                if x <= 5:
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
                    random_number = np.random.normal(mu, sigma)
                    if abs(prob_0 - prob_1) <= self.thresh * random_number:
                        self.model = self.model.learn_one(a, b)
                        cost += 1
                        self.thresh = self.thresh * (1 - self.s)
                    else:
                        self.thresh = self.thresh * (1 + self.s)

        accuracy_measure = []

        for p in range(math.floor(len(z) / 1000)):
            accuracy_measure.append(mean(acc[1:(p + 1) * 1000]))

        return accuracy_measure
