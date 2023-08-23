# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:28:14 2023
@author: smash
"""

import math
from statistics import mean

class MLModel:
    def __init__(self, df, model, initial_train):
        self.df = df
        self.model = model
        self.initial_train = initial_train

    def create_ml_model_accuracy_list(self):
        column_names = [col for col in self.df.columns]
        z = [(list(self.df.loc[i][0:self.df.shape[1]-1]), self.df.loc[i][self.df.shape[1]-1]) for i in range(self.df.shape[0])]
        prob1 = []
        prob0 = []
        diff = []
        acc = []
        correct_cnt = 0
        training_data = self.initial_train
        total_data = 0

        for x in range(len(z)):
            a = {}
            data = z[x][0]
            for p in range(len(column_names) - 1):
                a[column_names[p]] = data[p]
            b = z[x][1]
            total_data += 1 

            if x < training_data:
                self.model = self.model.learn_one(a, b)
            else:
                pred = self.model.predict_one(a)
                if pred == b:
                    acc.append(1)
                    correct_cnt += 1
                else:
                    acc.append(0)
                self.model = self.model.learn_one(a, b)
                training_data += 1

        from statistics import mean
        import math
        accuracy_measure = []

        for p in range(math.floor(len(z) / 1000)):
            accuracy_measure.append(mean(acc[1:(p + 1) * 1000]))

        return accuracy_measure
