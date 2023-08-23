# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:15:41 2023
@author: smash
"""

from statistics import mean
import math
import random

class RandomStrategy:
    def __init__(self, df, model, proportion, init_training):
        self.model = model
        self.df = df
        self.proportion = proportion
        self.initial_training = init_training
        
    def create_random_strategy_accuracy_list(self):
        column_names = [col for col in self.df.columns]
        z = [(list(self.df.loc[i][0:self.df.shape[1]-1]), self.df.loc[i][self.df.shape[1]-1]) for i in range(self.df.shape[0])]
        
        acc = []
        correct_cnt = 0
        training_data = self.initial_training
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
                
                if random.random() < self.proportion:
                    self.model = self.model.learn_one(a, b)
                    training_data += 1
        
        self.total_training_data = training_data
        accuracy_measure = []
        accuracy_measure.append(training_data)
        
        for p in range(math.floor(len(z) / 1000)):
            accuracy_measure.append(mean(acc[1:(p + 1) * 1000]))
         
        return accuracy_measure
