# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 00:52:19 2023
@author: smash
"""

import math
from statistics import mean


class VariableDensityBasedUncertainty:
    def __init__(self, df, thresh, model, Beta, budget, s):
        self.df = df
        self.thresh = thresh
        self.model = model
        self.Beta = Beta
        self.budget = budget
        self.s = s
    
    def create_variable_density_based_uncertainty_accuracy_list(self):
        # Create a vector that stores all the previously trained observations
        column_names = list(self.df.columns)
        z = [(list(self.df.loc[i][0:self.df.shape[1]-1]), self.df.loc[i][self.df.shape[1]-1]) for i in range(self.df.shape[0])]
        
        training_data_vec = []
        prob1 = []
        prob0 = []
        diff = []
        acc = []
        correct_cnt = 0
        cost = 0
        total_data = 0
        
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
                    training_data_vec.append(a)
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
                    
                    # Similarity Measure is calculated using a distance measure, Using the last set of observations used for training the model.
                    # Code to find the distance measure , with a window
                    # Assuming window size as 500
                    distance_measure = []
                    val1 = list(a.values())
                    
                    for m in training_data_vec[-500:]:
                        val2 = list(m.values())
                        distance = math.dist(val1, val2)
                        distance_measure.append(distance)
                    
                    dist = mean(distance_measure)
                    
                    if (abs(prob_0 - prob_1)) * dist ** self.Beta <= self.thresh:
                        self.model = self.model.learn_one(a, b)
                        cost += 1
                        training_data_vec.append(a)
                        self.thresh = self.thresh * (1 - self.s)
                    else:
                        self.thresh = self.thresh * (1 - self.s)
        
        accuracy_measure = []
        
        for p in range(math.floor(len(z) / 1000)):
            accuracy_measure.append(mean(acc[1:(p + 1) * 1000]))
       
        return accuracy_measure
