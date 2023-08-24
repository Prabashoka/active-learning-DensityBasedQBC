# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 01:55:28 2023
@author: smash
"""

import math
import random
from statistics import mean

def voteEntropy(committee):
    # Converting our list to filter list
    committeeUnique = [x for i, x in enumerate(committee) if x not in committee[:i]]
    statistic = []
    for i in committeeUnique:
        val = (committee.count(i) / len(committee)) * math.log(committee.count(i) / len(committee))
        statistic.append(val)
    return -sum(statistic)

class RandomizedDensityBasedQBC:
        
    def __init__(self, df, thresh, model, Beta, Committee, init_training, Propotion):
        self.df = df
        self.thresh = thresh
        self.model = model
        self.Beta = Beta
        self.initial_training = init_training
        self.Committee = Committee
        self.Propotion = Propotion
    
    def create_randomized_density_based_qbc_accuracy_list(self):
        # Create a vector that stores all the previously trained observations
        column_names = [col for col in self.df.columns]
        z = [(list(self.df.loc[i][0:self.df.shape[1]-1]), self.df.loc[i][self.df.shape[1]-1]) for i in range(self.df.shape[0])]
        
        training_data_vec = [] 
        acc = []
        correct_cnt = 0
        training_data = 2000
        total_data = 0
        model_1 = self.Committee[0] 
        model_2 = self.Committee[1]
        model_3 = self.Committee[2]
       
        for x in range(len(z)):
            committee = []
            a = {}
            data = z[x][0]
            for p in range(len(column_names) - 1):
                a[column_names[p]] = data[p]
            b = z[x][1]
            total_data += 1 
            if x < self.initial_training:
                self.model = self.model.learn_one(a, b)
                # Training the Committee using initial data
                model_1 = model_1.learn_one(a, b) # Adaptive Random Forest Classifier
                model_2 = model_2.learn_one(a, b) # Bagging Classifier
                model_3 = model_3.learn_one(a, b) # Leverage Bagging Classifier
                
                training_data_vec.append(a)
            else:
                pred = self.model.predict_one(a)
                if pred == b:
                    acc.append(1)
                    correct_cnt += 1
                else:
                    acc.append(0)
                # Making predictions Using Committees
                model_1_pred = model_1.predict_one(a) # Adaptive Random Forest Classifier
                committee.append(model_1_pred)
                model_2_pred = model_2.predict_one(a) # Bagging Classifier
                if model_2_pred == "True":
                    committee.append(1.0) 
                else: 
                    committee.append(0.0)
                model_3_pred = model_3.predict_one(a) # Leverage Bagging Classifier
                if model_3_pred == "True":
                    committee.append(1.0)
                else: 
                    committee.append(0.0)
                
                # Distance measure calculation
                distance_measure = []
                val1 = list(a.values())
                for m in training_data_vec[-500:]:
                    val2 = list(m.values())
                    distance = math.dist(val1, val2)
                    distance_measure.append(distance)
                dist = mean(distance_measure)
      
                if voteEntropy(committee) * dist**(self.Beta) >= self.thresh or random.random() < self.Propotion:
                    self.model = self.model.learn_one(a, b)
                    training_data += 1
                    training_data_vec.append(a)
      
        from statistics import mean
        import math
        accuracy_measure = []
        accuracy_measure.append(training_data)
        for p in range(math.floor(len(z) / 1000)):
            accuracy_measure.append(mean(acc[1:(p + 1) * 1000]))
        
        return accuracy_measure
