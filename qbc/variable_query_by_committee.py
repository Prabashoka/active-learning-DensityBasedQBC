# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 19:24:31 2023
@author: smash
"""

import math

def voteEntropy(committee):
    # converting our list to a filtered list
    committeeUnique = [x for i, x in enumerate(committee) if x not in committee[:i]]
    statistic = []
    for i in committeeUnique:
        val = (committee.count(i) / len(committee)) * math.log(committee.count(i) / len(committee))
        statistic.append(val)
    return -sum(statistic)

class VariableQueryByCommittee:
    def __init__(self, df, model, Committee, thresh, budget, s):
        self.model = model
        self.df = df
        self.thresh = thresh
        self.Committee = Committee
        self.budget = budget
        self.s = s

    def create_query_by_committee_accuracy_list(self):
        column_names = [col for col in self.df.columns]
        z = [(list(self.df.loc[i][0:self.df.shape[1]-1]), self.df.loc[i][self.df.shape[1]-1]) for i in range(self.df.shape[0])]

        acc = []
        correct_cnt = 0
        training_data = 2000
        cost = 0

        model_1 = self.Committee[0]
        model_2 = self.Committee[1]
        model_3 = self.Committee[2]

        for x in range(len(z)):
            committee_output = []
            a = {}
            data = z[x][0]
            for p in range(len(column_names) - 1):
                a[column_names[p]] = data[p]
            b = z[x][1]

            if cost / len(z) <= self.budget:
                if x <= 5:
                    self.model = self.model.learn_one(a, b)
                    # Training the Committee using initial data
                    model_1 = model_1.learn_one(a, b)  # Adaptive Random Forest Classifier
                    model_2 = model_2.learn_one(a, b)  # Bagging Classifier
                    model_3 = model_3.learn_one(a, b)  # Leverage Bagging Classifier

                else:
                    pred = self.model.predict_one(a)
                    if pred == b:
                        acc.append(1)
                        correct_cnt += 1
                    else:
                        acc.append(0)
                    # Making predictions Using Committees
                    model_1_pred = model_1.predict_one(a)  # Adaptive Random Forest Classifier
                    committee_output.append(model_1_pred)
                    model_2_pred = model_2.predict_one(a)  # Bagging Classifier
                    if model_2_pred == "True":
                        committee_output.append(1.0)
                    else:
                        committee_output.append(0.0)
                    model_3_pred = model_3.predict_one(a)  # Leverage Bagging Classifier
                    if model_3_pred == "True":
                        committee_output.append(1.0)
                    else:
                        committee_output.append(0.0)

                    if voteEntropy(committee_output) >= self.thresh:
                        self.model = self.model.learn_one(a, b)
                        model_1 = model_1.learn_one(a, b)  # Adaptive Random Forest Classifier
                        model_2 = model_2.learn_one(a, b)  # Bagging Classifier
                        model_3 = model_3.learn_one(a, b)  # Leverage Bagging Classifier

                        cost += 1
                        self.thresh = self.thresh * (1 + self.s)
                    else:
                        self.thresh = self.thresh * (1 - self.s)

        from statistics import mean
        import math
        accuracy_measure = []
        accuracy_measure.append(cost)
        for p in range(math.floor(len(z) / 1000)):
            accuracy_measure.append(mean(acc[1:(p + 1) * 1000]))
        return accuracy_measure
