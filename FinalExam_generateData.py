#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:32:52 2018

@author: qichengwang
"""

import numpy as np
""" please use the last 5 digits of your student ID as the seed """

np.random.seed(44391) ## 12345 this is the seed


""" 5 data instance ,3-dim feature """
A = np.random.random((5,3))

## task : compute the coviance matrix of A
## please sumbit your code and results to the blackboard

a= np.array(A)
mean_row=np.array(a.mean(1))
a=a-(mean_row.reshape(-1,1))
mean = np.mean(a)
covar=np.cov(a)
covar=(covar*(a.shape[1]-1))
covar=covar/(a.shape[1])

print("Covariance matrix\n")
print(covar)

print("\nEigen values, eigen vectors\n")
print(np.linalg.eig(covar))
