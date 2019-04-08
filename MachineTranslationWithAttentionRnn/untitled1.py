# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 08:46:07 2019

@author: ADMIN
"""
import torch
import torch.nn as nn
import numpy as np
#embedding = nn.Embedding(10, 3, padding_idx=0)
#input = torch.LongTensor([[0,2,0,5]])
#print(embedding(input))
a1=[2,2,2,1,1,0]
a2 = [1,2,3,4,5,6]
a=np.array(sorted(list(zip(a1,a2))))
print(np.split(a[:, 1], np.cumsum(np.unique(a[:, 0], )[1])[:-1]))






