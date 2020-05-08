#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:57:25 2020

@author: joelmcfarlane
"""

#%% - Convert df to dictionary

import pandas as pd
import pickle as pk
from collections import OrderedDict as odict



data2_4 = pd.read_csv('4.csv',skiprows=1)
data2_8 = pd.read_csv('8.csv',skiprows=1)
data2_16 = pd.read_csv('16.csv',skiprows=1)
data2_32 = pd.read_csv('32.csv',skiprows=1)
data2_64 = pd.read_csv('64.csv',skiprows=1)
data2_128 = pd.read_csv('128.csv',skiprows=1)
data2_256 = pd.read_csv('256.csv',skiprows=1)
#data2_512 = pd.read_csv('512.csv',skiprows=1)
#data2_1024 = pd.read_csv('1024.csv',skiprows=1)


#%%
list_data = [data2_4,data2_8,data2_16,data2_32,data2_64,data2_128,data2_256]#,data2_512,\
             # data2_1024]

All_data = odict()
    
lengths = [4,8,16,32,64,128,256]#,512,1024
i=0

for df in list_data:
    

    df = df.rename(columns={"pile_size": "h", "avalanche_size": "s"})
    h = list(df['h'])
    s = list(df['s'])
    dictionary = {'h':h, 's':s}
    
    All_data[lengths[i]] = dictionary
    i+=1
    
pk.dump(All_data,open('big_data', 'wb'))
