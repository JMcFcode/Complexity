#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:54:52 2020

@author: joelmcfarlane

Complexity and Networks Project

Module for classes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import *
import seaborn as sns
import collections
import pandas as pd
from tqdm import *
sns.set()

from logbin230119 import *

    

class System:
    
    """
    Create a system composed of a list of sites.
    
    Structured so we are based on slopes, not the heights. Will be able to
    calculate the heights at any given time.
    
    Note that steady state config is reached when a pile topples off the last
    site.
    
    Note that this is the second edition of the code, which is significantly
    faster than the first.
    
    """

    def __init__(self,size):
        
        self.prob=0.5
        self.sites = np.zeros(size)
        self.t = np.random.choice([1,2],size = size,p = [self.prob,1-self.prob])
        self.size = size
        self.h = []
        self.ava_sizes = []
        self.reach_edge = False  
        self.Tc = None
        
        self.grains_added = 0
        
    def drive(self):
        "Implement a drive at site i=0" # Add one to slope and height        
        self.sites[0] += 1

            
    def relax_i(self,i):    # FIX THIS
        "Perform relaxation at site i" #Removed the set_z function,
                                       #to speed up operation.
        if self.reach_edge == False and i == self.size-1:
            self.reach_edge = True
            self.Tc = self.grains_added #Find Tc Later
        
        if i == 0:
            self.sites[i] -= 2
            self.sites[i+1] += 1
            self.t[i] = np.random.choice([1,2],p=[self.prob,1-self.prob])
        
        elif i == self.size - 1:
            self.sites[i] -= 1
            self.sites[i-1] += 1
            self.t[i] = np.random.choice([1,2],p=[self.prob,1-self.prob])
            
        else:
            self.sites[i] -= 2
            self.sites[i+1] += 1
            self.sites[i-1] += 1   # Implement rules from Oslo algo
            self.t[i] = np.random.choice([1,2],p=[self.prob,1-self.prob])
    
    def site_relax_list(self):
        "Create list of sites to relax - TESTED"
        relax_list = np.array([i for i in range(self.size) if self.sites[i] > self.t[i]]) # Faster code
        return relax_list
            
            
    def add_grain(self):
        "Function that adds a grain then performs all relaxations caused by"
        " that grain. "
        self.drive()
        self.grains_added += 1
        relax_list = self.site_relax_list()
        
        ava_size = 0

        while len(relax_list) != 0:

            first_index = relax_list[0]
            self.relax_i(first_index)
            ava_size += 1
            relax_list = self.site_relax_list()
            
        self.h.append(sum(self.sites))
        self.ava_sizes.append(ava_size)
        

    
            
def moving_avg(values, win):
    " Calcualate the moving average in the fastest way computationally. "
    curr_win = np.repeat(1.0, win)/win
    mov_avg= np.convolve(values, curr_win, 'valid')
    return mov_avg

def power(t,A,s):
    " Simple power law function, for fitting. "
    return A*t**s

def scale_func(L,a0,a1,w1):
    " Scaling function from the notes. "
    h = a0 * L * (1 - a1 * L**(-w1))
    return h

def linear(x,m,c):
    "Function for linear regression"
    y = m*x+c
    return y

def gaussian(x,std,mu):
    y = 1/(std*np.sqrt(2*np.pi))*np.exp(-0.5*(x-mu)**2/std**2)
    return y

def frequency(array):
    " Calculate the probability of a certain height / avalanche size "
    
    freq = collections.Counter(array)
    df = pd.DataFrame.from_dict(freq, orient='index').reset_index()
    df.columns = ['Height', 'Frequency']
    df = df.sort_values(by = ['Height'])
    df['Probability'] = df['Frequency'] / len(array)
    
    return df