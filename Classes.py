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
#import seaborn as sns
import collections
import pandas as pd
from tqdm import *
#sns.set()

from logbin230119 import *

class Site:
    
    """
    Initialise a site with a random threshold, zero height and slope. 
    Add in slope in the System in another class.
    """
    
    def __init__(self,height = 0,slope = 0, \
                 special = False,prob = 0.5):
        self.h = height
        self.z = slope
        self.t = np.random.randint(1,3)
        self.end = special  # This is only used to indicate the end site.
        self.prob = prob
        
    def reset_thresh(self):
        "Randomly set threshold value to 1 or 2"
        self.t = np.random.choice([1,2],p=[self.prob,1-self.prob])
        
    def set_z(self,slope):
        self.z = slope
    
    def set_h(self,height):
        self.h = height
        
#    def check_end(self):
#        "Check if this is an end site"
#        if self.end == False:
#            return False
#        elif self.end == True:
#            return True
    

class System:
    
    """
    Create a system composed of a list of sites.
    
    Structured so we are based on slopes, not the heights. Will be able to
    calculate the heights at any given time.
    
    Note that steady state config is reached when a pile topples off the last
    site.
    
    """

    def __init__(self,size):
# =============================================================================
#         site_list = []
#         for i in range(size):
#             site_list.append(Site())    # Slow old code
# =============================================================================
        site_list = np.array([Site() for i in range(size)])  # Faster code
        
        site_list = np.append(site_list,Site(special = True))
        
        self.sites = site_list
        self.size = size
        self.ava_sizes = []
        self.reach_edge = False  
        self.Tc = None


    def show_h(self):
        print('The heights are:')
        print([site.h for site in self.sites])
            
    def find_h(self):    
        return np.array([height.h for height in self.sites])
    
    def find_z(self):    
        return np.array([height.h for height in self.sites])
            
    def show_z(self):
        print('The slopes are:')
        print([site.z for site in self.sites])
            
    def drive(self):
        "Implement a drive at site i=0" # Add one to slope and height        
        self.sites[0].z += 1
        self.sites[0].h += 1 

            
    def relax_i(self,i):    # FIX THIS
        "Perform relaxation at site i" #Removed the set_z function,
                                       #to speed up operation.
        if self.reach_edge == False and i == self.size-1:
            self.reach_edge = True
            self.Tc = np.sum(self.find_h())
        
        if i == 0:
            self.sites[i].z -= 2
            self.sites[i+1].z += 1
            self.sites[0].h -= 1
        
        elif i == self.size - 1:
            self.sites[i].z -= 1
            self.sites[i-1].z += 1
            
        else:
            self.sites[i].z -= 2
            self.sites[i+1].z += 1
            self.sites[i-1].z += 1   # Implement rules from Oslo algo
    
    def site_relax_list(self):
        "Create list of sites to relax - TESTED"
        relax_list = np.array([i for i in range(self.size) \
                      if self.sites[i].z > self.sites[i].t]) # Faster code
        return relax_list
                
    def res_select_thresh(self,relax_list):
        "Reset select threshold values" # Note check with a demonstrator if this works
        for k in range(len(relax_list)):
            site_no = relax_list[k]
            self.sites[site_no].reset_thresh()
        
#        map(lambda k : self.sites[k].reset_thresh(),relax_list) # Does not currently work
            
    def update_heights(self):
        " As we work using the slope values, this function updates height based"
        " on those slope values and the initial height."
        height_list = np.array([self.sites[0].h])  # Start with initial height, then add
                                                   # other heights
        for i in range(self.size-1):
            height_i = height_list[-1] - self.sites[i].z
            height_list = np.append(height_list,height_i)

        for i in range(self.size):
            self.sites[i].h = height_list[i]
#        map(lambda i : self.sites.h = height_list[i],height_list) 
        # Tried to use map functiomn to map the height to the site, much faster than for loop.
        # Check this with a demonstrator
            
    def add_grain(self):
        "Function that adds a grain then performs all relaxations caused by"
        " that grain. "
        self.drive()
        relax_list = self.site_relax_list()
        
        ava_size = 0

        while len(relax_list) != 0:
            
            first_index = relax_list[0]
            self.relax_i(first_index)
            self.res_select_thresh([first_index])   #List as method takes list.
            ava_size += 1
            relax_list = self.site_relax_list()
            
#            print(first_index)
#            print(ava_size)
            
        self.ava_sizes.append(ava_size)
        
        self.update_heights()
        
    def plot(self):
        
        x_axis = np.arange(self.size)
        
        y_axis = []
        
        for i in range(self.size):
            y_axis.append(self.sites[i].h)
        
        plt.bar(x_axis,y_axis)
        plt.xlabel('Pile')
        plt.ylabel('Height')
        

#        
        
def plot_hvt(size,iterations):
    " Plot the height vs time for a particular sized system. "
    sys = System(size)
    
    heights = [] 
    for i in range(iterations):
        sys.add_grain()
        h = sys.sites[0].h
        heights.append(h)
        
    x_axis = np.arange(iterations)
    
    plt.plot(x_axis,heights,label = 'Size L = '+str(size))
    plt.xlabel('Iterations')
    plt.ylabel('Height')
    
    plt.legend(loc= 'best')
        
        
            
        
            
def find_avg_h(iterations = 10,size = 10):
    
    "Finds the average height at site i=1, averaged over i iterations"
    
    height_list = []
    
    for i in range(iterations):
        
        s = System(size)
        
        for i in range(1000):
            s.add_grain()
            
        height_list.append(s.sites[0].h)
        
    return np.array(height_list).mean()


            
            
def plot_animate(System,range_):
    " Display an animated plot with the grains being added in real time "
    for i in range(range_):
        System.add_grain()
        
        x_axis = np.arange(System.size)
            
        y_axis = [System.sites[i].h for i in range(System.size)]
            
        plt.bar(x_axis,y_axis,color = 'blue')
        plt.pause(0.001)
        plt.xlabel('Pile')
        plt.ylabel('Height')
        plt.ylim(0,10)
        
    System.show_h()
    
    

    
    
            
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