#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:39:15 2020

@author: joelmcfarlane
"""

#%% - Reload data into python to avoid running system again.


#from Classes import *
from collections import OrderedDict as odict
from Classes import *
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
sns.set()

# - Change the name of this file.
Final_data = pk.load(open('height_and_avalanche_data','rb'))
Data_2 = pk.load(open('height_and_avalanche_data_2', 'rb'))
Big_data = pk.load(open('big_data', 'rb'))

#%% - Plot height vs Grains added (TASK 2A)

for length, dictionary in Big_data.items():
    plt.plot(dictionary['h'], label=length)
#    plt.loglog(dictionary['h'], label=length)

plt.xlabel("Grains added, $t$")
plt.ylabel("Height, $h$")

plt.legend(loc=0, title="Size $(L)$", framealpha=0.8, prop={'size':10})
plt.show()

# [14, 55, 179, 846, 3499, 13942, 56530, 224846]

#%%
#%%
#%% - Find Tc as a function of L (TASK 2b)
# So we added 200,000 grains after the steady state configurations were reached.
# Therefore the value of Tc for each system size is len(h) - 200,000

L = []
#Tc = [14, 55, 179, 846, 3499, 13942, 56530]
Tc = [17,61,214,857,3526,13750,55870,226116,901429]

for length, dictionary in Big_data.items():
    L.append(length)
    
L = np.array(L)
Tc = np.array(Tc)
    
#plt.loglog(L,Tc,marker = 'x',c='red')
plt.scatter(np.log10(L),np.log10(Tc),label='Raw Data',marker='x',c='red')

fit = np.polyfit(np.log10(L),np.log10(Tc),1)
plt.plot(np.log10(L),fit[0]*np.log10(L) + fit[1],label = 'Fitted Curve',c='blue')

plt.xlabel('log10($L$)')
plt.ylabel('log10($Tc(L)$)')

plt.legend(loc='best')

#np.poly1d(np.log(L),np.log(Tc))
# It can clearly be seen that there is a power law relation between Tc(L) and L
# This is because in a log log plot a linear pattern is seen. 
# So Tc(L) = A*L**(B)

# From the fit we find that log(L) = 1.98*log(L) - 0.0305
# Therefore L = 10**-0.0305 * L**1.98

# Use Chi-Square to determine that this is a good fit. FUTURE.


#%% - Create a scaling function to relate h and Tc

# First create smoothed out version of h, by using a moving average.
# Then plot the graph, as discussed on paper from TASK 2C

Data_heights = pd.DataFrame()

mov_period = 1


for length, dictionary in Big_data.items():
    mov = moving_avg(dictionary['h'],mov_period)
    t = np.arange(mov_period,len(dictionary['h'])+1)
    plt.loglog(t / ( length**2), mov / length,'.',ms=0.5,label = length)
    
plt.xlabel('$time/L^2$')
plt.ylabel("$<h>/L$")
plt.legend(loc='best')


#%%
""" THINK ABOUT THIS
 We see from the graph that the predictions were correct.
 When the recurrent configurations ae reached, we see a constant value of A
 reached, as discussed before.
 x = 1/L**2
 We see for large arguments that F(x) behaves linearly in L, and for small 
 arguments we see that we have that F(x) follows a power law. Hence for the
 transient states we have a power law and the steady state we have constant
 behavior. We have a sharp phase transition between the two.
 
 """
 
 
#%% - Look at how avg(h) increases with t during the transient state
# Use the largest system size as this most clearly shows the switch between 
# steady state and transient behavior. - TASK 2C continued.

h_256 = Data_2[256]['h'][0:56530]
popt,pcov = curve_fit(power,np.arange(len(h_256)),h_256)

plt.plot(h_256,label = 'Observed Data')
plt.plot(power(np.arange(len(h_256)),popt[0],popt[1]),'r--',label = 'Model')

plt.legend(loc = 'best')
plt.xlabel('$t$')
plt.ylabel('Height')

 
# We can quite clearly see that the power law fits the transient data, with
# A=1.72 and s = 0.51

#%%
#%%
#%% - Calculate the average height and standard deviation for each L:
# TASK 2E

stds = []
means = []
Ls = []

tc_list = [17,61,214,857,3526,13750,55870,226116,901429]

i = 0

for length, dictionary in Big_data.items():
    steady_hs = dictionary['h'][-tc_list[i]:]   # Last 200,000 values are in steady
                                            # state.
    stds.append(np.std(steady_hs))
    means.append(np.mean(steady_hs))
    Ls.append(length)
    
df = pd.DataFrame(list(zip(Ls,stds,means)),columns = \
                  ['Length','Standard dev','Mean'])

plt.scatter(df['Length'],df['Mean'],label =\
            'Observed Data',c='r',marker='x')
plt.xlabel('System Size $L$')
plt.ylabel('<h>')


popt, pcov = curve_fit(scale_func,df['Length'],df['Mean'])

print('a0  =',popt[0])
print('a1  =',popt[1])
print('w1  =',popt[2])

plt.plot(df['Length'],scale_func(df['Length'],\
         popt[0],popt[1],popt[2]),'r--',label = 'Correction to scaling')

plt.legend(loc = 'best')

#%% - More tests

# If we have no corrections to scaling, when we divide the y-axis by a0L, we 
# should get a constant value.

plt.scatter(df['Length'],df['Mean']/(popt[0]*df['Length']),label =\
            'Observed Data',c='r',marker='x')

plt.xlabel('System Size $L$')
plt.ylabel('<h>/a0L')

# Here we can clearly see that corrections to scaling exist, especially for
# small values of system size L. TALK TO DEMONSTRATOR ABOUT THIS

#%%
#%%
#%% - TASK 2F

# First plot the system size against the standard deviation for that size.

plt.scatter(df['Length'],df['Standard dev'],label =\
            'Observed Data',c='r',marker = 'x')

plt.xlabel('System Size $L$')
plt.ylabel('$\sigma$')

# It can clearly be seen from this plot that the fit is not linear,
# so try and plot this on a loglog scale, as it looks like a power law.
