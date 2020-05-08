#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:56:14 2020

@author: joelmcfarlane

Testing module - Complexity Project

"""
#%%
#from Classes import *
#
## Test End function
#test = Site(special = True)
#
#test.check_end()
#%%
from Classes import *

# Test threshold value
test = Site()

one = test.t
print(one)

test.reset_thresh()

two = test.t
print(two)

test.reset_thresh()

three = test.t
print(three)
#%%

from Classes import *

# Test setting values

test = Site(7,6)

print(test.h,test.z)
test.set_h(10)
test.set_z(2)

print(test.h,test.z)

#%%
from Classes import *

# Test generating a system, then test drive.

test_sys = System(10)

node_1 = test_sys.sites[0]
print(node_1.h)

test_sys.drive()

node_1 = test_sys.sites[0]
print(node_1.h)

#%%
from Classes import *
# Test relaxation. Note only currently displays correct z values
test_sys = System(10)

node_1 = test_sys.sites[0]
node_2 = test_sys.sites[1]
#node_3 = test_sys.sites[1]
#print(node_3.z)
print(node_1.z)
print(node_2.z)


test_sys.relax_i(0)

node_1 = test_sys.sites[0]
node_2 = test_sys.sites[1]
#node_3 = test_sys.sites[1]
#print(node_3.z)
print(node_1.z)
print(node_2.z)

#%% Test reseting of threshold values
test_sys = System(10)

for i in range(len(test_sys.sites)):
    thresh = test_sys.sites[i].t
    print(thresh)
    


relax_list_t = [0,1,2,3,4,5,6,7,8,9]

test_sys.res_select_thresh(relax_list_t)

print()
for i in range(len(test_sys.sites)):
    thresh = test_sys.sites[i].t
    print(thresh)

#%% - Test updated heights
    
from Classes import *

test_sys = System(10)

test_sys.show_h()

test_sys.drive()
test_sys.drive()
test_sys.drive()
test_sys.sites[4].set_h(100)
test_sys.drive()
test_sys.drive()

test_sys.update_heights()

test_sys.show_h()
#test

list_2 = test_sys.sites


test_sys.site_relax_list()


test_sys

#%% - Test drive and relax


from Classes import *

test_sys = System(100)


for i in range(10000):
    test_sys.add_grain()
    

test_sys.show_h()

ava_sizes = test_sys.ava_sizes

#avg = find_avg_h(test_sys)

# Tested by setting p = 1 and therefore defaulting to the BTW model.

test_sys.plot()

#%% - Make sure model is working properly. Test first by finding average i=0 
# height across 10 iterations
from Classes import *
find_avg_h(iterations=10,size=32)

# As expected for L=16 the average height at i=1 is 26.2 ~ 26.5

# As expected for L=32 the average height at i=1 is 53.6 ~ 53.9

#%% - Animate the system
from Classes import *
test_sys = System(4)
#print(test_sys.size)

#for site in test_sys.sites:
#    print(site.t)
    
plot_animate(test_sys,20)

#for site in test_sys.sites:
#    print(site.t)
test_sys.show_z()

#%% - Test the steady state configuration indicator.

test_sys = System(4)

for i in range(30):
    
    test_sys.add_grain()
    
test_sys.reach_edge


#%% - Task 2a. Plot height vs time
from Classes import *

size_list = [4,8,16,32,64,128,256]
#size_list = [8,16]

iter_list = [2500,2500,2500,2500,6000,17000,50000]

for i in range(len(size_list)):
    plot_hvt(size_list[i],100000)
    
    

    


#%% - Test find Tc

from Classes import *

test_sys = System(8)



for i in range(1000):
    test_sys.add_grain()
    
print(test_sys.Tc)
test_sys.find_h()
test_sys.show_h()

#%% - Task 2b. Plot Tc vs L
from Classes import *
Tc_v_L()

#%% - Test Probability calculation
from Classes import *
array = Final_data

test_df = frequency(steady_hs)

print(sum(test_df['Probability'])) # Test if sum equals one
type(test_df)    # Make sure we have a dataframe.


#%% - Test faster code
from Classes_2 import *
# Test out faster code to help run more systems
test_sys = System(32)

for i in range(10000):
#    print(i)
    test_sys.add_grain()
    
h = test_sys.h
s = test_sys.ava_sizes
z= test_sys.sites
Tc = test_sys.Tc

print(np.mean(h))

    

