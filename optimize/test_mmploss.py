# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:39:13 2018

@author: SrivatsanPC
"""

from mmp_loss import *
import numpy as np
#Assuming there are three states.
target_action = np.array([[0.1, 0.2, 0.1, 0.6], [ 0.5,0.2,0.2,0.1],[0,0.9,0,0.1]])
given_action_1 = np.array([[0.9,0.1,0,0],[0.25,0.25,0.25,0.25],[0,0.9,0,0.1]])
given_action_2 = np.array([[0.1, 0.2, 0.1, 0.6],[0.25,0.25,0.25,0.25],[0.9,0,0.1,0]])

state_frequencies = [100,25,25]
p1 = action_variation_loss(target_action, given_action_1, state_frequencies)
p2 = action_variation_loss(target_action, given_action_2, state_frequencies)
print("Penalty for deviation from high  freq state is" ,p1)
print("Penalty for deviation from low  freq state is",p2 )

#Test that the penalty for deviation from a high frequency state is more than low frequency state.
assert(p1>p2)