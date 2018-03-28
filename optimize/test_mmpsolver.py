# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:01:11 2018

@author: SrivatsanPC
"""

from mmpsolver import *


#Naive max margin
mmpsolver = MMTOpt(calc_pi_loss = False, fixed_loss = 1, soft = False)
mu_expert, mu_init = [5,5], [0,0]
#import pdb; pdb.set_trace()
mmpsolver.optimize(mu_expert, mu_init)
mu_2, mu_3, mu_4 = [1,1],[3,5],[5,3]
mmpsolver.optimize(mu_expert, mu_2)
mmpsolver.optimize(mu_expert, mu_3)
mmpsolver.optimize(mu_expert, mu_4)
