# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:39:11 2019

@author: valdis
"""

import numpy as np

class ExponentDecay():
    def __init__(self, border_epoch=0, learning_rate=1e-4, exp_power=1.0):    
        self.learning_rate = learning_rate
        self.exp_power = exp_power; self.border_epoch = border_epoch
        if self.border_epoch > 0:
            self.step_per_epoch = self.learning_rate / self.border_epoch 
    
    def __call__(self, epoch):
        zero_value = self.learning_rate/10
        if epoch < self.border_epoch:
            rate = zero_value + self.step_per_epoch*(epoch)
        else:
            rate = (zero_value + self.learning_rate) * np.exp(-self.exp_power*(epoch - self.border_epoch))
        
        return float(rate)
    
    
class StepDecay():
    def __init__(self, border_epoch=0, learning_rate=1e-4, step_factor=0.25, step_drop=2):   
        self.learning_rate = learning_rate; self.step_drop = step_drop
        self.step_factor = step_factor; self.border_epoch = border_epoch
        if self.border_epoch > 0:
            self.step_per_epoch = self.learning_rate / self.border_epoch         
        
    def __call__(self, epoch):
        zero_value = self.learning_rate/10
        if epoch < self.border_epoch:
            rate = zero_value + self.step_per_epoch*(epoch)
        else:
            exp = np.floor((1 + epoch - self.border_epoch) / self.step_drop)
            rate = (zero_value + self.learning_rate) * (self.step_factor ** exp)
        
        return float(rate)

class ConstantDecay():
    def __init__(self, border_epoch=0, learning_rate=1e-4):   
        self.border_epoch = border_epoch; self.learning_rate = learning_rate
        if self.border_epoch > 0:
            self.step_per_epoch = self.learning_rate / self.border_epoch        
    
    def __call__(self, epoch):
        zero_value = self.learning_rate/10
        if epoch < self.border_epoch:
            rate = zero_value + self.step_per_epoch*(epoch)
        else:
            rate = zero_value + self.learning_rate
        
        return float(rate)