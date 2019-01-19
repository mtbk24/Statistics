# -*- coding: utf-8 -*-
"""
Created on Sat May 27 12:35:14 2017

@author: KimiZ
"""
from __future__ import division
from collections import OrderedDict

import os, glob
import numpy as np
from numpy import exp, log, log10

import matplotlib.pyplot as plt

from scipy.stats import chi2

#%%

#        CHI-SQUARED DISTRIBUTION WITH VARIOUS DEGREES OF FREEDOM.

dof         = [1,2,3,4,6,9]
colors      = ['yellow', 'lime', 'cyan', 'blue', 'purple', 'red']


plt.figure(figsize=(11,7))

for i,df in enumerate(dof):
    x = np.linspace(chi2.ppf(0.0, df),
                  chi2.ppf(0.999, df), 1000)
                  
    plt.plot(x, chi2.pdf(x, df),
             '-', lw=2, color=colors[i], alpha=0.6, label= "dof=%i"%df)

plt.xlim(0,8)   
plt.ylim(0, 0.5)      
plt.ylabel("Probability", fontsize=16)
plt.xlabel(r"$\Delta \chi^2$ per degree of freedom", fontsize=16)
plt.legend(loc='best')
plt.title(r'Probability Density Function for $\chi^2$', fontsize=16)



#%%

#       TO GET THE PROBABILITY TO BE 0.01 OF GETTING A CHANGE IN DELTA CHI-SQUARED WHEN YOU HAVE A DIFFERENCE OF 1 DEGREE OF FREEDOM:

#   USE THE PPF -- inverse of the CDF function -- quantiles.
#               chi2.ppf(PROB, dof)
#   PROB - probability of NO change  = 1 - probability of change.
#   PROB = 1 - 0.01 = 0.99
#  So, if we choose a probability of  0.01 and we have 1 degree of freedom, what will the level of our change be at that probability? How many chi-squared units?
#  0.01 IS THE p-value.  
#  https://en.wikipedia.org/wiki/Chi-squared_distribution


df = 1  # DEGREES OF FREEDOM

chi2.ppf(0.99, df)  # =  6.6348966010212127 units of chi-squared.
# THE PPF IS THE PERCENT POINT FUNCTION, I.E., THE INVERSE OF THE CDF -- which is where your percentiles come from).

# THE INVERSE SURVIVAL FUNCTION GIVES YOU THE SAME THING WHEN YOU USE THE PROBABILITY OF 0.01 IN PLACE OF THE 0.99.
chi2.isf(0.01, df) # =  6.6348966010212171

#  SEE https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.stats.chi2.html FOR MORE DETAILS.

#%%
chi2.isf(0.05, 7)


#%%

x = np.linspace(chi2.ppf(0.0, df),
                  chi2.ppf(0.999, df), 1000)
                  
y = chi2.pdf(x, df)


#%%

df = 2

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

x = np.linspace(chi2.ppf(0.01, df),
              chi2.ppf(0.99999999999, df), 1000)
              
ax.plot(x, chi2.pdf(x, df),
         'r-', lw=5, alpha=0.6, label='chi2 pdf')

#%%

df = 353

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

x = np.linspace(chi2.ppf(0.01, df),
              chi2.ppf(0.99999999999, df), 1000)
              
ax.plot(x, chi2.pdf(x, df),
         'r-', lw=5, alpha=0.6, label='chi2 pdf')


#%%
dof = [1,2,3,4,6,9]
colors = ['yellow', 'lime', 'cyan', 'blue', 'purple', 'red']

#fig, ax = plt.subplots(1, 1)

plt.figure(figsize=(11,7))

for i,df in enumerate(dof):
    x = np.linspace(chi2.ppf(0.0, df),
                  chi2.ppf(0.999, df), 1000)
                  
    plt.plot(x, chi2.pdf(x, df),
             '-', lw=2, color=colors[i], alpha=0.6, label= "dof=%i"%df)

plt.xlim(0,8)   
plt.ylim(0, 0.5)      
plt.ylabel("Probability", fontsize=16)
plt.xlabel(r"$\Delta \chi^2$ per degree of freedom", fontsize=16)
plt.legend(loc='best')
plt.title(r'Probability Density Function for $\chi^2$', fontsize=16)

#%%
df = 
x = np.linspace(chi2.ppf(0.0, df),
                  chi2.ppf(0.999, df), 1000)
                  
y = chi2.pdf(x, df)



#%%
dat = zip(x, chi2.pdf(x, df))


#%%

deltaAIC = [0.0, 2.0, 4.71, 6.09, 6.57, 6.72, 7.84, 8.09, 9.34, 12.59, 13.43, 14.49, 18.96, 20.76, 22.97, 23.87, 24.81, 25.11, 34.30]
#%%
#for i in deltaAIC
kz = [exp(-i/2.) for i in deltaAIC]

#%%
dAICsum = np.cumsum(deltaAIC)
mAICsum = np.max(dAICsum)
#%%

W = []
for i,j in enumerate(deltaAIC):
    #w = exp(-j/2)/np.cumsum(exp(-deltaAIC/2))
    w = exp(-j/2)/exp(-np.cumsum(deltaAIC)/2)
    W.append(w)



