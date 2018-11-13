# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:54:43 2018

@author: Igor e Rafael

"""

import numpy as np
#import scipy.stats as sp

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io
import someFunctions as sf
import funcoes as fc
import KDEfunctions as KDE
import os.path
import methodDisc as md
from mpl_toolkits.mplot3d import Axes3D
from distAnalyze import diffArea
from scipy.interpolate import interp1d
from scipy.stats import norm, lognorm


if __name__ == '__main__':
    
    ## Control Variables
    nest = np.concatenate([list(range(10,250,10)),list(range(250,550,50)),list(range(600,1100,100)),list(range(1500,5500,500))])
    outlier = 0
    data = 0
    kinds = 'all'
    axis = 'probability'
    ROI = 20
    mu = 0
    sigma = 1
    weight = False
    interpolator = 'linear'
    distribuition = 'normal'
    seed=None
    ngrid = int(1e6)
    analitica = False
    #####################################
    
    probROIord = {}
    areaROIord = {}
    div = {}
    area = []
    n = []
    truth = truth1 = sf.pdf
    kinds = ['Linspace', 'CDFm', 'PDFm', 'iPDF1', 'iPDF2']
    
    if data:
        if distribuition == 'normal':
                d = np.random.normal(mu,sigma,data)
        elif distribuition == 'lognormal':
                d = np.random.lognormal(mu, sigma, data)
        inf,sup = min(d),max(d)
        
    else:
        if distribuition == 'normal':
              inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
            
        elif distribuition == 'lognormal':
              inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
              inf = lognorm.pdf(sup, sigma, loc = 0, scale = np.exp(mu))
              inf = lognorm.ppf(inf, sigma, loc = 0, scale = np.exp(mu))
              
    xgrid = np.linspace(inf,sup,ngrid)
    xgridROI = xgrid.reshape([ROI,ngrid//ROI])
        
    dx = np.diff(xgrid)[0]
    
    for kind in kinds:
        if kind == 'Linspace':
            if not data:  
                  xest = np.linspace(inf,sup,nest)
            else:
                  if distribuition == 'normal':
                        xest = np.linspace(inf,sup,nest)
                  elif distribuition == 'lognormal':
                        xest = np.linspace(inf,sup,nest)
            
        else:
            xest = getattr(md,kind)(data,nest,distribuition,mu,sigma,analitica)
    
    
    YY = sf.pdf(xest,mu, sigma,distribuition)
    fest = interp1d(xest,YY,kind = interpolator, bounds_error = False, fill_value = (YY[0],YY[-1]))
    
    yestGrid = []
    ytruthGrid = []
    ytruthGrid2 = []
    divi = []
        
    for i in range(ROI):
        yestGrid.append([fest(xgridROI[i])])
        ytruthGrid.append([truth(xgridROI[i],mu,sigma,distribuition)])
        ytruthGrid2.append([truth1(xgridROI[i],mu,sigma,distribuition)])
        divi.append(len(np.intersect1d(np.where(xest >= min(xgridROI[i]))[0], np.where(xest < max(xgridROI[i]))[0])))
    
    diff2 = np.concatenate(abs((np.array(yestGrid) - np.array(ytruthGrid))*dx))
    #diff2[np.isnan(diff2)] = 0
    areaROI = np.sum(diff2,1)
    
    divi = np.array(divi)   
    divi[divi == 0] = 1
    
    try:
        probROI = np.mean(np.sum(ytruthGrid2,1),1)
    except:
        probROI = np.mean(ytruthGrid2,1)
    
    
    probROIord[kind] = np.sort(probROI)
    index = np.argsort(probROI)
    
    areaROIord[kind] = areaROI[index]
    
    area = np.append(area,np.sum(areaROIord[kind]))
    n = np.append(n,len(probROIord[kind]))
    div[kind] = divi[index]
    
    retorno = area,[probROIord,areaROIord]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    