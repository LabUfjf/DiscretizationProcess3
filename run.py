# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:54:43 2018

@author: Igor e Rafael

"""

import numpy as np
import someFunctions as sf
from scipy.interpolate import interp1d
from scipy.stats import norm, lognorm
import methodDisc as md
import divergency as dv
import matplotlib.pyplot as plt

#import matplotlib.patches as mpatches
#import scipy.io
#import funcoes as fc
#import KDEfunctions as KDE
#import os.path
#from mpl_toolkits.mplot3d import Axes3D
#from distAnalyze import diffArea



if __name__ == '__main__':
    
    ## FIXED Control Variables
    npts = np.concatenate([list(range(10,250,10)),list(range(250,550,50)),list(range(600,1100,100)),list(range(1500,5500,500))])
    kinds = ['Linspace', 'CDFm', 'PDFm', 'iPDF1', 'iPDF2']
    ROI = 1
    seed = None
    ngrid = int(1e6)
    weight = False
    outlier = 0
    curve = 3           # 1:5
    plotKind = 'Linspace CDFm PDFm iPDF1 iPDF2'.split()
    plotNest = [10,100,500,1000,1500,3500]
    #-------------------------------------------------------
    np.random.seed(seed)    #seed = np.random.get_state()  np.random.set_state(seed)
    
    sig = [0.01,0.5,1,1.25,2]
    
    mu = 0
    sigma = sig[curve-1]
    
    divergence = 'L1'       # L2 e KL
    interpolator = 'linear'    # nearest
    distribuition = 'lognormal'   # lognormal
    data = 0
    #####################################
    # Definition of the distribuition parameters
    
    
    #####################################
    
    if data:
        analitica = False
    else:
        analitica = True
 
    probROIord = {}
    areaROIord = {}
    
    for kind in kinds:
        probROIord[kind] = []
        areaROIord[kind] = []
    div = {}
    area = []
    n = []
    truth = truth1 = sf.pdf
    
    if data:
        if distribuition == 'normal':
                d = np.random.normal(mu,sigma,data)
        elif distribuition == 'lognormal':
                d = np.random.lognormal(mu, sigma, data)
        inf,sup = min(d),max(d)
        
    else:
        d=0
        if distribuition == 'normal':
              inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
            
        elif distribuition == 'lognormal':
              inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
              inf = lognorm.pdf(sup, sigma, loc = 0, scale = np.exp(mu))
              inf = lognorm.ppf(inf, sigma, loc = 0, scale = np.exp(mu))
              
    xgrid = np.linspace(inf,sup,ngrid)
    xgridROI = xgrid.reshape([ROI,ngrid//ROI])
        
    dx = np.diff(xgrid)[0]
    
    for nest in npts:
        print('#\nCalculating - Nº of estimation points = %d' %(nest))
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


            #### TESTE ####
            
            YN=fest(xgrid)   # Y calculado pelos metodos
            
            YT=truth(xgrid,mu,sigma,distribuition) # Y calculado pelo TRUTH
            
            diver = getattr(dv,divergence)(YT,YN)
            
            areaROIord[kind]=np.append(areaROIord[kind],diver)
            
            
            
#            ###################################################################
#            yestGrid = []
#            ytruthGrid = []
#            ytruthGrid2 = []
#            divi = []
#                
#            for i in range(ROI):
#                yestGrid.append([fest(xgridROI[i])])
#                ytruthGrid.append([truth(xgridROI[i],mu,sigma,distribuition)])
#                ytruthGrid2.append([truth1(xgridROI[i],mu,sigma,distribuition)])
#                divi.append(len(np.intersect1d(np.where(xest >= min(xgridROI[i]))[0], np.where(xest < max(xgridROI[i]))[0])))
#            
#            diff2 = np.concatenate(abs((np.array(yestGrid) - np.array(ytruthGrid))*dx))
#            #diff2[np.isnan(diff2)] = 0
#            areaROI = np.sum(diff2,1)
#            
#            divi = np.array(divi)   
#            divi[divi == 0] = 1
#            
#            try:
#                probROI = np.mean(np.sum(ytruthGrid2,1),1)
#            except:
#                probROI = np.mean(ytruthGrid2,1)
#            
#            
#            probROIord[kind] = np.append(probROIord[kind],np.sort(probROI))
#            index = np.argsort(probROI)
#            
#            areaROIord[kind]=np.append(areaROIord[kind],areaROI[index])
#            
#            area = np.append(area,np.sum(areaROIord[kind]))
#            n = np.append(n,len(probROIord[kind]))
#            div[kind] = divi[index]
            
            #retorno = area,[probROIord,areaROIord]
            
            
            ####### PLOTS ######
            if any(num == nest for num in plotNest) and any(k in kind for k in plotKind):
                title = kind + ' - ' + str(nest)
                plt.figure(title,figsize=(8,6),dpi=100)
                plt.plot(xest,YY,'o',lw=4,label='Estimated',alpha = 0.5)
                plt.plot(xgrid,YN,lw=4,label='Interpolated',alpha = 1)
                plt.plot(xgrid,YT,label='Truth')
                plt.xlabel('Samples')
                plt.ylabel('Probability')
                plt.legend()
                plt.title('Analytical = %s - Distribuition = %s - Discretizator = %s \n Interpolator = %s - Nº of Nest = %d - Nº of Events = %d' 
                          %(analitica,distribuition.upper(),kind,interpolator.upper(),nest,data))
                plt.savefig('./Figures/' + kind + '/MIDLEPLOT_' + kind + '_' + str(analitica).upper() + '_' + distribuition.upper() + '_' + interpolator.upper() + '_' + str(nest) + '_' + str(data) + '.png', bbox_inches='tight')
                
    fig, ax1 = plt.subplots(figsize=(8,6),dpi=100)
    for kind in kinds:
        plt.plot(npts,areaROIord[kind],'-o',label = kind, ms = 3)
    plt.yscale('log')
    plt.xlabel('Number of estimation points')
    plt.ylabel('%s Divergence' %(divergence))
    plt.legend()
    plt.title('Analytical = %s - Distribuition = %s \n Interpolator = %s - Nº of Events = %d' %(str(analitica).upper(),distribuition.upper(),interpolator.upper(),data))
    fig.savefig('./Figures/ERRORPLOT_' + divergence + '_' + str(analitica).upper() + '_' + distribuition.upper() + '_' + interpolator.upper() + '_' + str(data) + '.png', bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    