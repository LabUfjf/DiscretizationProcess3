# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:37:41 2018

@author: Igor
"""

def L2(P,Q):
    import numpy as np
    simi = (np.sqrt(np.sum((P-Q)**2)))/np.size(P)
    return simi


def L1(P,Q):
    import numpy as np
    simi = (np.sum(np.abs(P-Q)))/np.size(P)
    return simi

def KL(P,Q):
    import numpy as np
    
    simi=(Q*np.log10((Q/P)));
    indN = np.where(np.isnan(simi) == False)
    indF = np.where(np.isinf(simi) == False)
    index = np.intersect1d(indN,indF)
    simi= np.sum(simi[index])/np.size(P)
    
    return simi