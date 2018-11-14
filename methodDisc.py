"""
Created on Fri Oct 26 14:57:59 2018

@author: Rafael Mascarenha
@author2: Igor Abritta

In this file we have a few functions:
    - CDFm
    - PDFm
    - iPDF1
    - iPDF2

"""


def CDFm(data,nPoint,dist = 'normal', mu = 0, sigma = 1,analitica = False):
    import numpy as np
    from scipy.interpolate import interp1d
    from statsmodels.distributions import ECDF
    from scipy.stats import norm, lognorm
    
    eps = 5e-5
    yest = np.linspace(0+eps,1-eps,nPoint)
    
    if not analitica:    
        ecdf = ECDF(data)
        inf,sup = min(data),max(data)
        xest = np.linspace(inf,sup,int(100e3))
        yest = ecdf(xest)
        interp = interp1d(yest,xest,fill_value = 'extrapolate', kind = 'nearest')
        y = np.linspace(eps,1-eps,nPoint)
        x = interp(y)
    else:
        if dist is 'normal':
            x = norm.ppf(yest, loc = mu, scale = sigma)
        elif dist is 'lognormal':
            x = lognorm.ppf(yest, sigma, loc = 0, scale = np.exp(mu))
    
    return x

def PDFm(data,nPoint,dist = 'normal', mu = 0, sigma = 1,analitica = False):
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.stats import norm, lognorm
    eps = 5e-5
    
    if not analitica:
        yest,xest = np.histogram(data,bins = 'fd',density = True)
        xest = np.mean(np.array([xest[:-1],xest[1:]]),0)
        M = np.where(yest == max(yest))[0][0]
        m = np.where(yest == min(yest))[0][0]
        
        if M:
            interpL = interp1d(yest[:M+1],xest[:M+1], fill_value = 'extrapolate')
            interpH = interp1d(yest[M:],xest[M:])
        
            y1 = np.linspace(yest[m]+eps,yest[M],nPoint//2+1)
            x1 = interpL(y1)
            
            y2 = np.flip(y1,0)
            x2 = interpH(y2)
               
                
            x = np.concatenate([x1[:-1],x2])
            y = np.concatenate([y1[:-1],y2])
        else:
            interp = interp1d(yest,xest,fill_value='extrapolate')
            if not nPoint%2:
                nPoint = nPoint+1
            y = np.linspace(yest[M],yest[m],nPoint)
            x = interp(y)
    else:
        if dist is 'normal':
            inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
              
      
            X1 = np.linspace(inf,mu,int(1e6))
            Y1 = norm.pdf(X1, loc = mu, scale = sigma)
            interp = interp1d(Y1,X1)
            y1 = np.linspace(Y1[0],Y1[-1],nPoint//2+1)
            x1 = interp(y1)
              
            X2 = np.linspace(mu,sup,int(1e6))
            Y2 = norm.pdf(X2, loc = mu, scale = sigma)
            interp = interp1d(Y2,X2)
            y2 = np.flip(y1,0)
            x2 = interp(y2)
        elif dist is 'lognorm':
            inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
            inf = lognorm.pdf(sup, sigma, loc = 0, scale = np.exp(mu))
            inf = lognorm.ppf(inf, sigma, loc = 0, scale = np.exp(mu))
            mode = np.exp(mu - sigma**2)
              
            X1 = np.linspace(inf,mode,int(1e6))
            Y1 = lognorm.pdf(X1, sigma, loc = 0, scale = np.exp(mu))
            interp = interp1d(Y1,X1)
            y1 = np.linspace(Y1[0],Y1[-1],nPoint//2+1)
            x1 = interp(y1)
              
            X2 = np.linspace(mode,sup,int(1e6))
            Y2 = lognorm.pdf(X2, sigma, loc = 0, scale = np.exp(mu))
            interp = interp1d(Y2,X2)
            y2 = np.flip(y1,0)
            x2 = interp(y2)
        x = np.concatenate([x1[:-1],x2])
    
    return x

def iPDF1(data,nPoint,dist = 'normal', mu = 0, sigma = 1,analitica = False):
    import numpy as np
    from scipy.interpolate import interp1d
    from methodDisc import mediaMovel
    from scipy.stats import norm, lognorm
    from someFunctions import ash, dpdf
    eps = 5e-5
    n = 5
    if not analitica:       
    #x,y = ash(data,m=10,tip='linear',normed=True)
    #m = np.where(y == 0)
    #y[m]=np.min(y)
        y,x = np.histogram(data,bins = 'fd',density = True)
        x = np.mean(np.array([x[:-1],x[1:]]),0)
      
        y = abs(np.diff(mediaMovel(y,n)))
        x = x[:-1]+np.diff(x)[0]/2
        
        cdf = np.cumsum(y)    
        cdf = cdf/max(cdf)
        
        interp = interp1d(cdf,x, fill_value = 'extrapolate')
        Y = np.linspace(eps,1-eps,nPoint)
        X = interp(Y)
    
    else:
        ngrid = int(1e6)
        if dist is 'normal':
            inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
        elif dist is 'lognormal':
            inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
            inf = lognorm.pdf(sup, sigma, loc = 0, scale = np.exp(mu))
            inf = lognorm.ppf(inf, sigma, loc = 0, scale = np.exp(mu))  
       
        x = np.linspace(inf,sup,ngrid)
        y = dpdf(x,mu,sigma,dist)
        cdf = np.cumsum(y)
        cdf = cdf/max(cdf)
        interp = interp1d(cdf,x, fill_value = 'extrapolate')
        Y = np.linspace(eps,1-eps,nPoint)
        X = interp(Y)
    
    
    return X

def iPDF2(data,nPoint,dist = 'normal', mu = 0, sigma = 1,analitica = False):
    import numpy as np
    from scipy.interpolate import interp1d
    from someFunctions import ash, ddpdf
    from scipy.stats import norm, lognorm
    eps = 5e-5
    n = 5
          
#    x,y = ash(data,m=10,tip='linear',normed=True)
#    m = np.where(y == 0)
#    y[m]=np.min(y)
    if not analitica:
        y,x = np.histogram(data,bins = 'fd',density = True)
        x = np.mean(np.array([x[:-1],x[1:]]),0)
      
        y = abs(np.diff(mediaMovel(y,n),2))
        x = x[:-2]+np.diff(x)[0]
        y = y/(np.diff(x)[0]*sum(y))
        
        cdf = np.cumsum(y)    
        cdf = cdf/max(cdf)
        
        interp = interp1d(cdf,x, fill_value = 'extrapolate')
        Y = np.linspace(eps,1-eps,nPoint)
        X = interp(Y)
    else:
        ngrid = int(1e6)
        if dist is 'normal':
            inf, sup = norm.interval(0.9999, loc = mu, scale = sigma)
        elif dist is 'lognormal':
            inf, sup = lognorm.interval(0.9999, sigma, loc = 0, scale = np.exp(mu))
            inf = lognorm.pdf(sup, sigma, loc = 0, scale = np.exp(mu))
            inf = lognorm.ppf(inf, sigma, loc = 0, scale = np.exp(mu))
            
        x = np.linspace(inf,sup,ngrid)
        y = ddpdf(x,mu,sigma,dist)
        cdf = np.cumsum(y)
        cdf = cdf/max(cdf)
        interp = interp1d(cdf,x, fill_value = 'extrapolate')
        Y = np.linspace(eps,1-eps,nPoint)
        X = interp(Y)
    return X

def mediaMovel(x,n):
      from numpy import mean
      for i in range(len(x)):
            if i < n//2:
                  x[i] = mean(x[:n//2])
            else:
                  x[i] = mean(x[i-n//2:i+n//2])
      return x




