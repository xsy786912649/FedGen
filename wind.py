#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:21:13 2020

@author: zqy5086@AD.PSU.EDU
"""
from math import pi
import numpy as np
import matplotlib.pyplot as plt


#phi=np.zeros((10,10))
def S(theta,U,cp):
    s=11.5*(U/cp)**(-2.5)
    return 2/pi*np.cos(theta)**(2*s)
def wind(x,y,E):
    np.random.seed(E)
    phi=np.random.random((10,10))*2*pi
    h=0
    U=15
    cp=7.5
    i=0
    j=0
    for w in np.linspace(0,1,10)*2*pi:
        dw=1/9
        j=0
        for th in np.linspace(-1,1,10)*pi/2:
            dth=1/9
            k=w/U
            h=h+np.sqrt(2*S(th,U,cp)*dw*dth)*np.cos(k*x*np.cos(th)+k*y*np.sin(th)+phi[i,j])
            j=j+1
        i=i+1
    return h*0.001

if __name__=='__main__':
    size=100
    X=np.linspace(0,1,size)
    Y=np.linspace(0,1,size)
    YY, XX = np.meshgrid(X, Y) 
    H=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            H[i,j]=wind(X[i]*100,Y[j]*100,1)
    
    plt.contourf(XX*100,YY*100,H*100)
    plt.colorbar()
            
     