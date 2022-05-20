from math import pi
import numpy as np

def S(theta,U,cp):
    s=11.5*(U/cp)**(-2.5)
    return 2/pi*np.cos(theta)**(2*s)

def wind(x,y):
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
    return h*0.1