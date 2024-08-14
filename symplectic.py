import numpy as np

def BS():
    """ Define the symplectic transformation for a balance beam splitter"""
    return 1/np.sqrt(2)*np.array([[1,-1,0, 0],
                                    [1, 1,0, 0],
                                    [0, 0,1,-1],
                                    [0, 0,1, 1]])

def S( r):
    """ Define the symplectic transformation for squeezing"""
    return np.array([[np.exp(-2*r), 0          ],
                        [0,            np.exp(2*r)]])

def P(theta):
    """ Define the symplectic transformation for a phase shift"""
    return np.array([[np.cos(theta),-np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])