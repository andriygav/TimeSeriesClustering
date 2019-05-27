import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from scipy.special import softmax

import pandas as pd

from sklearn.preprocessing import scale

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA

import math


def RandomFunction(x):
    n = 2
    N = np.arange(1, n + 1, 1)
    A = np.random.randn(n)
    B = np.random.randn(n)
    A0 = np.random.randn(1)
    
    y = 0.5*np.ones_like(x)*A0
    
    for n, a, b in zip(N, A, B):
        y += a*np.sin(n*x) + b*np.cos(n*x)
    
    return y

def GenerateImpulses(n = 20, T = 2, k = 2, function = np.sin):
    
    t = int(T)//2
    
    x = np.linspace(start = 0, stop = T*np.pi, num = n)
    
    List_y = []
    
    for i in range(k):
        y_temp = 5*np.random.randn()*function(x + np.random.rand()*2*np.pi)
        List_y.append(y_temp)
    
    y = np.array(List_y[0])
    
    y2 = List_y[np.random.randint(0, k)]
    
    for i in range(0, t):
        if np.random.rand() < 0.1:
            y2 = List_y[np.random.randint(0, k)]
        
        ind = np.where(x <= 2*(i + 1)*np.pi)
        ind = np.where(x[ind] > 2*i*np.pi)
        y[ind] = y2[ind]
        
    return y
    

def GeneratorOfTimeSeries(n = 100, m = 16384, k = 20):
    T1 = []
    T2 = []
    T3 = []
    for _ in range(m):
        numPi = 80 + np.random.randint(0, 20)
        numPi = n//k
        function = np.sin
        if np.random.rand() < -4*0.5:
            function = RandomFunction
            
        series = GenerateImpulses(n = n, T = numPi, k = np.random.randint(K, K+1), function=function)
        T1.append(series + 0.5*np.random.randn(n))
    T1 = np.asarray(T1)
    
    return np.reshape(T1, [T1.shape[0], T1.shape[1], 1])

  

def return_h(input, i, l = 10):
    return np.sum(input[:, i:i+l, :], axis = -1)

def return_phase_track(input, l = 10):
    """
    input has a shape [batch_size, time_len, 1]
    """

    phase_track = np.zeros([input.shape[0], input.shape[1] - l, l])
    
    for i in range(0, input.shape[1] - l):
        phase_track[:, i, :] = return_h(input, i, l)
    
    return phase_track[0]
    

def local_basis(phase_track, m = 2, T = 20):

    result_pca_1 = phase_track

    List_of_basis_vector = []
    List_of_basis_vector_s = []
    List_of_basis_vector_c = []

    model_pca = PCA(n_components=2)

    for n in tqdm(range(T, result_pca_1.shape[0] - T, 1)):
        if n-T >- 0:
            arr = result_pca_1[n-T:n+T]
        else:
            arr = result_pca_1[:n]
        
        model_pca_answ = model_pca.fit_transform(arr)
        
        List_of_basis_vector_s.append(model_pca.singular_values_)
        
        List_of_basis_vector_c.append(model_pca_answ[-1])
        List_of_basis_vector.append(model_pca.components_)

    List_of_basis_vector = np.array(List_of_basis_vector)
    List_of_basis_vector_s = np.array(List_of_basis_vector_s)
    List_of_basis_vector_c = np.array(List_of_basis_vector_c)

    return List_of_basis_vector, List_of_basis_vector_s, List_of_basis_vector_c
 

def get_pairwise_matrix(List_of_basis_vector, List_of_basis_vector_s, List_of_basis_vector_c):

    Volum = np.zeros([2, List_of_basis_vector.shape[0], List_of_basis_vector.shape[0]])

    cos_beta = np.abs(List_of_basis_vector[:, 0, :]@List_of_basis_vector[:, 1, :].T)
    cos_alpha = np.array(np.diagonal(cos_beta))
    cos_gamma = np.abs(List_of_basis_vector[:, 1, :]@List_of_basis_vector[:, 1, :].T)

    cos_beta[np.where(cos_beta > 1-10**(-10))] = 1-10**(-10)
    cos_alpha[np.where(cos_alpha > 1-10**(-10))] = 1-10**(-10)
    cos_gamma[np.where(cos_gamma > 1-10**(-10))] = 1-10**(-10)

    cos_beta[np.where(cos_beta < 10**(-10))] = 0
    cos_alpha[np.where(cos_alpha < 10**(-10))] = 0
    cos_gamma[np.where(cos_gamma < 10**(-10))] = 0


    temp_a = np.sqrt(1-cos_beta**2)
    cos_A = np.abs((cos_alpha.reshape([-1,1]) - cos_gamma*cos_beta)/(np.sqrt(1-cos_gamma**2)*np.sqrt(1-cos_beta**2)))
    h = temp_a*np.sqrt(1-cos_A**2)

    Volum[0] = h* np.sqrt(1-cos_gamma**2)
        
    cos_beta = np.abs(List_of_basis_vector[:, 0, :]@List_of_basis_vector[:, 0, :].T)
    cos_gamma = np.abs(List_of_basis_vector[:, 1, :]@List_of_basis_vector[:, 0, :].T)

    cos_alpha = np.array(np.diagonal(cos_gamma))

    cos_beta[np.where(cos_beta > 1-10**(-10))] = 1-10**(-10)
    cos_alpha[np.where(cos_alpha > 1-10**(-10))] = 1-10**(-10)
    cos_gamma[np.where(cos_gamma > 1-10**(-10))] = 1-10**(-10)


    cos_beta[np.where(cos_beta < 10**(-10))] = 0
    cos_alpha[np.where(cos_alpha < 10**(-10))] = 0
    cos_gamma[np.where(cos_gamma < 10**(-10))] = 0


    temp_a = np.sqrt(1-cos_beta**2)
    cos_A = (cos_alpha.reshape([-1,1]) - cos_gamma*cos_beta)/(np.sqrt(1-cos_gamma**2)*np.sqrt(1-cos_beta**2))
    h = temp_a*np.sqrt(1-cos_A**2)

    Volum[1] = h* np.sqrt(1-cos_gamma**2)
          
    vol = np.max(Volum, axis = 0)

    for i in tqdm(range(vol.shape[0])):
        for j in range(vol.shape[0]):
            vol[i,j] = max(vol[i,j], vol[j,i])  

    dist = np.sqrt((List_of_basis_vector_s[:, :1] - List_of_basis_vector_s[:, :1].T)**2 + (List_of_basis_vector_s[:, 1:2] - List_of_basis_vector_s[:, 1:2].T)**2)
    dist = dist/np.max(dist)

    full_dist = np.sqrt(vol**2+dist**2)

    return full_dist



