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

from scipy.interpolate import interp1d

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

    for n in range(T, result_pca_1.shape[0] - T, 1):
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

    for i in range(vol.shape[0]):
        for j in range(vol.shape[0]):
            vol[i,j] = max(vol[i,j], vol[j,i])  

    dist = np.sqrt((List_of_basis_vector_s[:, :1] - List_of_basis_vector_s[:, :1].T)**2 + (List_of_basis_vector_s[:, 1:2] - List_of_basis_vector_s[:, 1:2].T)**2)
    dist = dist/np.max(dist)

    full_dist = np.sqrt(vol**2+dist**2)

    return full_dist


def find_points(points, line_point):
    """
    points have a shape [N x 2]
    line_point has a shape [2 x 1]
    """
    List_of_points_plus = []
    List_of_points_minus = []
    
    List_of_t_plus = []
    List_of_t_minus = []
    
    for i in range(len(points) - 1):
        if (line_point[1]*points[i][0] - line_point[0]*points[i][1] < 0) and(line_point[1]*points[i+1][0] - line_point[0]*points[i+1][1] > 0):
            List_of_points_plus.append(points[i])
            List_of_t_plus.append(i)
        if (line_point[1]*points[i][0] - line_point[0]*points[i][1] > 0) and(line_point[1]*points[i+1][0] - line_point[0]*points[i+1][1] < 0):
            List_of_points_minus.append(points[i])
            List_of_t_minus.append(i)
    
    return np.array(List_of_points_plus), np.array(List_of_points_minus), np.array(List_of_t_plus), np.array(List_of_t_minus)

def find_distance(points, line_point):
    """
    points have a shape [N x 2]
    line_point has a shape [2 x 1]
    """
    
    sum_distance = 0
    
    normal = np.array([line_point[1], -line_point[0]])
    normal = normal/np.sqrt((normal*normal).sum())
    
    for p in points:
        sum_distance += ((normal*p).sum())
    
    
    return sum_distance

def find_segment(X, T):
    phase_track = return_phase_track(X, T)
    model = PCA(n_components=2)

    ress = model.fit_transform(phase_track)
    
    ress[:, 0] = ress[:, 0]/np.sqrt(((ress[:, 0]**2).mean()))

    ress[:, 1] = ress[:, 1]/np.sqrt(((ress[:, 1]**2).mean()))

    Phi = np.linspace(-np.pi, np.pi, 200)

    All_List = np.array(list(map(lambda phi: find_points(ress, np.array([np.sin(phi), np.cos(phi)])), Phi)))

    List_of_std = []
    for l, phi in zip(All_List, Phi):
        List_of_std.append(find_distance(np.vstack([l[0], l[1]]), np.array([np.sin(phi), np.cos(phi)])))

    List_of_std = np.array(List_of_std)
    
    phi = Phi[np.argmin(List_of_std)]
    
    
    line_point = np.array([np.sin(phi), np.cos(phi)])

    List_of_points_plus, List_of_points_minus, List_of_t_plus, List_of_t_minus = find_points(ress, line_point)
    
    return List_of_points_plus, List_of_points_minus, List_of_t_plus, List_of_t_minus, line_point, ress
    
    
def segmentation(X_all, prediction_vector, T):
    List_of_point = []
    List_of_All = []
    for t in np.unique(prediction_vector):
        ind = np.where(prediction_vector == t)[0]

        X = X_all[:, ind, :]
        List_of_t = np.arange(0, X.shape[1], 1)

        List_of_points_plus, List_of_points_minus, List_of_t_plus, List_of_t_minus, line_point, ress = find_segment(X, T)

        List_of_All.append([X, List_of_t, List_of_points_plus, List_of_points_minus, List_of_t_plus, List_of_t_minus, line_point, ress])
        List_of_point.append((np.where(prediction_vector == t)[0])[List_of_t_minus])

    return List_of_All, List_of_point

def normalizer(x, t, n = None):
    if n == None:
        t_new = np.arange(np.min(t), np.max(t), 0.01)
    else:
        t_new = np.linspace(np.min(t), np.max(t), n, endpoint=True)
    
    f = interp1d(t, x, kind='cubic')
    
    return f(t_new)

def sort_prediction(prediction_vector):
    prediction_vector += 1000
    iterator = 0

    need = np.where(prediction_vector >= 1000)[0]
    while len(need) > 0:
        prediction_vector[np.where(prediction_vector == prediction_vector[need[0]])] = iterator
        iterator += 1
        need = np.where(prediction_vector >= 1000)[0]
        
    return prediction_vector
