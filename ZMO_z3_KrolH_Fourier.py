# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 18:03:54 2023

@author: huber
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cmath
import sys

#(DFT)
def DFT(x):
    N = np.size(x)
    X = np.zeros((N,),dtype=np.complex128)
    for m in range(0,N):    
        for n in range(0,N): 
            X[m] += x[n]*np.exp(-np.pi*2j*m*n/N)
    return X


#(IDFT)
def IDFT(x):
    N = np.size(x)
    X = np.zeros((N,),dtype=np.complex128)
    for m in range(0,N):
        for n in range(0,N): 
            X[m] += x[n]*np.exp(np.pi*2j*m*n/N)
            
    return X/N

#(IFFT)
def IFFT(x):
    N = np.size(x)
    X = np.zeros((N,),dtype=np.complex128)
    for m in range(0,N):
        for n in range(0,N): 
            X[m] += x[n]*np.exp(np.pi*2j*m*n/N)
            
    return X/N


#(FFT)
def FFT(x):
    N = len(x)
  
    if N==1:
        return x
    
    M = N // 2
        
    x_even = np.array([x[2*i] for i in range(M)], dtype=complex)
    x_odd = np.array([x[2*i+1] for i in range(M)], dtype=complex)
    
    f_even = FFT(x_even)
    f_odd = FFT(x_odd)
    
    coe = np.empty(N, dtype=complex)
    for k in range(M):
        exp = cmath.exp(complex(0, -2*cmath.pi*k/N)) * f_odd[k]
        coe[k] = f_even[k] + exp
        coe[k+M] = f_even[k] - exp
        
    return coe

def compress(avg, xd):
    return np.array([xd[i] if abs(xd[i]) > avg else 0 for i in range(len(xd))],dtype = complex)
    
    
#filename = 'C:\\Users\huber\\Desktop\\Data\\dane_03_a.in'
filename = sys.argv[1]
    
with open(filename, 'r') as file:
   lines = file.readlines()

if ' ' in lines[2]:
    
    with open(filename, 'r') as file:
       dim = int(next(file))
       dims = np.array([int(x) for x in next(file). split()], dtype = int)
       data = np.array([[float(x) for x in line.split()] for line in file], dtype = float)
       plt.figure(1)
       plt.title("Dane przed kompresją")
       plt.imshow(np.real(data))
       coe = np.fft.fft2(data)
       plt.figure(2)
       plt.title("widmo 2D przed kompresją")
       plt.imshow(np.real(coe))
       coeifft =np.real(np.fft.ifft2(coe))
       plt.figure(3)
       plt.title("IFFT przed komresją")
       plt.imshow(coeifft)
       
       
       
       data2 = np.empty(shape=dims, dtype=complex)
       for i in range(len(data)):
       
           avg = np.average(abs(data))
           
           data2[i] = compress(avg, data[i])
        
       plt.figure(4) 
       plt.title("Dane po kompresji")
       plt.imshow(np.real(data2))
       coe2 =np.real(np.fft.fft2(data2))
       
       plt.figure(5)
       plt.title("widmo 2D Po kompresji")
       plt.imshow(coe2)
       
       coe3 =np.real(np.fft.ifft2(coe2))
       plt.figure(6)
       plt.title("IFFT po komresji")
       plt.imshow(coe3)
       
       
else:
    df = pd.DataFrame(columns=['A'])
    for i in lines:
        df = df.append({'A' : i }, ignore_index = True)
        
        
    df = df.astype(float)
    df2 = df.iloc[2:]
    df2.reset_index(drop=True, inplace=True)
    
    
    
    
    '''
    x = df2.to_numpy()
    # compute DFT
    
    arrdft=DFT(x)
    arridft = IDFT(arrdft)
    
    h = [ele.real for ele in arrdft]
    # extract imaginary part
    g = [ele.imag for ele in arrdft]
    plt.figure(1)
    plt.plot(h)
    plt.xlabel('N harmoniczna') 
    plt.ylabel('Modul harmonicznej')
    plt.title("Real DFT")
    
    plt.figure(2)
    plt.plot(g)
    plt.xlabel('N harmoniczna') 
    plt.ylabel('Modul Harmonicznej')
    plt.title("Imaginary DFT")
    '''
    '''
    a_list = list(arridft)
    
    z = [ele.real for ele in a_list]
    plt.figure(3)
    plt.plot(z)
    plt.xlabel('N') 
    plt.ylabel('Sygnal')
    plt.title("Real IDFT")
    '''
    
    
    x = df2.to_numpy()
    
    
    
    xd = FFT(x)
    avg = np.average(abs(xd))
    comp = compress(avg, xd)
    plt.figure(1)
    
    
    plt.plot(abs(xd))
    plt.plot(abs(comp))
    plt.xlabel('N Harmoniczna') 
    plt.ylabel('Modul harmonicznej')
    plt.title("widmo 1D przed i po kompresji (FFT)")

    xd2 = IFFT(comp)
    plt.figure(2)
    plt.plot(x)
    plt.plot(xd2)
    plt.xlabel('N') 
    plt.ylabel('Signal')
    plt.title("Dane wejsciowe i dane (IFFT) po kompresji")



