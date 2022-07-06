# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:54:22 2022

@author: Georgios Zaverdinos
"""






from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import special
from decimal import Decimal as D   



exp_mean = [65.46748667, 12.76041667, 10.36577778, 12.77738889, 12.69999091, 17.38184, 16.65666154 ,20.87132727 ,70.17311538]


def network(neurons,N,J_e,J_i,C_e,C_i,v_ext,gamma,a,g,c_theta,c_sigma,t_init,t_end):


    spikes = 0
    dt     = float(t_end - t_init) / N
    sim_times = t_end - t_init -1
    y_init = 0
    tau = 0.020
    t_e = 0.0035
    t_i = 0.0035
    t_ref = 0.0020
    t_vth = 8
    sptime = []
    I = J_e*tau*v_ext
    
    matrixx = np.zeros((N + 1)*neurons).reshape(neurons,N+1)
    
        
        
    def mu(y,row,column):
       
        return c_theta * (matrixx[row][column] - y)

    def sigma(t):
        
        return float(c_sigma)/np.sqrt(tau)

    def dW(delta_t):
        
        return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

    ts = np.arange(t_init, t_end + dt, dt)
    ys = np.zeros((N + 1)*neurons).reshape(neurons,N+1)
    Pre = np.zeros((N+1)*neurons).reshape(neurons,N+1)
    pre_pre = np.zeros((N+1)*neurons).reshape(neurons,N+1)
    
    Vth = np.zeros((N+1)*neurons).reshape(neurons,N+1)
    
    
    random_list = []
    neurons_index = np.arange(0,neurons,1)
    for w in range(0,int(neurons-gamma*neurons)):
        random_choice = np.random.choice(neurons_index,size = int(C_e*neurons*(1/gamma)))
        random_list.append(random_choice)
        
    for w in range(int(neurons-gamma*neurons),neurons):
        random_choice = np.random.choice(neurons_index,size = int(C_i*neurons*(1/gamma)))
        random_list.append(random_choice)
    
    
    tik = np.zeros(neurons*1).reshape(neurons,1)
    
    spike_time = 1 
    index = 0 
    tikkie = 0 
    count = 0

    pre = 0
    bonus = np.zeros(neurons).reshape(neurons,1)
    for i in range(1, ts.size):
       
        
        for w in range(0,int(neurons-gamma*neurons)):
            pre_pre[w][i] += J_e*bonus[w][0]
            
            Vth[w][0] = 20 
            vth = Vth[w][i-1]
            Vth[w][i] = vth + ((20 - vth)/t_vth) * dt 
          
        for w in range(0,int(neurons-gamma*neurons)):
            pre =Pre[w][i-1]
            Pre[w][i] = pre + ((- pre/t_e) + pre_pre[w][i-1]) * dt 
            matrixx[w][i] = I + 1000*Pre[w][i]
            
            
        
        for w in range(int(neurons-gamma*neurons),neurons):
            pre_pre[w][i] += J_i*bonus[w][0]
            
            Vth[w][0] = 20 
            vth = Vth[w][i-1]
            Vth[w][i] = vth + ((20 - vth)/t_vth) * dt 
            
            
            
        for w in range(int(neurons-gamma*neurons),neurons):
            pre =Pre[w][i-1]
            Pre[w][i] = pre + ((- pre/t_i) +  pre_pre[w][i-1]) * dt 
            matrixx[w][i] = I + 1000*Pre[w][i]
            
            
            
            
        bonus = np.zeros(neurons).reshape(neurons,1)
           
        
        for k in range(0,neurons):
            
            
            if tik[k][0] > i* dt :
                ys[k][i] = 10
  
            else:
        
                t = t_init + (i - 1) * dt
      
                y = ys[k][i-1]
                ys[k][i] = y + mu(y,k,i) * dt + sigma(t) * dW(dt)
             
                if ys[k][i]>Vth[k][i]:
                
                    ys[k][i] = 80
                    
                    Vth[k][i] = Vth[k][i] + a 
                    
                    tik[k][0] = (i)*dt + t_ref
                    
                    if k < (neurons-gamma*neurons):
                        spikes += 1 
                
                    
                    
                    if k > (neurons-gamma*neurons): 
                        for r in random_list[k]:
                            if r < (neurons-gamma*neurons):
                                bonus[r][0] =(bonus[r][0] - g)
                                
                            elif r > (neurons-gamma*neurons) or r == (neurons-gamma*neurons):
                                bonus[r][0] = (bonus[r][0] - g)
                        
                    else:
                        for r in random_list[k]:  
                            if r < (neurons-gamma*neurons):
                                bonus[r][0] =(bonus[r][0] + 1)
                                
                            elif r > (neurons-gamma*neurons) or r == (neurons-gamma*neurons):
                                bonus[r][0] = (bonus[r][0] + 1)
                                
                                
    for v in range(0,1000):
        first_order_time = []
        for x in range(0,ts.size):
              if ys[v][x] == 80 :
                  first_order_time.append(x*dt)
              else:
                  continue
        
        sptime.append(first_order_time)
        
        
        
    ################# Getting frequency #####################
    freq_list = np.zeros(ts.size)
    for x in range(0,ts.size-1):
        frequency = 0 
        for v in range(0,int(1000)):       
            if ys[v][x] == 80 :
                frequency += 1
            else:
                continue
        
        freq_list[x] = frequency
        
        
        
    return ys, ts, Pre,matrixx, float(spikes)/float(neurons), sptime, Vth, freq_list



def burst(vector):
    
    burst_beginning = []
    burst_end = []
    burst_duration = []
    time = len(vector)
    freq = vector
    dt = 0.0005
    
    for i in range(3,time-3):
        
        
        if freq[i] > 10 and freq[i-1] < 10 and freq[i-2] < 10 and freq[i-3] < 10:
            burst_beginning.append(dt*i)
            
        elif freq[i] <10 and freq[i-2] > 10 and freq[i+1] < 10 and freq[i+2] < 10 and freq[i+3] < 10:
            burst_end.append(dt*i)
             
                
    IBI = []
    index = []
    diff = 0
    for t in range(1,len(burst_beginning)):
        diff = burst_beginning[t] - burst_end[t-1]
        burst_duration.append(burst_end[t-1] - burst_beginning[t-1])
        IBI.append(diff)
        
    
    for i in range(0,len(burst_beginning)):
        time_range = (burst_beginning)[i] + 0.02
        max_list = []
        for k in range(int(2000*(burst_beginning[i])), int(2000*time_range)):
            max_list.append((qe[7])[int(k)])

        index.append(max(max_list))
        
        
        


    return burst_beginning, burst_end, burst_duration, index , IBI

#### CHANGE THE Ni #########
Ni = 0.2*1000
Ne = 1000-Ni
C_e = 0.078*Ne + 17.6
C_e = C_e / 1000


##### Look up ######
qe = network(1000,2000*50,2,2,C_e,0.02,1000,0.2,12.5,4,50,3,0,50)


plt.plot(qe[1],(qe[0])[100])
plt.show()

print("10%")
print("1000,2000*50,2,2,C_e,0.02,1000,0.1,12.5,4,50,3,0,50")



neuralData = qe[5]

plt.eventplot(neuralData,orientation =  'horizontal' , color = 'black')     

 

# Provide the title for the spike raster plot

plt.title('Spike raster plot')

 

# Give x axis label for the spike raster plot

plt.xlabel('Time')

 

# Give y axis label for the spike raster plot

plt.ylabel('Neuron')

 

# Display the spike raster plot
plt.savefig("raster_plot_10%(variance)_0.02_1000.png")
plt.clf()


print(burst(qe[7])[0])
print(burst(qe[7])[1])
print(burst(qe[7])[2])
print(burst(qe[7])[3])
print(burst(qe[7])[4])


burst_list = burst(qe[7])

import statistics
std = statistics.stdev(burst_list[4])
std = std
sim_mean = statistics.mean(burst_list[4])
sim_mean = sim_mean


print(std,sim_mean)
    


##### ERROR FUNCTION ######
inhibitory_perc = [0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.95]
exp_variance = 0 
sim_variance = 0

E = 1/2*((exp_mean[3] - sim_mean)**2 + (exp_variance - sim_variance)**2)

print(E)
    
