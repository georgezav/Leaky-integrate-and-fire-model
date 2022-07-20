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
   

##### Poisson input #######
# No need to define anything here # 
def poisson(N, V_in, freq):
     
     poisson_list = np.zeros(N+1)
     index = []
     
     for i in range(freq):
         
         index.append(int(random.randint(1,N)))
         
             
     for k in index:
         
         poisson_list[k] += V_in

     
     return poisson_list 
 
####### Network function #########

def network(neurons,N,J_e,J_i,C_e,C_i,I,v_freq,gamma,a,g,c_theta,c_sigma,t_init,t_end):


    spikes = 0
    dt     = float(t_end - t_init) / N
    sim_times = t_end - t_init -1
    y_init = 0 # Membrane voltage at point zero 
    tau = 0.020 # Membrane constant 
    t_e = 0.0035 # Excitatory neurons decay time 
    t_i = 0.0035 # Inhibitory neurons decay time
    t_ref = 0.0020 # Refractory period
    t_vth = 8 # Adaptation decay time 
    ###### Poisson variables ######
    V_in = 10 # Voltage value of external poisson input 
    ###### Steady input current ######
    sptime = []
    matrixx = np.zeros((N + 1)*neurons).reshape(neurons,N+1) # defining a matrix ''matrixx'' which sums all inputs each neuron receives
    
    poisson_spikes = poisson(N,V_in,v_freq*t_end) # calling poisson function 
        
    
    ######################### Functions for Euler approximation ####################################
    def mu(y,row,column):
       
        return c_theta * (matrixx[row][column] - y)

    def sigma(t):
        
        return float(c_sigma)/np.sqrt(tau)

    def dW(delta_t):
        
        return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))
    ################################################################################################
    
    
    ts = np.arange(t_init, t_end + dt, dt)   # time matrix
    ys = np.zeros((N + 1)*neurons).reshape(neurons,N+1) #voltage matrix 
    Pre = np.zeros((N+1)*neurons).reshape(neurons,N+1) #synaptic integration matrix 
    pre_pre = np.zeros((N+1)*neurons).reshape(neurons,N+1) #synaptic integration matrix, used for euler
    
    Vth = np.zeros((N+1)*neurons).reshape(neurons,N+1) # Threshold matrix (used when we have adaptation)
    
    
    ######################## Defining the connections each neuron receives #########################
    # First we define the excitatory synapses by multiplying the number of connections per neuron C_e   
    # with neurons*(1/(1-gamma))
    # Same thing for inhibitory neurons but with C_i multiplied by neurons*(1/gamma))
    # gamma is the fraction of inhibitory neurons divided by the total number of neurons.
    random_list = []
    neurons_index = np.arange(0,neurons,1)
    for w in range(0,int(neurons-gamma*neurons)):
        random_choice = np.random.choice(neurons_index,size = int(C_e*neurons*(1/(1-gamma)))) 
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
    bonus = np.zeros(neurons).reshape(neurons,1) ### matrix that spikes are integrated at and kept track of.
    for i in range(1, ts.size): ### one time step at each iteration
       
    ################################################################################################
    
        for w in range(0,int(neurons-gamma*neurons)): ### updating excitatory neurons
            pre_pre[w][i] += J_e*bonus[w][0] ### exc synaptic strength multiplied by bonus matrix (look line 110)
            
            Vth[w][0] = 20 # initial value of threshold potential , could be defined outside the loop 
            vth = Vth[w][i-1] ###########################
            Vth[w][i] = vth + ((20 - vth)/t_vth) * dt ### euler for threshold potential 
          
        for w in range(0,int(neurons-gamma*neurons)): ### Euler for synaptic integration for exc neurons 
            pre =Pre[w][i-1] 
            Pre[w][i] = pre + ((- pre/t_e) + pre_pre[w][i-1]) * dt ## Euler
            matrixx[w][i] = I + 1000*Pre[w][i] + poisson_spikes[i] ## matrix 'matrixx' summs all kinds of inputs (external , network)
            
            
        
        for w in range(int(neurons-gamma*neurons),neurons): ### Same process for inhibitory neurons 
            pre_pre[w][i] += J_i*bonus[w][0]
            
            Vth[w][0] = 20 
            vth = Vth[w][i-1]
            Vth[w][i] = vth + ((20 - vth)/t_vth) * dt 
            
            
            
        for w in range(int(neurons-gamma*neurons),neurons):
            pre =Pre[w][i-1]
            Pre[w][i] = pre + ((- pre/t_i) +  pre_pre[w][i-1]) * dt 
            matrixx[w][i] = I + 1000*Pre[w][i] + poisson_spikes[i]
            
       ############################################################################################     
            
            
        bonus = np.zeros(neurons).reshape(neurons,1) ### The matrix that keeps track of the spikes is reset 
        ### because we are about to initiate a new time step.
        ### Very important , the bonus matrix adds all the spikes that keeps track on the next time step.
        ### Meaning, we need to keep track of spike generation in this time step , in order to add them at the 
        ### next time step, using the code at lines 115-141
           
        
        for k in range(0,neurons): ### Going through each neuron 
            
            
            if tik[k][0] > i* dt :
                ys[k][i] = 10
  
            else:
        
                t = t_init + (i - 1) * dt
      
                y = ys[k][i-1]
                ys[k][i] = y + mu(y,k,i-1) * dt + sigma(t) * dW(dt)  ### Euler for each neuron
             
                if ys[k][i]>Vth[k][i]:  ### Checking if a neuron is over the threshold 
                
                    ys[k][i] = 80 ### If it is over the threshold, voltage is set to 80mV 
                    
                    Vth[k][i] = Vth[k][i] + a  ### and adaptation a is added to the threshold potential 
                    
                    tik[k][0] = (i)*dt + t_ref ### tik keeps track of the refractory period 
                    
                    spikes += 1 ### measuring total spikes
                
                    
                    
                    if k > (neurons-gamma*neurons): ## if neuron is inhibitory then bonus gets -g which is the relative synaptic strength
                        for r in random_list[k]:
                            if r < (neurons-gamma*neurons):
                                bonus[r][0] =(bonus[r][0] - g)
                                
                            elif r > (neurons-gamma*neurons) or r == (neurons-gamma*neurons):
                                bonus[r][0] = (bonus[r][0] - g)
                        
                    else:
                        for r in random_list[k]:  ### else if neuron is excitatory then it gets +1 
                            if r < (neurons-gamma*neurons):
                                bonus[r][0] =(bonus[r][0] + 1)
                                
                            elif r > (neurons-gamma*neurons) or r == (neurons-gamma*neurons):
                                bonus[r][0] = (bonus[r][0] + 1)
                                
    ##########################################################################################

    # This part is for appending the spike times for each neuron # 
    # Usually helps to plot a raster plot # 
    # Not the most efficient code ? # 
    for v in range(0,1000):
        first_order_time = []
        for x in range(0,ts.size):
              if ys[v][x] == 80 :
                  first_order_time.append(x*dt)
              else:
                  continue
        
        sptime.append(first_order_time)
        
        
        
    ##########################################################################################
    
    # This is calculating the number of spikes at each time step #
    freq_list = np.zeros(ts.size)
    for x in range(0,ts.size-1):
        frequency = 0 
        for v in range(0,int(1000)):       
            if ys[v][x] == 80 :
                frequency += 1
            else:
                continue
        
        freq_list[x] = frequency
        
    # Those last two could be done more effiently # 
        
        
        
    return ys, ts, Pre,matrixx, float(spikes)/float(neurons), sptime, Vth, freq_list



############################ Max interval Burst algorithm ############################

def burst(ys_neuron):
    
    
    spike_time = []
    counting = 0
    for i in ys_neuron:
        counting += 1
        if i == 80:
            spike_time.append(counting)
            
            
    isi = np.zeros(len(spike_time))
    count_spikes = 1
    burst_duration = 0 
    burst_beginning = []
    burst_end = []
    IBI = []
    for k in range(1,(len(spike_time))):
        isi[k] = spike_time[k] - spike_time[k-1]
        
        
        if isi[k] < 45 and (isi[k-1] > 45 or isi[k-1] == 0):
            
            burst_beginning.append(spike_time[k-1])
            count_spikes += 1
            burst_duration += isi[k-1]
            
                           
        elif isi[k] > 45 and (isi[k-1] < 45 or isi[k-1] == 45) and burst_duration > 40:
            
            burst_end.append(spike_time[k-1])
       
            count_spikes = 0
            burst_duration = 0
            
        elif isi[k] < 45:
            count_spikes += 1
            burst_duration += isi[k-1]
            
            
    
            
    return burst_beginning, burst_end, burst_duration, isi, count_spikes , spike_time, IBI


########### Custom burst detector  ###########

def custom_burst(vector):
    
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
Ni = 0.2*1000 # Define number of inhibitory neurons 
Ne = 1000-Ni # Define number of excitatory neurons 
C_e = 0.078*Ne + 17.6 
C_e = C_e / 1000 ### Use this C_e when calling the network function 

#### Call the network function with the desired values for the parameters ####
qe = network()



####################################### Raster plot #########################################
neuralData = qe[5]
plt.eventplot(neuralData,orientation =  'horizontal' , color = 'black')     

# Provide the title for the spike raster plot

plt.title('Spike raster plot')

 

# Give x axis label for the spike raster plot

plt.xlabel('Time')

 

# Give y axis label for the spike raster plot

plt.ylabel('Neuron')

 

# Display the spike raster plot
plt.show()




####### Calling max interval burst detection algorithm #######

print(burst((qe[0])[790])[0])
print(burst((qe[0])[790])[1])
print(burst((qe[0])[790])[3])
print(burst((qe[0])[790])[4])
print(burst((qe[0])[790])[4])

#### Calling custom burst detection algorithm  #####

print(burst(qe[7])[0])
print(burst(qe[7])[1])
print(burst(qe[7])[2])
print(burst(qe[7])[3])
print(burst(qe[7])[4])


