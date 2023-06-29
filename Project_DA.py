#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as grb
from datetime import datetime
import numpy as np
import pyomo.environ
from pyomo.environ import *
import pyomo
from pyomo.core.expr.current import evaluate_expression as evaluate
import random


# In[22]:


# Simulation Inputs
Max_SOC            = 10
Ch_power           = 1
Dis_power          = 1


# In[23]:


def gen_timeseries(T,mean=50,amplitude=10,peak_h=12,noise=20,lb=None,ub=None,seed=42):
    """Simple function to generate synthetic timeseries. Generates a sine centered around 'mean' 
        with an 'amplitude'. The sine curve repeats daily peaking at 'peak_h'. 
        Noise from a standard normal distribution with std = 'noise' is added. 
        The timeseries can be clipped at 'lb' and 'ub'
     """
    np.random.seed(seed)
    ts = pd.Series(index=T,
                data=mean + amplitude*np.sin((T.hour-peak_h+6)/12*np.pi) + noise*np.random.standard_normal(len(T)))
    if lb is not None:
        ts[ts<lb]=lb
    if ub is not None:
        ts[ts>ub]=ub
    return ts


# In[24]:


T             = pd.date_range('2021-12-01T00:00','2022-11-30T23:00',freq='H') 
C_da          = gen_timeseries(T,mean=40,noise=20,lb=0)
P_gen_max     = gen_timeseries(T,mean=.6,amplitude=0,noise=.6,lb=0.1,ub=1,peak_h=2)


# In[25]:


# Real Data Price
with open('DA_price.txt', 'r') as f:
    c_DA_temp = np.array(f.read().splitlines(), dtype=float)
    
# Real Data Power
with open('total_wind.txt', 'r') as f:
    P_temp    = np.array(f.read().splitlines(), dtype=float)/10000

# Filter
y       = c_DA_temp
x       = P_temp
mask    = ~np.isnan(y)
y_clean = y[mask]
x_clean = x[mask]

C_da.loc[:]      = y_clean
P_gen_max.loc[:] = x_clean

# plot
fig, ax = plt.subplots()
ax.plot(C_da.index, C_da.values)
ax.set(xlabel='Time', ylabel='Price [EUR]',
       title='Day-ahead Market Price')
ax.grid()
plt.show()


# In[26]:


# plot power
fig, ax = plt.subplots()
ax.plot(P_gen_max.index, P_gen_max.values)
ax.set(xlabel='Time', ylabel='Power [MW]',
       title='Generated Power')
ax.grid()
plt.show()


# In[27]:


m   = ConcreteModel()
m.T = Set(initialize=T) # Time indexes


# In[28]:


### Input Parameters ###
# Store 
# Simulation Inputs
m.P_ch_max  = Param(initialize=Dis_power)  # Maximum charge [MW]
m.P_dis_max = Param(initialize=Ch_power)  # Maximum discharge [MW]
m.SOC_max   = Param(initialize=Max_SOC) # Storage capacity [MWh]
SOC_start   = 5                    # Initial SOC
m.eta_ch    = Param(initialize=.8) # Charge efficiency
m.eta_dis   = Param(initialize=.9) # Discharge efficiency 

# marginal cost of generator
m.c_gen     = Param(initialize=0.001) #€/MWh

# Load (Electric resistor)
m.P_load_max = Param(initialize=0.1) # Maximum generator [MW]
m.P_load_min = Param(initialize=0)  # Minimum generator [MW]

# Grid connection 
m.G_max      = Param(initialize=20) # Sice of grid connection [MW]

# Costs
m.c_inverter  = Param(initialize=0.001) #€/MWh
m.C_da        = Param(T,initialize=C_da)


# In[29]:


### Model Variables ###
# Store 
m.P_ch     = Var(m.T,domain=NonNegativeReals,bounds=(0,m.P_ch_max))
m.P_dis    = Var(m.T,domain=NonNegativeReals,bounds=(0,m.P_dis_max))
m.SOC      = Var(m.T,domain=NonNegativeReals,bounds=(0,m.SOC_max ))

# Generator 
P_max            = P_gen_max.values
upper_bound_dict = dict(zip(T, P_max))
m.P_gen          = Var(m.T)
T_length         = len(T)

for t in m.T:
    t_index = T.get_loc(t)  # Get the index of the current timestamp
    if t_index < T_length:
            m.P_gen[t].setub(upper_bound_dict[t])
    else:
        print("Error: Timestamp exceeds the length of the upper_bound_array.")

# Load
m.P_load   = Var(m.T,domain=NonNegativeReals,bounds=(m.P_load_min,m.P_load_max))

# Grid connection 
m.G        = Var(m.T,domain=Reals,bounds=(-m.G_max,m.G_max))

# Day-ahead participation 
m.P_da     = Var(m.T,domain=Reals)


# In[30]:


### Constraints ###
# Storage constraints
def soc_rule(m,t):
    if t == T[0]:
        return m.SOC[T[0]]==SOC_start
    else :
        return m.SOC[t] == m.SOC[T[np.where(T==t)[0][0]-1]] - m.P_dis[t]*1/m.eta_dis+m.P_ch[t]*m.eta_ch

m.soc_con = Constraint(m.T,rule=soc_rule)

# Energy balance 
m.eb_int = Constraint(m.T,rule=lambda m,t: m.G[t]==m.P_dis[t]-m.P_ch[t]+m.P_gen[t]-m.P_load[t])


# In[31]:


# Objective function
rev_markets    = lambda t: m.G[t]*m.C_da[t] 
rev_operations = lambda t: m.P_dis[t]*m.c_inverter + m.P_ch[t]*m.c_inverter + m.P_gen[t]*m.c_gen
m.rev          = Var(m.T,domain=Reals)
m.rev_cons     = Constraint(m.T,rule=lambda m,t: m.rev[t]==rev_markets(t)-rev_operations(t))
m.obj          = Objective(expr=sum([m.rev[t] for t in m.T] ),sense=maximize)


# In[32]:


solver = SolverFactory('gurobi')
solver.solve(m) 


# In[33]:


total_rev = sum(m.rev[t].value for t in m.T)
total_rev_million_eur = total_rev / 1000000

print("Total Revenue (in million Euros):", total_rev_million_eur)


# In[34]:


# General plot
plt.figure(figsize=(12,6))
plt.plot(T,[m.G[t].value for t in T],lw=3, label='Grid')
plt.plot(T,P_gen_max, label='Maximum Wind Power')
plt.plot(T,[m.P_gen[t].value for t in T],label='Generated Wind Power')
plt.plot(T,[m.P_dis[t].value for t in T],label='Discharged')
plt.plot(T,[m.P_ch[t].value for t in T],label='Charged')
plt.title("General Figure")
plt.ylabel('Power (MW)')
plt.legend()
plt.show()


# In[41]:


# General plot for one week
start_date = '2022-05-01T00:00'
end_date   = '2022-05-07T23:00'
T_subset   = T[(T >= start_date) & (T <= end_date)]

plt.figure(figsize=(12,6))
plt.plot(T_subset,[m.G[t].value for t in T_subset],lw=3, label='Grid')
plt.plot(T_subset,[P_gen_max[t] for t in T_subset],lw=2, label='Maximum Wind Power')
plt.plot(T_subset,[m.P_gen[t].value for t in T_subset],label='Generated Wind Power')
plt.plot(T_subset,[m.P_dis[t].value for t in T_subset],label='Charged')
plt.plot(T_subset,[m.P_ch[t].value for t in T_subset],label='Discharged')
plt.title("General Figure")
plt.ylabel('Power (MW)')
plt.legend()
plt.show()
# 2 subplotss one for duration and the other plot


# In[36]:


# Plot for Day-ahead market
plt.figure(figsize=(12,6))
plt.stackplot(T,[m.G[t].value*m.C_da[t] for t in T])
plt.title("Day-Ahead Market Participation")
plt.ylabel("Euro")
plt.show()


# In[37]:


# Subset T for the first week of February
start_date = '2022-02-01T00:00'
end_date   = '2022-02-07T23:00'
T_subset   = T[(T >= start_date) & (T <= end_date)]

plt.figure(figsize=(12,6))
plt.stackplot(T_subset,[m.G[t].value*m.C_da[t] for t in T_subset])
plt.title("Day-Ahead Market Participation")
plt.ylabel("Euro")
plt.show()


# In[38]:


# Plot for BESS
plt.figure(figsize=(12,6))
plt.plot(T,[m.SOC[t].value for t in T],'--',alpha=.5,c='red',label='SOC')
plt.plot(T,[m.P_ch[t].value for t in T],'--',alpha=0.5,c='blue',label='P_ch')
plt.plot(T,[-m.P_dis[t].value for t in T],'--',alpha=0.5,c='orange',label='P_dis')
plt.legend()
plt.show()


# In[39]:


# Plot for BESS for one week 
start_date = '2022-03-01T00:00'
end_date   = '2022-03-08T00:00'
T_subset   = T[(T >= start_date) & (T <= end_date)]

plt.figure(figsize=(12,6))
plt.plot(T_subset,[m.SOC[t].value for t in T_subset],'--',alpha=.5,c='red',label='SOC')
plt.plot(T_subset,[m.P_ch[t].value for t in T_subset],'--',alpha=0.5,c='blue',label='P_ch')
plt.plot(T_subset,[-m.P_dis[t].value for t in T_subset],'--',alpha=0.5,c='orange',label='P_dis')
plt.legend()
plt.show()


# In[40]:


# Duration Curves
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Sort the data
sorted_grid                 = sorted([m.G[t].value for t in T], reverse=True)
sorted_max_wind_power       = sorted(P_gen_max, reverse=True)
sorted_generated_wind_power = sorted([m.P_gen[t].value for t in T], reverse=True)
sorted_discharged           = sorted([m.P_dis[t].value for t in T], reverse=True)
sorted_charged              = sorted([m.P_ch[t].value for t in T], reverse=True)

# Duration curve for Grid
axs[0, 0].plot(sorted_grid)
axs[0, 0].set_title("Grid")
axs[0, 0].set_ylabel('Cumulative Power (MW)')

# Duration curve for Maximum Wind Power
axs[0, 1].plot(sorted_max_wind_power)
axs[0, 1].set_title("Maximum Wind Power")
axs[0, 1].set_ylabel('Cumulative Power (MW)')

# Duration curve for Generated Wind Power
axs[1, 0].plot(sorted_generated_wind_power)
axs[1, 0].set_title("Generated Wind Power")
axs[1, 0].set_ylabel('Cumulative Power (MW)')

# Duration curve for Discharged
axs[1, 1].plot(sorted_discharged)
axs[1, 1].set_title("Discharged")
axs[1, 1].set_ylabel('Cumulative Power (MW)')

# Duration curve for Charged
axs[2, 0].plot(sorted_charged)
axs[2, 0].set_title("Charged")
axs[2, 0].set_ylabel('Cumulative Power (MW)')

fig.delaxes(axs[2, 1])
plt.tight_layout()
plt.show()

