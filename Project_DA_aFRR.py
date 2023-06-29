#!/usr/bin/env python
# coding: utf-8

# In[71]:


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
import seaborn as sns


# In[72]:


# Inputs 
seed_value         = 10
percent_zero_value = 50
scenario_num       = 1
Max_SOC            = 10
Ch_power           = 1
Dis_power          = 1
init_SOC           = 0
grid_limit         = 20


# In[73]:


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


# In[74]:


def generate_random_series(T, seed, percent_zero=0):
    """
    Generates a Pandas Series of random numbers between a lower and upper bound,
    with a specified seed value and a specified percentage of zeros.

    Parameters:
        T (pd.DatetimeIndex): the index for the output Series, as a Pandas DatetimeIndex
        seed (int): the seed value for the random number generator
        percent_zero (float): the percentage of numbers to set to zero (default 0)
    """
    random.seed(seed)
    z    = np.array([random.uniform(0, 1) for i in range(len(T))])>(percent_zero / 100)
    r    = np.array([random.uniform(0, 1) for i in range(len(T))])
    r    = 2*r-1
    up   = r*(r>0)*z
    down = -r*(r<0)*z
    
    # create the Series with index T for both up and down
    up   = pd.Series(data=up, index=T)
    down = pd.Series(data=down, index=T)
    
    return up,down


# In[75]:


T             = pd.date_range('2021-12-01T00:00','2022-11-30T23:00',freq='H') 
C_da          = gen_timeseries(T,mean=40,noise=20,lb=0)
P_gen_max     = gen_timeseries(T,mean=.6,amplitude=0,noise=.6,lb=0.1,ub=1,peak_h=2)
aFRR_up       = gen_timeseries(T,mean=40,noise=20,lb=0)
aFRR_down     = gen_timeseries(T,mean=40,noise=20,lb=0)
C_af_cap      = gen_timeseries(T,mean=40,noise=20,lb=0)


# In[76]:


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
C_af_cap.loc[:]  = y_clean/4
P_gen_max.loc[:] = x_clean

# plot
fig, ax = plt.subplots()
ax.plot(C_da.index, C_da.values)
ax.set(xlabel='Time', ylabel='Price [EUR]',
       title='Day-ahead Market Price')
ax.grid()
plt.show()


# In[77]:


# plot power
fig, ax = plt.subplots()
ax.plot(P_gen_max.index, P_gen_max.values)
ax.set(xlabel='Time', ylabel='Power [MW]',
       title='Generated Power')
ax.grid()
plt.show()


# In[78]:


# Mean of generated wind power
mean  = np.mean(P_gen_max)
mean1 = mean/8
print("Mean of P_gen_max:", mean, "MWh")
print("Mean of P_gen_max:", mean1, "MWh for one turbine")


# In[79]:


# Randam up and down generation
r  = 2*gen_timeseries(T,mean=-1,amplitude=.9,noise=.9,lb=0,ub=1,peak_h=14,seed=1)-1
z  = gen_timeseries(T,mean=-1,amplitude=.9,noise=.9,lb=0.5,ub=1,peak_h=14,seed=1)

# Random number generation (the function that I wrote)
[up,down]  = generate_random_series(T, seed=seed_value, percent_zero=percent_zero_value)


# In[80]:


# Plot the "up" series
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.step(T, up)
ax1.step(T, -down)
ax1.plot(T, up*down)
ax1.set_xlabel('Date')
ax1.set_ylabel('Up')
ax1.set_title('Up and Down Series')
plt.show()


# In[81]:


# aFRR up
with open('aFRR_up.txt', 'r') as f:
    aFRR_up_temp   = np.array(f.read().splitlines(), dtype=float)*0.1341
    
# aFRR down
with open('aFRR_down.txt', 'r') as f:
    aFRR_down_temp = np.array(f.read().splitlines(), dtype=float)*0.1341

aFRR_up.loc[:]     = aFRR_up_temp
aFRR_down.loc[:]   = aFRR_down_temp

# Plot
plt.figure(figsize=(12, 8))
plt.plot(T, aFRR_up_temp, label='aFRR_up')
plt.plot(T, aFRR_down_temp, label='aFRR_down')
plt.xlabel('Date')
plt.ylabel('Price [EUR]')
plt.title('Plot of aFRR')
plt.legend()  
plt.show()


# In[82]:


m   = ConcreteModel()
m.T = Set(initialize=T) # Time indexes
m.s = Set(initialize=[0,1,2]) # Scenarios, 0 = down, 1= realisation, 2= up
ref_scenario = scenario_num


# In[83]:


# Up activation
def init_up(m,t,s_i):
    if s_i==0:
        return 0
    elif s_i==2: 
        return 1
    elif s_i == 1: 
        return up[t]
m.up = Param(m.T,m.s,initialize=init_up,domain=Reals) 

# Down activation
def init_down(m,t,s_i):
    if s_i==0:
        return 1
    elif s_i==2: 
        return 0
    elif s_i == 1: 
        return down[t]
m.down = Param(m.T,m.s,initialize=init_down,domain=Reals)


# In[84]:


### Input Parameters ###
# Store 
m.P_ch_max  = Param(initialize=Ch_power)  # Maximum charge [MW]
m.P_dis_max = Param(initialize=Dis_power)  # Maximum discharge [MW]
m.SOC_max   = Param(initialize=Max_SOC)  # Storage capacity [MWh]
SOC_start   = init_SOC                    # Initial SOC #explain report
m.eta_ch    = Param(initialize=.8) # Charge efficiency  # ref?
m.eta_dis   = Param(initialize=.9) # Discharge efficiency 

# marginal cost of generator
m.c_gen     = Param(initialize=0.001) #€/MWh

# Load (Electric resistor)
m.P_load_max = Param(initialize=0.1) # Maximum generator [MW]
m.P_load_min = Param(initialize=0)  # Minimum generator [MW]

# Grid connection 
m.G_max      = Param(initialize=grid_limit) # Sice of grid connection [MW]

# Costs
m.c_inverter  = Param(initialize=0.001) #€/MWh
m.C_da        = Param(T,initialize=C_da)
m.C_aFRR_up   = Param(T,initialize=aFRR_up) 
m.C_aFRR_down = Param(T,initialize=aFRR_down) 
m.C_af_cap    = Param(T,initialize=C_af_cap) 


# In[85]:


P_max = P_gen_max.values


# In[86]:


### Model Variables ###
# Store 
m.P_ch     = Var(m.T,m.s,domain=NonNegativeReals,bounds=(0,m.P_ch_max))
m.P_dis    = Var(m.T,m.s,domain=NonNegativeReals,bounds=(0,m.P_dis_max))
m.SOC      = Var(m.T,m.s,domain=NonNegativeReals,bounds=(0,m.SOC_max ))

# Generator 
upper_bound_dict = dict(zip(T, P_max))
m.P_gen          = Var(m.T, m.s)
T_length         = len(T)

for t in m.T:
    t_index = T.get_loc(t)  # Get the index of the current timestamp
    if t_index < T_length:
        for s in m.s:
            m.P_gen[t, s].setub(upper_bound_dict[t])
    else:
        print("Error: Timestamp exceeds the length of the upper_bound_array.")

# Load
m.P_load   = Var(m.T,m.s,domain=NonNegativeReals,bounds=(m.P_load_min,m.P_load_max))

# Grid connection 
m.G        = Var(m.T,m.s,domain=Reals,bounds=(-m.G_max,m.G_max))

# Day-ahead participation 
m.P_da     = Var(m.T,domain=Reals)

# aFRR 
m.P_af_up_promis      = Var(m.T,domain=NonNegativeReals)
m.P_af_up_delivered   = Var(m.T,m.s,domain=NonNegativeReals)
m.P_af_down_promis    = Var(m.T,domain=NonNegativeReals)
m.P_af_down_delivered = Var(m.T,m.s,domain=NonNegativeReals)


# In[87]:


### Constraints ###
# Storage constraints
def soc_rule(m,t,s):
    if t == T[0]:
        return m.SOC[T[0],s]==SOC_start
    else :
        return m.SOC[t,s] == m.SOC[T[np.where(T==t)[0][0]-1],s] - m.P_dis[t,s]*1/m.eta_dis+m.P_ch[t,s]*m.eta_ch

m.soc_con = Constraint(m.T,m.s,rule=soc_rule)

# aFRR constraints
m.af_up_cons   = Constraint(m.T,m.s,rule=lambda m,t,s: m.P_af_up_delivered[t,s] == m.P_af_up_promis[t]*m.up[t,s])
m.af_down_cons = Constraint(m.T,m.s,rule=lambda m,t,s: m.P_af_down_delivered[t,s] == m.P_af_down_promis[t]*(m.down[t,s]))

# Energy balance internal
m.eb_int = Constraint(m.T,m.s,rule=lambda m,t,s: m.G[t,s]==m.P_dis[t,s]-m.P_ch[t,s]+m.P_gen[t,s]-m.P_load[t,s])

# Energy Balance external
m.eb_ext = Constraint(m.T,m.s,rule=lambda m,t,s: m.G[t,s]==m.P_da[t]+m.P_af_up_delivered[t,s]-m.P_af_down_delivered[t,s])


# In[88]:


# Objective function
#rev_markets    = lambda t: m.P_da[t]*m.C_da[t] + m.P_af_up_promis[t]*m.C_af_cap[t]*2 +m.P_af_down_promis[t]*m.C_af_cap[t] + m.P_af_up_delivered[t,ref_scenario]*(m.C_da[t]+m.C_aFRR_up[t])+m.P_af_down_delivered[t,ref_scenario]*(m.C_da[t]+m.C_aFRR_down[t])
rev_markets    = lambda t: m.P_da[t]*m.C_da[t] + (m.P_af_up_promis[t]+m.P_af_down_promis[t])*m.C_af_cap[t] + m.P_af_up_delivered[t,ref_scenario]*(m.C_da[t]+m.C_aFRR_up[t])+m.P_af_down_delivered[t,ref_scenario]*(m.C_da[t]+m.C_aFRR_down[t])
rev_operations = lambda t: sum(m.P_dis[t,s]*m.c_inverter + m.P_ch[t,s]*m.c_inverter + m.P_gen[t,s]*m.c_gen for s in m.s)
m.rev          = Var(m.T,domain=Reals)
m.rev_cons     = Constraint(m.T,rule=lambda m,t: m.rev[t]==rev_markets(t)-rev_operations(t))
m.obj          = Objective(expr=sum([m.rev[t] for t in m.T] ),sense=maximize)


# In[89]:


solver = SolverFactory('gurobi')
solver.solve(m) 


# In[90]:


colors = ['#FF0000', '#00008B', '#FFA500', '#40E0D0', '#FFFF00']
plt.figure(figsize=(12, 7))
plt.stackplot(T,
              [m.P_da[t].value * m.C_da[t] for t in T],
              [m.P_af_up_promis[t].value * m.C_af_cap[t] for t in T],
              [m.P_af_down_promis[t].value * m.C_af_cap[t] for t in T],
              [m.P_af_up_delivered[t, ref_scenario].value * (m.C_da[t] + m.C_aFRR_up[t]) for t in T],
              [m.P_af_down_delivered[t, ref_scenario].value * (m.C_da[t] + m.C_aFRR_down[t]) for t in T],
              labels=['P_da', 'aFRR up capacity', 'aFRR down capacity', 'aFRR up energy', 'aFRR down energy'],
              colors=colors,
              linewidth=1.5)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue [€]', fontsize=12)
plt.title('Revenue obtained from Market Categories', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[91]:


# Subset T for the first week of February
start_date = '2022-06-01T00:00'
end_date = '2022-06-07T23:00'
T_subset = T[(T >= start_date) & (T <= end_date)]

colors = ['#FF0000', '#00008B', '#FFA500', '#40E0D0', '#FFFF00']
plt.figure(figsize=(12, 7))
plt.stackplot(T_subset,
              [m.P_da[t].value * m.C_da[t] for t in T_subset],
              [m.P_af_up_promis[t].value * m.C_af_cap[t] for t in T_subset],
              [m.P_af_down_promis[t].value * m.C_af_cap[t] for t in T_subset],
              [m.P_af_up_delivered[t, ref_scenario].value * (m.C_da[t] + m.C_aFRR_up[t]) for t in T_subset],
              [m.P_af_down_delivered[t, ref_scenario].value * (m.C_da[t] + m.C_aFRR_down[t]) for t in T_subset],
              labels=['Day-ahead Market', 'aFRR Up Capacity Market', 'aFRR Down Capacity Market',
                      'aFRR Up Energy Market', 'aFRR Down Energy Market'],
              colors=colors,
              linewidth=1.5)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue [€]', fontsize=12)
plt.title('Revenue obtained from Market Categories', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[92]:


plt.figure(figsize=(10, 8))

plt.title("Percentage Distribution of Revenue by Market Categories", fontsize=18)

totals = [
    sum(abs(m.P_da[t].value * m.C_da[t]) for t in T),
    sum(abs(m.P_af_up_promis[t].value * m.C_af_cap[t]) for t in T),
    sum(abs(m.P_af_down_promis[t].value * m.C_af_cap[t]) for t in T),
    sum(abs(m.P_af_up_delivered[t, ref_scenario].value * (m.C_da[t] + m.C_aFRR_up[t])) for t in T),
    sum(abs(m.P_af_down_delivered[t, ref_scenario].value * (m.C_da[t] + m.C_aFRR_down[t])) for t in T)
]

labels = ['Day-ahead Market', 'aFRR Up Capacity Market', 'aFRR Down Capacity Market', 'aFRR Up Energy Market', 'aFRR Down Energy Market']

colors = ['#FF0000', '#00008B', '#FFA500', '#40E0D0', '#FFFF00']

explode = (0.1, 0.1, 0.1, 0.1, 0.1)

label_distances = np.array([0.15, 0, 0.1, 0, 0.2], dtype=float)  # Adjust the label distances individually

pie, _ = plt.pie(totals, labels=labels, colors=colors, explode=explode,
                 shadow=True, startangle=140, textprops={'fontsize': 12, 'fontweight': 'bold'},
                 labeldistance=None)  # Set labeldistance to None

legend_labels = [f'{label} ({total/sum(totals)*100:.2f}%)' for label, total in zip(labels, totals)]  # Generate labels with percentages (2 decimal places)

for i, (label, distance) in enumerate(zip(legend_labels, label_distances)):
    angle = (pie[i].theta2 - pie[i].theta1) / 2 + pie[i].theta1  # Calculate the mid-angle of the wedge
    x = np.cos(np.deg2rad(angle))
    y = np.sin(np.deg2rad(angle))
    xt = x + distance
    yt = y + distance
    alignment = 'left' if xt > 0 else 'right'
    plt.text(xt, yt, label, ha=alignment, va='center', fontsize=14, fontweight='bold')

plt.axis('equal')
plt.show()


# In[93]:


# Compute the total revenue
total_rev = sum(m.rev[t].value for t in m.T)
total_rev_million_eur = total_rev / 1_000_000

print("Total Revenue (in million Euros):", total_rev_million_eur)

