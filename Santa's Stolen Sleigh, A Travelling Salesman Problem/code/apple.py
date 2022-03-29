# -*- coding: utf-8 -*-
# author: apple.py
# IDs: 10557066, 10365619, 9976380, 10309814, 10467760, 10474285, 10469133

#%% Importing necessary libraries

import numpy as np
import pandas as pd
from timeit import default_timer as timer
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from random import sample
import random
from random import choice
import seaborn as sns

#%% Evaluation Functions

def haversine_np(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between paired arrays representing
    points on the earth (specified in decimal degrees)

    All args must be numpy arrays of equal length.

    Returns an array of distances for each pair of input points.

    """
    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))

    # calculate haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    d = np.sin(dlat * 0.5)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5)**2
    c = 2.0 * 6371.0
    return c * np.arcsin(np.sqrt(d))


def weighted_trip_length(stops_latitude, stops_longitud, weights):
    north_pole = (90,0)
    sleigh_weight = 10
    dist = 0.0
    # Start at the north pole with the sleigh full of gifts.
    prev_lat, prev_lon = north_pole
    prev_weight = np.sum(weights) + sleigh_weight
    for lat, lon, weight in zip(stops_latitude, stops_longitud, weights):
        # Idea 1: Calculating the distances between the points repeatedly is
        # slow. Calculate all distances once into a matrix, then use that
        # matrix here.
        dist += haversine_np(lat, lon, prev_lat, prev_lon) * prev_weight   ##changed haversine
        prev_lat, prev_lon = lat, lon
        prev_weight -= weight

    # Last trip back to north pole, with just the sleigh weight
    dist += haversine_np(north_pole[0], north_pole[1], prev_lat, prev_lon) * sleigh_weight  ##changed haversine
        
    return dist

def weighted_reindeer_weariness(all_trips, weight_limit = 1000):
    uniq_trips = all_trips.TripId.unique()
    
    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")
 
    dist = 0.0
    for t in uniq_trips:
        # Idea 2: There may be better/faster/simpler ways to represent a solution.
        this_trip = all_trips[all_trips.TripId==t]
        dist += weighted_trip_length(this_trip.Latitude, this_trip.Longitude, this_trip.Weight)
    
    return dist

def sort_trip(dataset):
    weight = 0 # Starts the weight list count from 0
    trip_df = []
    trip_no = 1
    for i in range(len(dataset)):
        weight += dataset.iloc[i,3] # Adds the weight of each gift on every i-iteration index of the dataset to the weight list
        if weight <= 900.0:
            trip_df.append(trip_no) # Append to the trip number for each added gifts as long as it's within the 1000kg limit
        else:
            weight = dataset.iloc[i,3] # If it exceeds the weight limit, assign it to the next trip number 
            trip_no += 1
            trip_df.append(trip_no)
    #Combine the trip assigned to the dataframe
    trip_df = pd.DataFrame(trip_df)
    dataset = dataset.reset_index(drop=True)
    dataset = dataset.join(trip_df)
    dataset.rename(columns= {dataset.columns[4]: "TripId"}, inplace= True)
    return dataset

#%% Loading the dataset
    
gifts = pd.read_csv('gifts.csv')
gifts.head()

#%% Random Search Algorithm

def RandomSearch(dataset, N, iterations):
    print("\n------- Calling Random Search --------\n")
    solutions = []
    best = []
    seed = np.random.randint(0,100)
    for i in range(30):
        print("\n------- Random Search #", str(i), "--------\n")
        np.random.seed(seed*i)
        sample_set = dataset.sample(n=N)
        sample_set = sort_trip(sample_set)
        for r in range(iterations*N):
            new_set = sample_set.sample(N)
            sample_solution = weighted_reindeer_weariness(new_set)
            print("Solution", str(r+1), ":", sample_solution)
            solutions.append(sample_solution)
        rs_sols = pd.DataFrame(solutions)
        rs_sols.to_csv("rs_sols_" + str(N) + "_" + str(i+1) + ".csv") # Save all wrws for each random seed
        best.append(np.min(solutions))
        solutions.clear()
    print("\n------- Evaluations completed --------\n")
    print('\n ---We did',i+1,'runs---\n''All our best wrws:',best) 
    return best

#%% Random Search with 10 sample size
    
start_time = timer()
best_RS_10 = RandomSearch(gifts, 1000, 10)
end_time = timer()
print("(Time: {:.2f} seconds)".format(end_time - start_time))
# 65.33 seconds

#%% Random Search with 100 sample size    
start_time = timer()
best_RS_10 = RandomSearch(gifts, 100, 100)
end_time = timer()
print("(Time: {:.2f} seconds)".format(end_time - start_time))

#%% Random Search with 1000 sample size    
start_time = timer()
best_RS_10 = RandomSearch(gifts, 1000, 100)
end_time = timer()
print("(Time: {:.2f} seconds)".format(end_time - start_time))

#%% Neighbourhood Move Function

""" 
Neighbourhood Move 3
"""
def NM3(sorted_dataset):
    all_TripIds = sorted_dataset.TripId.unique()
    
    if max(sorted_dataset.TripId)< 2:
        raise Exception("Can't do neighbourhood move because number of trips is less than 2")
    
    else:
    # Shuffle the gift list
        new_sol = sorted_dataset.sample(frac = 1, random_state = 512).reset_index()[['GiftId', 'Latitude','Longitude','Weight','TripId']]
    # Select one gift and give it a new TripId
        old_TripId = new_sol.iloc[0,4] # Get a random gift
        all_TripIds = all_TripIds.tolist()
        for t_id in all_TripIds:
            if t_id == old_TripId:
                all_TripIds.remove(t_id)
        new_TripId = choice(all_TripIds) # Get a random TripId
        new_sol.iloc[0,4] = new_TripId # Give new TripId to the gift    
    # Weight check
        while any(new_sol.groupby('TripId').Weight.sum() > 1000.0):
            if len(all_TripIds) == 1:
                new_sol = sorted_dataset
            else:
                new_TripId = choice(all_TripIds) 
                new_sol.iloc[0,4] = new_TripId
        
    return new_sol

"""
Neighbourhood Move 6
"""
def neighbourhood_move6(dataset):
    # Generate 3 random integer number from 1 to max trip id
    i = random.sample(range(1, max(dataset["TripId"])+1), 3)
    # Split the dataset into 3 randomly selected trips
    a = dataset[dataset["TripId"] == i[0]]
    b = dataset[dataset["TripId"] == i[1]]
    c = dataset[dataset["TripId"] == i[2]]
    # Generate 3 random integer from the len of selected trip to determine which gift will be the suffix point
    j = np.random.randint(len(a)) 
    k = np.random.randint(len(b)) 
    l = np.random.randint(len(c))
    r = [j, k, l]
    r = random.choice(r)
    # The total len of each dataset according to their trip id
    m = len(dataset[dataset["TripId"] == i[0]]) 
    n = len(dataset[dataset["TripId"] == i[1]])
    o = len(dataset[dataset["TripId"] == i[2]])
    # NM6
    a, b, c = a.append(b.iloc[r:]).drop(a.index[r:m]), b.append(c.iloc[r:]).drop(b.index[r:n]), c.append(a.iloc[r:]).drop(c.index[r:o])
    a["TripId"], b["TripId"], c["TripId"] = i[0], i[1], i[2] 

    swapped_set = dataset[dataset["TripId"] != i[0]]
    swapped_set = swapped_set[swapped_set["TripId"] != i[1]]
    swapped_set = swapped_set[swapped_set["TripId"] != i[2]]
    return swapped_set.append(a, ignore_index=True).append(b, ignore_index=True).append(c, ignore_index=True)

def NM6(dataset):
    swapped_set = neighbourhood_move6(dataset)     
    while any(swapped_set.groupby("TripId").Weight.sum() > 1000.0):
        swapped_set = neighbourhood_move6(swapped_set)
        if all(swapped_set.groupby("TripId").Weight.sum() < 1000.0):
            break
    return swapped_set

#%% Simulated Annealing Algorithm with NM3
    
def SimulatedAnnealing(dataset, N, T, alpha, iterations, algorithm):
    best_wrws = []
    seed = np.random.randint(0,100)
    for i in range(30):          # Iterate 30 times with different initial random seed
        print("\n------- Calling Simulated Annealing--------\n")
        np.random.seed(seed*i)
        sourcedata = dataset.sample(N)
        best_solution = sort_trip(sourcedata)
        best_wrw = weighted_reindeer_weariness(best_solution)  
        sa_nm3 = []          
        for r in range(iterations*N):     # Change the max evaluation number  
            new_solution = algorithm(best_solution) # Applying the first neighbourhood move
            new_wrw = weighted_reindeer_weariness(new_solution)
            print('New solution',r+1,':',new_wrw)  
            sa_nm3.append(new_wrw) # Save new wrws
            if new_wrw < best_wrw or random.random() < np.exp(((best_wrw - new_wrw)/best_wrw)/T):
                best_wrw = new_wrw
                best_solution = new_solution 
            T = T*alpha  
        print('\n------- Evaluations completed --------\n')   
        print('Best solution:',best_wrw)
        sa_nm3 = pd.DataFrame(sa_nm3)
        sa_nm3.to_csv("sa_nm3_1000_"+str(i+1)+".csv") # Save all wrws for each random seed
        print(sa_nm3.describe())    
        print('\n--------------------------------------\n')  
        sa_nm3 = [] 
        best_wrws.append(best_wrw)    
    print('\n ---We did',i+1,'runs---\n''All our best wrws:',best_wrws) 
    return best_wrws 

#%% Simulated Annealing with NM3 for 100 sample size

start_time = timer()
sa_nm3 = SimulatedAnnealing(gifts, 100, 1, 0.95, 10, NM3) #to save the solution
end_time = timer() 
print("(Time: {:.2f} seconds)".format(end_time - start_time))

#%% Simulated Annealing with NM3 for 1000 sample size

start_time = timer()
sa_nm3 = SimulatedAnnealing(gifts, 1000, 1, 0.95, 10, NM3) #to save the solution
end_time = timer() 
print("(Time: {:.2f} seconds)".format(end_time - start_time))


#%% Simulated Annealing with both NM3 and NM6

def SimulatedAnnealing2(dataset, N, T, alpha, iterations, algorithm, algorithm2):
    best_wrws = []
    seed = np.random.randint(0,100)
    for i in range(30):          # Iterate 30 times with different initial random seed
        print("\n------- Calling Simulated Annealing--------\n")
        np.random.seed(seed*i)
        sourcedata = dataset.sample(N)
        best_solution = sort_trip(sourcedata)
        best_wrw = weighted_reindeer_weariness(best_solution)  
        sa_2nm = []          
        for r in range(iterations*N):     # Change the max evaluation number  
            new_solution = algorithm(best_solution) # Applying the first neighbourhood move
            new_solution = algorithm2(best_solution)
            new_wrw = weighted_reindeer_weariness(new_solution)
            print('New solution',r+1,':',new_wrw)  
            sa_2nm.append(new_wrw) # Save new wrws
            if new_wrw < best_wrw or random.random() < np.exp(((best_wrw - new_wrw)/best_wrw)/T):
                best_wrw = new_wrw
                best_solution = new_solution 
            T = T*alpha  
        print('\n------- Evaluations completed --------\n')   
        print('Best solution:',best_wrw)
        sa_2nm = pd.DataFrame(sa_2nm)
        sa_2nm.to_csv("sa_2nm_1000_"+str(i+1)+".csv") # Save all wrws for each random seed
        print(sa_2nm.describe())    
        print('\n--------------------------------------\n')  
        sa_2nm = [] 
        best_wrws.append(best_wrw)    
    print('\n ---We did',i+1,'runs---\n''All our best wrws:',best_wrws) 
    return best_wrws 

#%% Simulated Annealing with both NM3 and NM6 for 1000 sample sizes

start_time = timer()
sa_2nm = SimulatedAnnealing2(gifts, 1000, 1, 0.95, 10, NM3, NM6) #to save the solution
end_time = timer() 
print("(Time: {:.2f} seconds)".format(end_time - start_time))

#%% Temperature Decay

start_time = timer()
sa_nm3 = SimulatedAnnealing(gifts, 1000, 1, 0.98, 10, NM3) #to save the solution
end_time = timer() 
print("(Time: {:.2f} seconds)".format(end_time - start_time))

start_time = timer()
sa_nm3 = SimulatedAnnealing(gifts, 1000, 1, 0.95, 10, NM3) #to save the solution
end_time = timer() 
print("(Time: {:.2f} seconds)".format(end_time - start_time))

start_time = timer()
sa_nm3 = SimulatedAnnealing(gifts, 1000, 1, 0.90, 10, NM3) #to save the solution
end_time = timer() 
print("(Time: {:.2f} seconds)".format(end_time - start_time))

start_time = timer()
sa_nm3 = SimulatedAnnealing(gifts, 1000, 1, 0.80, 10, NM3) #to save the solution
end_time = timer() 
print("(Time: {:.2f} seconds)".format(end_time - start_time))

start_time = timer()
sa_nm3 = SimulatedAnnealing(gifts, 1000, 1, 0.50, 10, NM3) #to save the solution
end_time = timer() 
print("(Time: {:.2f} seconds)".format(end_time - start_time))

#%% Plots

# ALGORITHM 1: RANDOM SEARCH

#Importing data
sol_10 = pd.read_csv('best_10_sols.csv')
sol_100 = pd.read_csv('best_100_sols.csv')
sol_1000 = pd.read_csv('best_1000_sols.csv')
#Preparing data
sol_10['sample_size'] = 10
sol_10.columns = ['seed_iteration','solution','sample_size']
sol_100['sample_size'] = 100
sol_100.columns = ['seed_iteration','solution','sample_size']
sol_1000['sample_size'] = 1000
sol_1000.columns = ['seed_iteration','solution','sample_size']
#Aggregating data
f_randomsearch = [sol_10, sol_100, sol_1000]
df_randomsearch = pd.concat(f_randomsearch)
df_randomsearch.to_csv("df_randomsearch.csv")

#Individual boxplot
sns.boxplot(y=sol_10["solution"]).set_title('Random Search with sample size = 10')
sns.boxplot(y=sol_100["solution"]).set_title('Random Search with sample size = 100')
sns.boxplot(y=sol_1000["solution"]).set_title('Random Search with sample size = 1000')
#plot_10 = plt.subplots()

#Individual scatter plot
# Plot N=10
plt.scatter(x=sol_10["seed_iteration"], y=sol_10["solution"], alpha=0.5)
plt.title('Random Search with sample size = 10')
plt.xlabel('index')
plt.ylabel('WRW')
plt.show()
# Plot N=100
plt.scatter(x=sol_100["seed_iteration"], y=sol_100["solution"], alpha=0.5)
plt.title('Random Search with sample size = 100')
plt.xlabel('index')
plt.ylabel('WRW')
plt.show()
# Plot N=1000
plt.scatter(x=sol_1000["seed_iteration"], y=sol_1000["solution"], alpha=0.5)
plt.title('Random Search with sample size = 1000')
plt.xlabel('index')
plt.ylabel('WRW')
plt.show()
# Grouped boxplot
#sns.boxplot(x="sample_size", y="0", hue="sample_size", data=df_randomsearch, palette="Set1")

""" Random Search Data
Info 10:
mean   4.017837e+06
std    1.338797e+06
min    1.128019e+06
max    7.444516e+06 
Time: 116.63 s

Info 100:
mean   2.823178e+08
std    1.416722e+07
min    2.540158e+08
max    3.193109e+08
Time: 2469.24 s

Info 1000:
mean   4.183253e+09
std    6.057952e+07
min    4.054347e+09
max    4.329836e+09
Time: 8392.38 s
"""
#Random Search with Sample Size N=10
def plotn10():
    # set Data
    Min = [1.128019e+06]
    Max = [7.444516e+06]
    Average = [4.017837e+06]
    Std = [1.338797e+06] 
    # Set position of bar on X axis
    r1 = np.arange(len(Min))
    r2 = [x + 0.2 for x in r1]
    r3 = [x + 0.2 for x in r2]
    r4 = [x + 0.2 for x in r3]
    #"#55DDE0", "#33658A", "#2F4858", "#F6AE2D", "#F26419"
    # Make the plot
    plt.bar(r1, Min, color='#55DDE0', width=0.2, edgecolor='white', label='Min')
    plt.bar(r2, Max, color='#33658A', width=0.2, edgecolor='white', label='Max')
    plt.bar(r3, Average, color='#F26419', width=0.2, edgecolor='white', label='Average')
    plt.bar(r4, Std, color='#F6AE2D', width=0.2, edgecolor='white', label='Std')
    # Remove xticks on the middle of the group bars
    plt.title('Random Search with Sample Size N=10')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    # Create legend & Show graphic
    plt.legend()
    plt.show()
#Random Search with Sample Size N=100
def plotn100():
    # set Data
    Min = [2.540158e+08]
    Max = [3.193109e+08]
    Average = [2.823178e+08]
    Std = [1.416722e+07] 
    # Set position of bar on X axis
    r1 = np.arange(len(Min))
    r2 = [x + 0.2 for x in r1]
    r3 = [x + 0.2 for x in r2]
    r4 = [x + 0.2 for x in r3]
    #"#55DDE0", "#33658A", "#2F4858", "#F6AE2D", "#F26419"
    # Make the plot
    plt.bar(r1, Min, color='#55DDE0', width=0.2, edgecolor='white', label='Min')
    plt.bar(r2, Max, color='#33658A', width=0.2, edgecolor='white', label='Max')
    plt.bar(r3, Average, color='#F26419', width=0.2, edgecolor='white', label='Average')
    plt.bar(r4, Std, color='#F6AE2D', width=0.2, edgecolor='white', label='Std')
    # Remove xticks on the middle of the group bars
    plt.title('Random Search with Sample Size N=100')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    # Create legend & Show graphic
    plt.legend()
    plt.show()
#Random Search with Sample Size N=1000
def plotn1000():
    # set Data
    Min = [4.054347e+09]
    Max = [4.329836e+09]
    Average = [4.183253e+09]
    Std = [6.057952e+07] 
    # Set position of bar on X axis
    r1 = np.arange(len(Min))
    r2 = [x + 0.2 for x in r1]
    r3 = [x + 0.2 for x in r2]
    r4 = [x + 0.2 for x in r3]
    #"#55DDE0", "#33658A", "#2F4858", "#F6AE2D", "#F26419"
    # Make the plot
    plt.bar(r1, Min, color='#55DDE0', width=0.2, edgecolor='white', label='Min')
    plt.bar(r2, Max, color='#33658A', width=0.2, edgecolor='white', label='Max')
    plt.bar(r3, Average, color='#F26419', width=0.2, edgecolor='white', label='Average')
    plt.bar(r4, Std, color='#F6AE2D', width=0.2, edgecolor='white', label='Std')
    # Add xticks on the middle of the group bars
    plt.title('Random Search with Sample Size N=1000')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    # Create legend & Show graphic
    plt.legend()
    plt.show()
plotn10()
plotn100()
plotn1000()

# ALGORITHM 2: SA WITH NM3

#Importing csv
sa_nm3_1000_1 = pd.read_csv('sa_nm3_1000_1.csv')
sa_nm3_1000_2 = pd.read_csv('sa_nm3_1000_2.csv')
sa_nm3_1000_3 = pd.read_csv('sa_nm3_1000_3.csv')
sa_nm3_1000_4 = pd.read_csv('sa_nm3_1000_4.csv')
sa_nm3_1000_5 = pd.read_csv('sa_nm3_1000_5.csv')
sa_nm3_1000_6 = pd.read_csv('sa_nm3_1000_6.csv')
sa_nm3_1000_7 = pd.read_csv('sa_nm3_1000_7.csv')
sa_nm3_1000_8 = pd.read_csv('sa_nm3_1000_8.csv')
sa_nm3_1000_9 = pd.read_csv('sa_nm3_1000_9.csv')
sa_nm3_1000_10 = pd.read_csv('sa_nm3_1000_10.csv')
sa_nm3_1000_11 = pd.read_csv('sa_nm3_1000_11.csv')
sa_nm3_1000_12 = pd.read_csv('sa_nm3_1000_12.csv')
sa_nm3_1000_13 = pd.read_csv('sa_nm3_1000_13.csv')
sa_nm3_1000_14 = pd.read_csv('sa_nm3_1000_14.csv')
sa_nm3_1000_15 = pd.read_csv('sa_nm3_1000_15.csv')
sa_nm3_1000_16 = pd.read_csv('sa_nm3_1000_16.csv')
sa_nm3_1000_17 = pd.read_csv('sa_nm3_1000_17.csv')
sa_nm3_1000_18 = pd.read_csv('sa_nm3_1000_18.csv')
sa_nm3_1000_19 = pd.read_csv('sa_nm3_1000_19.csv')
sa_nm3_1000_20 = pd.read_csv('sa_nm3_1000_20.csv')
sa_nm3_1000_21 = pd.read_csv('sa_nm3_1000_21.csv')
sa_nm3_1000_22 = pd.read_csv('sa_nm3_1000_22.csv')
sa_nm3_1000_23 = pd.read_csv('sa_nm3_1000_23.csv')
sa_nm3_1000_24 = pd.read_csv('sa_nm3_1000_24.csv')
sa_nm3_1000_25 = pd.read_csv('sa_nm3_1000_25.csv')
sa_nm3_1000_26 = pd.read_csv('sa_nm3_1000_26.csv')
sa_nm3_1000_27 = pd.read_csv('sa_nm3_1000_27.csv')
sa_nm3_1000_28 = pd.read_csv('sa_nm3_1000_28.csv')
sa_nm3_1000_29 = pd.read_csv('sa_nm3_1000_29.csv')
sa_nm3_1000_30 = pd.read_csv('sa_nm3_1000_30.csv')
#Preparing data
sa_nm3_1000_1['seed_iteration'] = 1
sa_nm3_1000_2['seed_iteration'] = 2
sa_nm3_1000_3['seed_iteration'] = 3
sa_nm3_1000_4['seed_iteration'] = 4
sa_nm3_1000_5['seed_iteration'] = 5
sa_nm3_1000_6['seed_iteration'] = 6
sa_nm3_1000_7['seed_iteration'] = 7
sa_nm3_1000_8['seed_iteration'] = 8
sa_nm3_1000_9['seed_iteration'] = 9
sa_nm3_1000_10['seed_iteration'] = 10
sa_nm3_1000_11['seed_iteration'] = 11
sa_nm3_1000_12['seed_iteration'] = 12
sa_nm3_1000_13['seed_iteration'] = 13
sa_nm3_1000_14['seed_iteration'] = 14
sa_nm3_1000_15['seed_iteration'] = 15
sa_nm3_1000_16['seed_iteration'] = 16
sa_nm3_1000_17['seed_iteration'] = 17
sa_nm3_1000_18['seed_iteration'] = 18
sa_nm3_1000_19['seed_iteration'] = 19
sa_nm3_1000_20['seed_iteration'] = 20
sa_nm3_1000_21['seed_iteration'] = 21
sa_nm3_1000_22['seed_iteration'] = 22
sa_nm3_1000_23['seed_iteration'] = 23
sa_nm3_1000_24['seed_iteration'] = 24
sa_nm3_1000_25['seed_iteration'] = 25
sa_nm3_1000_26['seed_iteration'] = 26
sa_nm3_1000_27['seed_iteration'] = 27
sa_nm3_1000_28['seed_iteration'] = 28
sa_nm3_1000_29['seed_iteration'] = 29
sa_nm3_1000_30['seed_iteration'] = 30
df_sa_nm3 = pd.concat([sa_nm3_1000_1,sa_nm3_1000_2,sa_nm3_1000_3,sa_nm3_1000_4,sa_nm3_1000_5,sa_nm3_1000_6,sa_nm3_1000_7,sa_nm3_1000_8,sa_nm3_1000_9,sa_nm3_1000_10,
                       sa_nm3_1000_11,sa_nm3_1000_12,sa_nm3_1000_13,sa_nm3_1000_14,sa_nm3_1000_15,sa_nm3_1000_16,sa_nm3_1000_17,sa_nm3_1000_18,sa_nm3_1000_19,sa_nm3_1000_20,
                       sa_nm3_1000_21,sa_nm3_1000_22,sa_nm3_1000_23,sa_nm3_1000_24,sa_nm3_1000_25,sa_nm3_1000_26,sa_nm3_1000_27,sa_nm3_1000_28,sa_nm3_1000_29,sa_nm3_1000_30])
df_sa_nm3.columns = ['index','solution','seed_iteration']
df_sa_nm3_1000_095 = df_sa_nm3
df_sa_nm3_1000_095.to_csv("df_sa_nm3_1000_095_300k.csv")
#best wrw of each iteration
#sa_nm3_1000_best = pd.DataFrame({'index':[0,1,2,3,4,5,6,7,8,9],
                                 #'solution':[4140171165.082735, 4228401155.9160647, 3997748642.9382157, 4147070462.7533193, 3865233227.2504716, 4214984063.9996696, 4238558044.8940206, 4191007747.578338, 4334700290.913228, 4221439989.3942857],
                                 #'seed_iteration':[1,2,3,4,5,6,7,8,9,10]
                                 #})   
sa_nm3_30_1000_095_best = pd.DataFrame({'index':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
                                 'solution':[4048286826.979431, 3914928656.38006, 4026001859.129481, 4110760440.659727, 4056670019.7908273, 3939233524.709127, 3983938236.2480984, 4059596122.024478, 4014283515.921572, 4068970174.614766, 4109832227.24275, 4095120080.7041903, 3964145314.5537605, 4057685009.0264807, 3853779478.7169094, 3896995215.5141716, 3914841828.082347, 4076818759.332067, 3929094015.263887, 4120546943.537638, 4050100423.3436317, 3931194369.443309, 4027415400.5120068, 4023497628.0582404, 4111352496.658318, 4008795571.543056, 4065555603.748015, 4026192425.661281, 4041861188.8086715, 4176828733.8995523],
                                 'seed_iteration':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
                                 })
sa_nm3_30_1000_095_best.to_csv("df_sa_nm3_30_1000_095_best_300k.csv")
sa_nm6_30_1000_best = pd.DataFrame({'index':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
                                 'solution':[3275221020.9897313, 3110852207.339792, 3148690896.9424267, 3141860171.191749, 3249002080.7698493, 3162717629.407945, 3092096859.232091, 3214851555.0546956, 3138384659.647428, 3251969825.598315, 3303284743.235824, 3234086237.085889, 3215458440.327368, 3111220443.956155, 3382816224.9812436, 3028228422.211967, 3352877112.583847, 3389937019.9828954, 3287589534.542054, 3141287530.257996, 3196343042.282548, 3554699940.7227635, 3269827046.1114964, 3137158496.5874457, 3406377056.8539343, 3412664127.7828965, 3291954988.4795256, 3218058710.322794, 3292709739.0160227, 3210345886.932815],
                                 'seed_iteration':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
                                 })
sa_nm6_30_1000_best.to_csv("df_sa_nm6_1000_best.csv")

#Importing data
df_sa_nm3_1000_098 = pd.read_csv('df_sa_nm3_1000_098.csv')
sa_nm3_30_1000_098_best = pd.read_csv('sa_nm3_30_1000_098_best.csv')
df_sa_nm3_1000_095 = pd.read_csv('df_sa_nm3_1000_095.csv')
sa_nm3_30_1000_095_best = pd.read_csv('sa_nm3_30_1000_095_best.csv')
df_sa_nm3_1000_090 = pd.read_csv('df_sa_nm3_1000_090.csv')
sa_nm3_30_1000_090_best = pd.read_csv('sa_nm3_30_1000_090_best.csv')
df_sa_nm3_1000_080 = pd.read_csv('df_sa_nm3_1000_080.csv')
sa_nm3_30_1000_080_best = pd.read_csv('sa_nm3_30_1000_080_best.csv')
df_sa_nm3_1000_050 = pd.read_csv('df_sa_nm3_1000_050.csv')
sa_nm3_30_1000_050_best = pd.read_csv('sa_nm3_30_1000_050_best.csv')

sa_nm3_30_1000_080_best.columns = ['index','solution']
df_sa_nm3_1000_080.columns = ['index','solution','seed_iteration']

#Grouped scatter plot
#Alpha=0.98
plt.scatter(x=df_sa_nm3_1000_098["seed_iteration"], y=df_sa_nm3_1000_098["solution"], alpha=0.5, color='blue')
plt.scatter(x=sa_nm3_30_1000_098_best["index"], y=sa_nm3_30_1000_098_best["solution"], alpha=0.5, color='red')
plt.title('SA with NM3 with alpha=0.98')
plt.xlabel('iteration')
plt.ylabel('WRW')
plt.show()
#Alpha=0.95
plt.scatter(x=df_sa_nm3_1000_095["seed_iteration"], y=df_sa_nm3_1000_095["solution"], alpha=0.5, color='blue')
plt.scatter(x=sa_nm3_30_1000_095_best["index"], y=sa_nm3_30_1000_095_best["solution"], alpha=0.5, color='red')
plt.title('SA with NM3 with alpha=0.95')
plt.xlabel('iteration')
plt.ylabel('WRW')
plt.show()
#Alpha=0.90
plt.scatter(x=df_sa_nm3_1000_090["seed_iteration"], y=df_sa_nm3_1000_090["solution"], alpha=0.5, color='blue')
plt.scatter(x=sa_nm3_30_1000_090_best["index"], y=sa_nm3_30_1000_090_best["solution"], alpha=0.5, color='red')
plt.title('SA with NM3 with alpha=0.90')
plt.xlabel('iteration')
plt.ylabel('WRW')
plt.show()
#Alpha=0.80
plt.scatter(x=df_sa_nm3_1000_080["seed_iteration"], y=df_sa_nm3_1000_080["solution"], alpha=0.5, color='blue')
plt.scatter(x=sa_nm3_30_1000_080_best["index"], y=sa_nm3_30_1000_080_best["solution"], alpha=0.5, color='red')
plt.title('SA with NM3 with alpha=0.80')
plt.xlabel('iteration')
plt.ylabel('WRW')
plt.show()
#Alpha=0.50
plt.scatter(x=df_sa_nm3_1000_050["seed_iteration"], y=df_sa_nm3_1000_050["solution"], alpha=0.5, color='blue')
plt.scatter(x=sa_nm3_30_1000_050_best["index"], y=sa_nm3_30_1000_050_best["solution"], alpha=0.5, color='red')
plt.title('SA with NM3 with alpha=0.50')
plt.xlabel('iteration')
plt.ylabel('WRW')
plt.show()

#SA NM3 Data: Sample Size 1000
sa_nm3_summary = pd.read_csv('sa_nm3_summary.csv')
def plotn10():
    # set Data
    Min = sa_nm3_summary["min"]
    Max = sa_nm3_summary["max"]
    Average = sa_nm3_summary["mean"]
    Std = sa_nm3_summary["std"]
    # Set position of bar on X axis
    r1 = np.arange(len(Min))
    r2 = [x + 0.2 for x in r1]
    r3 = [x + 0.2 for x in r2]
    r4 = [x + 0.2 for x in r3]
    # Make the plot
    plt.bar(r1, Min, color='#55DDE0', width=0.2, edgecolor='white', label='Min')
    plt.bar(r2, Max, color='#33658A', width=0.2, edgecolor='white', label='Max')
    plt.bar(r3, Average, color='#F26419', width=0.2, edgecolor='white', label='Average')
    plt.bar(r4, Std, color='#F6AE2D', width=0.2, edgecolor='white', label='Std')
    # Add xticks on the middle of the group bars
    plt.title('SA with NM3 with different alpha')
    plt.xlabel('Sample Size (N)', fontweight='bold')
    plt.xticks([r + 0.2 for r in range(len(Min))], ['Alpha=0.98', 'Alpha=0.95', 'Alpha=0.90', 'Alpha=0.80', 'Alpha=0.50'])
    # Create legend & Show graphic
    plt.legend()
    plt.show()

#Comparing Different Alpha Performance: Time
def plot_time():
    # Data
    df_time=pd.DataFrame({'x': ['Alpha=0.98', 'Alpha=0.95', 'Alpha=0.90', 'Alpha=0.80', 'Alpha=0.50'], 
                          'y1': [16.59, 26.13, 16.56, 20.21, 45.94] })
    # multiple line plot
    plt.plot('x', 'y1', data=df_time, marker='', color='#55DDE0', linewidth=2, label="SA with NM3")
    plt.legend()
    plt.title('Effect of Temperature Decay on Running Time of SA with NM3')
    plt.ylabel('Time (s)', fontweight='bold')
    plt.show()
plot_time()
# seed starts at 42
def plot_time():
    # Data
    df_time=pd.DataFrame({'x': ['Alpha=0.98', 'Alpha=0.95', 'Alpha=0.90', 'Alpha=0.80', 'Alpha=0.50'], 
                          'y1': [48.34, 48.78, 48.97, 49.02, 47.18] })
    # multiple line plot
    plt.plot('x', 'y1', data=df_time, marker='', color='#55DDE0', linewidth=2, label="SA with NM3")
    plt.legend()
    plt.title('Effect of Temperature Decay on Running Time of SA with NM3')
    plt.ylabel('Time (s)', fontweight='bold')
    plt.show()
plot_time()
# seed starts at 42
sa_nm3_10_alpha501 = pd.read_csv('sa_nm3_10_alpha501.csv')
sa_nm3_10_alpha801 = pd.read_csv('sa_nm3_10_alpha801.csv')
sa_nm3_10_alpha901 = pd.read_csv('sa_nm3_10_alpha901.csv')
sa_nm3_10_alpha951 = pd.read_csv('sa_nm3_10_alpha951.csv')
sa_nm3_10_alpha981 = pd.read_csv('sa_nm3_10_alpha981.csv')
sa_nm3_10_alpha501.columns = ['index','solution']
sa_nm3_10_alpha801.columns = ['index','solution']
sa_nm3_10_alpha901.columns = ['index','solution']
sa_nm3_10_alpha951.columns = ['index','solution']
sa_nm3_10_alpha981.columns = ['index','solution']
def plot_iteration():
    # Data
    df_seed1=pd.DataFrame({'x': sa_nm3_10_alpha501["index"], 
                          'y1': sa_nm3_10_alpha501["solution"],
                          'y2': sa_nm3_10_alpha801["solution"], 
                          'y3': sa_nm3_10_alpha901["solution"],
                          'y4': sa_nm3_10_alpha951["solution"], 
                          'y5': sa_nm3_10_alpha981["solution"]})
    # multiple line plot
    #"#55DDE0", "#33658A", "#2F4858", "#F6AE2D", "#F26419"
    plt.plot('x', 'y1', data=df_seed1, marker='', color='#55DDE0', linewidth=2, label="alpha=0.50")
    plt.plot('x', 'y2', data=df_seed1, marker='', color='#33658A', linewidth=2, label="alpha=0.80")
    plt.plot('x', 'y3', data=df_seed1, marker='', color='#2F4858', linewidth=2, label="alpha=0.90")
    plt.plot('x', 'y4', data=df_seed1, marker='', color='#F6AE2D', linewidth=2, label="alpha=0.95")
    plt.plot('x', 'y5', data=df_seed1, marker='', color='#F26419', linewidth=2, label="alpha=0.98")
    plt.legend()
    plt.title('Effect of Temperature Decay on SA with NM3')
    plt.ylabel('WRW', fontweight='bold')
    plt.xlabel('iteration', fontweight='bold')
    plt.show()
plot_iteration()

"""
alpha=0.98
(Time: 48.34 seconds)
count  3.000000e+01
mean   4.580324e+09
std    1.364279e+08
min    4.377049e+09
25%    4.491535e+09
50%    4.544295e+09
75%    4.651030e+09
max    4.891590e+09

alpha=0.95
(Time: 48.78 seconds)
count  3.000000e+01
mean   4.589206e+09
std    1.195482e+08
min    4.404317e+09
25%    4.499934e+09
50%    4.573279e+09
75%    4.673850e+09
max    4.812404e+09

alpha=0.90
(Time: 48.97 seconds)
count  3.000000e+01
mean   4.589780e+09
std    1.288445e+08
min    4.379779e+09
25%    4.496997e+09
50%    4.587114e+09
75%    4.666228e+09
max    4.807508e+09

alpha=0.80
(Time: 49.02 seconds)
count  3.000000e+01
mean   4.571097e+09
std    1.183386e+08
min    4.389577e+09
25%    4.505388e+09
50%    4.560058e+09
75%    4.630512e+09
max    4.855872e+09

alpha=0.50
(Time: 47.18 seconds)
count  3.000000e+01
mean   4.492231e+09
std    9.910743e+07
min    4.322918e+09
25%    4.422792e+09
50%    4.487407e+09
75%    4.538149e+09
max    4.727024e+09
"""
#data
df_sa_nm3_1000_098_300k = pd.read_csv('df_sa_nm3_1000_098_300k.csv')
df_sa_nm3_30_1000_098_best_300k = pd.read_csv('df_sa_nm3_30_1000_098_best_300k.csv')
df_sa_nm3_1000_095_300k = pd.read_csv('df_sa_nm3_1000_095_300k.csv')
df_sa_nm3_30_1000_095_best_300k = pd.read_csv('df_sa_nm3_30_1000_095_best_300k.csv')
df_sa_nm3_1000_090_300k = pd.read_csv('df_sa_nm3_1000_090_300k.csv')
df_sa_nm3_30_1000_090_best_300k = pd.read_csv('df_sa_nm3_30_1000_090_best_300k.csv')
df_sa_nm3_1000_080_300k = pd.read_csv('df_sa_nm3_1000_080_300k.csv')
df_sa_nm3_30_1000_080_best_300k = pd.read_csv('df_sa_nm3_30_1000_080_best_300k.csv')
df_sa_nm3_1000_050_300k = pd.read_csv('df_sa_nm3_1000_050_300k.csv')
df_sa_nm3_30_1000_050_best_300k = pd.read_csv('df_sa_nm3_30_1000_050_best_300k.csv')
#summary all alphas
df_sa_nm3_1000_098_300k.describe()
df_sa_nm3_1000_095_300k.describe()
df_sa_nm3_1000_090_300k.describe()
df_sa_nm3_1000_080_300k.describe()
df_sa_nm3_1000_050_300k.describe()
"""
Final Summary of SA with NM3 

alpha=0.98
count	300000
mean	4275721000
std	109472500
min	3754759000
25%	4196532000
50%	4281345000
75%	4348631000
max	4541491000

alpha=0.95
count	300000
mean	4256115000
std	96464230
min	3828135000
25%	4183084000
50%	4265912000
75%	4323333000
max	4560089000

alpha=0.90
count	300000
mean	4256660000
std	79545840
min	3803273000
25%	4214972000
50%	4265971000
75%	4312347000
max	4523813000

alpha=0.80
count	300000
mean	4280277000
std	95175840
min	3871397000
25%	4210256000
50%	4275817000
75%	4365955000
max	4539347000

alpha=0.50
count	300000
mean	4176513000
std	103906400
min	3983866000
25%	4118903000
50%	4168793000
75%	4230129000
max	4408554000
"""

#Redoing all alphas of NM3 with 30*10N iteration
def scatter_nm3_098():
    plt.scatter(x=df_sa_nm3_1000_098_300k["seed_iteration"], y=df_sa_nm3_1000_098_300k["solution"], alpha=0.5, color='blue', label="solutions")
    plt.scatter(x=df_sa_nm3_30_1000_098_best_300k["seed_iteration"], y=df_sa_nm3_30_1000_098_best_300k["solution"], alpha=0.5, color='red', label="best WRW")
    plt.title('SA with NM3 with alpha=0.98')
    plt.xlabel('seed iteration')
    plt.ylabel('WRW')
    plt.legend()
    plt.show()

def scatter_nm3_095():
    plt.scatter(x=df_sa_nm3_1000_095_300k["seed_iteration"], y=df_sa_nm3_1000_095_300k["solution"], alpha=0.5, color='blue', label="solutions")
    plt.scatter(x=df_sa_nm3_30_1000_095_best_300k["seed_iteration"], y=df_sa_nm3_30_1000_095_best_300k["solution"], alpha=0.5, color='red', label="best WRW")
    plt.title('SA with NM3 with alpha=0.95')
    plt.xlabel('seed iteration')
    plt.ylabel('WRW')
    plt.legend()
    plt.show()

def scatter_nm3_090():
    plt.scatter(x=df_sa_nm3_1000_090_300k["seed_iteration"], y=df_sa_nm3_1000_090_300k["solution"], alpha=0.5, color='blue', label="solutions")
    plt.scatter(x=df_sa_nm3_30_1000_090_best_300k["seed_iteration"], y=df_sa_nm3_30_1000_090_best_300k["solution"], alpha=0.5, color='red', label="best WRW")
    plt.title('SA with NM3 with alpha=0.90')
    plt.xlabel('seed iteration')
    plt.ylabel('WRW')
    plt.legend()
    plt.show()

def scatter_nm3_080():
    plt.scatter(x=df_sa_nm3_1000_080_300k["seed_iteration"], y=df_sa_nm3_1000_080_300k["solution"], alpha=0.5, color='blue', label="solutions")
    plt.scatter(x=df_sa_nm3_30_1000_080_best_300k["seed_iteration"], y=df_sa_nm3_30_1000_080_best_300k["solution"], alpha=0.5, color='red', label="best WRW")
    plt.title('SA with NM3 with alpha=0.80')
    plt.xlabel('seed iteration')
    plt.ylabel('WRW')
    plt.legend()
    plt.show()

def scatter_nm3_050():
    plt.scatter(x=df_sa_nm3_1000_050_300k["seed_iteration"], y=df_sa_nm3_1000_050_300k["solution"], alpha=0.5, color='blue', label="solutions")
    plt.scatter(x=df_sa_nm3_30_1000_050_best_300k["seed_iteration"], y=df_sa_nm3_30_1000_050_best_300k["solution"], alpha=0.5, color='red', label="best WRW")
    plt.title('SA with NM3 with alpha=0.50')
    plt.xlabel('seed iteration')
    plt.ylabel('WRW')
    plt.legend()
    plt.show()

def plot_nm3_final():
    # Data
    df_nm3=pd.DataFrame({'x': df_sa_nm3_1000_095_300k["index"], 
                          'y1': df_sa_nm3_1000_050_300k.loc[df_sa_nm3_1000_050_300k['seed_iteration'].isin(['1'])]["solution"],
                          'y2': df_sa_nm3_1000_080_300k.loc[df_sa_nm3_1000_080_300k['seed_iteration'].isin(['1'])]["solution"], 
                          'y3': df_sa_nm3_1000_090_300k.loc[df_sa_nm3_1000_090_300k['seed_iteration'].isin(['1'])]["solution"],
                          'y4': df_sa_nm3_1000_095_300k.loc[df_sa_nm3_1000_095_300k['seed_iteration'].isin(['1'])]["solution"], 
                          'y5': df_sa_nm3_1000_098_300k.loc[df_sa_nm3_1000_098_300k['seed_iteration'].isin(['1'])]["solution"]})
    # multiple line plot
    #"#55DDE0", "#33658A", "#2F4858", "#F6AE2D", "#F26419"
    plt.plot('x', 'y5', data=df_nm3, marker='', color='#F26419', linewidth=1, label="alpha=0.98")
    plt.plot('x', 'y4', data=df_nm3, marker='', color='#F6AE2D', linewidth=1, label="alpha=0.95")
    plt.plot('x', 'y3', data=df_nm3, marker='', color='#2F4858', linewidth=1, label="alpha=0.90")
    plt.plot('x', 'y2', data=df_nm3, marker='', color='#55DDE0', linewidth=1, label="alpha=0.80")
    plt.plot('x', 'y1', data=df_nm3, marker='', color='#33658A', linewidth=1, label="alpha=0.50")
    plt.legend()
    plt.title('Effect of Temperature Decay on SA with NM3')
    plt.ylabel('WRW', fontweight='bold')
    plt.xlabel('iteration', fontweight='bold')
    plt.show()
plot_nm3_final()

def scatter_nm3_onebyone():
    #"#55DDE0", "#33658A", "#2F4858", "#F6AE2D", "#F26419"
    plt.scatter(x=df_sa_nm3_1000_098_300k.loc[df_sa_nm3_1000_098_300k['seed_iteration'].isin(['1'])]["index"], y=df_sa_nm3_1000_098_300k.loc[df_sa_nm3_1000_098_300k['seed_iteration'].isin(['1'])]["solution"], alpha=0.5, color='#F26419', label="alpha=0.98", s=1)
    plt.scatter(x=df_sa_nm3_1000_095_300k.loc[df_sa_nm3_1000_095_300k['seed_iteration'].isin(['1'])]["index"], y=df_sa_nm3_1000_095_300k.loc[df_sa_nm3_1000_095_300k['seed_iteration'].isin(['1'])]["solution"], alpha=0.5, color='#F6AE2D', label="alpha=0.95", s=1)
    plt.scatter(x=df_sa_nm3_1000_090_300k.loc[df_sa_nm3_1000_090_300k['seed_iteration'].isin(['1'])]["index"], y=df_sa_nm3_1000_090_300k.loc[df_sa_nm3_1000_090_300k['seed_iteration'].isin(['1'])]["solution"], alpha=0.5, color='#2F4858', label="alpha=0.90", s=1)
    plt.scatter(x=df_sa_nm3_1000_080_300k.loc[df_sa_nm3_1000_080_300k['seed_iteration'].isin(['1'])]["index"], y=df_sa_nm3_1000_080_300k.loc[df_sa_nm3_1000_080_300k['seed_iteration'].isin(['1'])]["solution"], alpha=0.5, color='#55DDE0', label="alpha=0.80", s=1)
    plt.scatter(x=df_sa_nm3_1000_050_300k.loc[df_sa_nm3_1000_050_300k['seed_iteration'].isin(['1'])]["index"], y=df_sa_nm3_1000_050_300k.loc[df_sa_nm3_1000_050_300k['seed_iteration'].isin(['1'])]["solution"], alpha=0.5, color='#33658A', label="alpha=0.50", s=1)
    plt.legend()
    plt.title('Effect of Temperature Decay on SA with NM3')
    plt.ylabel('WRW', fontweight='bold')
    plt.xlabel('iteration', fontweight='bold')
    #plt.ylim(4.1e+09, 4.13e+09)
    plt.show()

#Barplot of summary of NM3
summary_nm3_final = pd.read_csv('summary_nm3_final.csv')
def plot_summary_nm3():
    # set Data
    Min = summary_nm3_final["min"]
    Max = summary_nm3_final["max"]
    Average = summary_nm3_final["mean"]
    Std = summary_nm3_final["std"]
    # Set position of bar on X axis
    r1 = np.arange(len(Min))
    r2 = [x + 0.2 for x in r1]
    r3 = [x + 0.2 for x in r2]
    r4 = [x + 0.2 for x in r3]
    # Make the plot
    plt.bar(r1, Min, color='#55DDE0', width=0.2, edgecolor='white', label='Min')
    plt.bar(r2, Max, color='#33658A', width=0.2, edgecolor='white', label='Max')
    plt.bar(r3, Average, color='#F26419', width=0.2, edgecolor='white', label='Average')
    plt.bar(r4, Std, color='#F6AE2D', width=0.2, edgecolor='white', label='Std')
    # Add xticks on the middle of the group bars
    plt.title('SA with NM3 with different alpha')
    plt.ylabel('WRW', fontweight='bold')
    plt.xlabel('Alpha', fontweight='bold')
    plt.xticks([r + 0.2 for r in range(len(Min))], ['Alpha=0.98', 'Alpha=0.95', 'Alpha=0.90', 'Alpha=0.80', 'Alpha=0.50'])
    # Create legend & Show graphic
    plt.legend(loc='center left')
    plt.show()

#Boxplot
def boxplot_nm3():
    df_sa_nm3_1000_098_300k['alpha'] = 0.98
    df_sa_nm3_1000_095_300k['alpha'] = 0.95
    df_sa_nm3_1000_090_300k['alpha'] = 0.90
    df_sa_nm3_1000_080_300k['alpha'] = 0.80
    df_sa_nm3_1000_050_300k['alpha'] = 0.50
    df_sa_nm3_final = pd.concat([df_sa_nm3_1000_098_300k,df_sa_nm3_1000_095_300k,df_sa_nm3_1000_090_300k,df_sa_nm3_1000_080_300k,df_sa_nm3_1000_050_300k])
    #Grouped boxplot
    sns.boxplot(x="alpha", y="solution", data=df_sa_nm3_final, palette="Set3").set_title('SA with NM3')
boxplot_nm3()

# ALGORITHM 3: SA WITH NM3 & NM6

#SA with NM 3+6
#Importing csv
sa_2nm_1000_1 = pd.read_csv('sa_2nm_1000_1.csv')
sa_2nm_1000_2 = pd.read_csv('sa_2nm_1000_2.csv')
sa_2nm_1000_3 = pd.read_csv('sa_2nm_1000_3.csv')
sa_2nm_1000_4 = pd.read_csv('sa_2nm_1000_4.csv')
sa_2nm_1000_5 = pd.read_csv('sa_2nm_1000_5.csv')
sa_2nm_1000_6 = pd.read_csv('sa_2nm_1000_6.csv')
sa_2nm_1000_7 = pd.read_csv('sa_2nm_1000_7.csv')
sa_2nm_1000_8 = pd.read_csv('sa_2nm_1000_8.csv')
sa_2nm_1000_9 = pd.read_csv('sa_2nm_1000_9.csv')
sa_2nm_1000_10 = pd.read_csv('sa_2nm_1000_10.csv')
sa_2nm_1000_11 = pd.read_csv('sa_2nm_1000_11.csv')
sa_2nm_1000_12 = pd.read_csv('sa_2nm_1000_12.csv')
sa_2nm_1000_13 = pd.read_csv('sa_2nm_1000_13.csv')
sa_2nm_1000_14 = pd.read_csv('sa_2nm_1000_14.csv')
sa_2nm_1000_15 = pd.read_csv('sa_2nm_1000_15.csv')
sa_2nm_1000_16 = pd.read_csv('sa_2nm_1000_16.csv')
sa_2nm_1000_17 = pd.read_csv('sa_2nm_1000_17.csv')
sa_2nm_1000_18 = pd.read_csv('sa_2nm_1000_18.csv')
sa_2nm_1000_19 = pd.read_csv('sa_2nm_1000_19.csv')
sa_2nm_1000_20 = pd.read_csv('sa_2nm_1000_20.csv')
sa_2nm_1000_21 = pd.read_csv('sa_2nm_1000_21.csv')
sa_2nm_1000_22 = pd.read_csv('sa_2nm_1000_22.csv')
sa_2nm_1000_23 = pd.read_csv('sa_2nm_1000_23.csv')
sa_2nm_1000_24 = pd.read_csv('sa_2nm_1000_24.csv')
sa_2nm_1000_25 = pd.read_csv('sa_2nm_1000_25.csv')
sa_2nm_1000_26 = pd.read_csv('sa_2nm_1000_26.csv')
sa_2nm_1000_27 = pd.read_csv('sa_2nm_1000_27.csv')
sa_2nm_1000_28 = pd.read_csv('sa_2nm_1000_28.csv')
sa_2nm_1000_29 = pd.read_csv('sa_2nm_1000_29.csv')
sa_2nm_1000_30 = pd.read_csv('sa_2nm_1000_30.csv')
#Preparing data
sa_2nm_1000_1['seed_iteration'] = 1
sa_2nm_1000_2['seed_iteration'] = 2
sa_2nm_1000_3['seed_iteration'] = 3
sa_2nm_1000_4['seed_iteration'] = 4
sa_2nm_1000_5['seed_iteration'] = 5
sa_2nm_1000_6['seed_iteration'] = 6
sa_2nm_1000_7['seed_iteration'] = 7
sa_2nm_1000_8['seed_iteration'] = 8
sa_2nm_1000_9['seed_iteration'] = 9
sa_2nm_1000_10['seed_iteration'] = 10
sa_2nm_1000_11['seed_iteration'] = 11
sa_2nm_1000_12['seed_iteration'] = 12
sa_2nm_1000_13['seed_iteration'] = 13
sa_2nm_1000_14['seed_iteration'] = 14
sa_2nm_1000_15['seed_iteration'] = 15
sa_2nm_1000_16['seed_iteration'] = 16
sa_2nm_1000_17['seed_iteration'] = 17
sa_2nm_1000_18['seed_iteration'] = 18
sa_2nm_1000_19['seed_iteration'] = 19
sa_2nm_1000_20['seed_iteration'] = 20
sa_2nm_1000_21['seed_iteration'] = 21
sa_2nm_1000_22['seed_iteration'] = 22
sa_2nm_1000_23['seed_iteration'] = 23
sa_2nm_1000_24['seed_iteration'] = 24
sa_2nm_1000_25['seed_iteration'] = 25
sa_2nm_1000_26['seed_iteration'] = 26
sa_2nm_1000_27['seed_iteration'] = 27
sa_2nm_1000_28['seed_iteration'] = 28
sa_2nm_1000_29['seed_iteration'] = 29
sa_2nm_1000_30['seed_iteration'] = 30
df_sa_nm6 = pd.concat([sa_2nm_1000_1,sa_2nm_1000_2,sa_2nm_1000_3,sa_2nm_1000_4,sa_2nm_1000_5,sa_2nm_1000_6,sa_2nm_1000_7,sa_2nm_1000_8,sa_2nm_1000_9,sa_2nm_1000_10,
                       sa_2nm_1000_11,sa_2nm_1000_12,sa_2nm_1000_13,sa_2nm_1000_14,sa_2nm_1000_15,sa_2nm_1000_16,sa_2nm_1000_17,sa_2nm_1000_18,sa_2nm_1000_19,sa_2nm_1000_20,
                       sa_2nm_1000_21,sa_2nm_1000_22,sa_2nm_1000_23,sa_2nm_1000_24,sa_2nm_1000_25,sa_2nm_1000_26,sa_2nm_1000_27,sa_2nm_1000_28,sa_2nm_1000_29,sa_2nm_1000_30])
df_sa_nm6.columns = ['index','solution','seed_iteration']
df_sa_nm6_1000_300k = df_sa_nm6
df_sa_nm6_1000_300k.to_csv("df_sa_nm6_1000_300k.csv")

def scatter_nm6():
    df_sa_nm6_1000_300k = pd.read_csv('df_sa_nm6_1000_300k.csv')
    df_sa_nm6_1000_best = pd.read_csv('df_sa_nm6_1000_best.csv')
    plt.scatter(x=df_sa_nm6_1000_300k["seed_iteration"], y=df_sa_nm6_1000_300k["solution"], alpha=0.5, color='blue', label="solutions")
    plt.scatter(x=df_sa_nm6_1000_best["seed_iteration"], y=df_sa_nm6_1000_best["solution"], alpha=0.5, color='red', label="best WRW")
    plt.title('SA with NM3+6 with alpha=0.95')
    plt.xlabel('seed iteration')
    plt.ylabel('WRW')
    plt.legend()
    plt.show()
    
df_sa_nm6_1000_300k.describe()
"""
NM3&6
count	3.00E+04
mean	3.53E+09
std	2.25E+08
min	3.03E+09
25%	3.36E+09
50%	3.51E+09
75%	3.68E+09
max	4.29E+09
"""
sa_2nm_1000_alpha50 = pd.read_csv('sa_2nm_1000_alpha50.csv')
sa_2nm_1000_alpha80 = pd.read_csv('sa_2nm_1000_alpha80.csv')
sa_2nm_1000_alpha90 = pd.read_csv('sa_2nm_1000_alpha90.csv')
sa_2nm_1000_alpha95 = pd.read_csv('sa_2nm_1000_alpha95.csv')
sa_2nm_1000_alpha98 = pd.read_csv('sa_2nm_1000_alpha98.csv')
sa_2nm_1000_alpha50.columns = ['index','solution']
sa_2nm_1000_alpha80.columns = ['index','solution']
sa_2nm_1000_alpha90.columns = ['index','solution']
sa_2nm_1000_alpha95.columns = ['index','solution']
sa_2nm_1000_alpha98.columns = ['index','solution']

def plot_2nm_final():
    # Data
    df_nm6=pd.DataFrame({'x': sa_2nm_1000_alpha50["index"], 
                          'y1': sa_2nm_1000_alpha50["solution"],
                          'y2': sa_2nm_1000_alpha80["solution"], 
                          'y3': sa_2nm_1000_alpha90["solution"],
                          'y4': sa_2nm_1000_alpha95["solution"], 
                          'y5': sa_2nm_1000_alpha98["solution"]})
    # multiple line plot
    #"#55DDE0", "#33658A", "#2F4858", "#F6AE2D", "#F26419"
    plt.plot('x', 'y1', data=df_nm6, marker='', color='#55DDE0', linewidth=1, label="alpha=0.50")
    plt.plot('x', 'y2', data=df_nm6, marker='', color='#33658A', linewidth=1, label="alpha=0.80")
    plt.plot('x', 'y3', data=df_nm6, marker='', color='#2F4858', linewidth=1, label="alpha=0.90")
    plt.plot('x', 'y4', data=df_nm6, marker='', color='#F6AE2D', linewidth=1, label="alpha=0.95")
    plt.plot('x', 'y5', data=df_nm6, marker='', color='#F26419', linewidth=1, label="alpha=0.98")
    plt.legend()
    plt.title('Effect of Temperature Decay on SA with both NM3 & NM6')
    plt.ylabel('WRW', fontweight='bold')
    plt.xlabel('iteration', fontweight='bold')
    plt.show()
plot_2nm_final()

#Boxplot NM6
def boxplot_nm6():
    sa_2nm_1000_alpha98['alpha'] = 0.98
    sa_2nm_1000_alpha95['alpha'] = 0.95
    sa_2nm_1000_alpha90['alpha'] = 0.90
    sa_2nm_1000_alpha80['alpha'] = 0.80
    sa_2nm_1000_alpha50['alpha'] = 0.50
    df_sa_all_nm6 = pd.concat([sa_2nm_1000_alpha98, sa_2nm_1000_alpha95, sa_2nm_1000_alpha90, sa_2nm_1000_alpha80, sa_2nm_1000_alpha50])
    #Grouped boxplot
    sns.boxplot(x="alpha", y="solution", data=df_sa_all_nm6, palette="Set3").set_title('SA with NM3 & NM6')
boxplot_nm6()

"""
Time for NM3
alpha 0.98 (Time: 31905.73 seconds)
alpha 0.95 (Time: 23968.51 seconds)
alpha 0.90 (Time: 14438.93 seconds)
alpha 0.80 (Time:  9361.93 seconds)
alpha 0.50 (Time:  7413.55 seconds)

Time for NM3 and NM6
1000 iterations, 1 run(seed=16)
alpha = 0.5(Time: 740.82 seconds)
alpha = 0.8(Time: 880.23 seconds)
alpha = 0.9(Time: 562.57 seconds)
alpha = 0.95(Time: 489.96 seconds)
alpha = 0.98(Time: 417.72 seconds)
"""
def plot_time_final():
    # Data
    df_time=pd.DataFrame({'x': ['Alpha=0.98', 'Alpha=0.95', 'Alpha=0.90', 'Alpha=0.80', 'Alpha=0.50'], 
                          'y1': [31905.73, 23968.51, 14438.93, 9361.93, 7413.55],
                          'y2': [417.72, 489.96, 562.57, 880.23, 740.82] })
    # multiple line plot
    plt.plot('x', 'y1', data=df_time, marker='', color='#55DDE0', linewidth=2, label="SA with NM3")
    plt.plot('x', 'y2', data=df_time, marker='', color='#33658A', linewidth=2, label="SA with NM3 & NM6")
    plt.legend()
    plt.title('Effect of Temperature Decay on Running Time of SA')
    plt.ylabel('Time (s)', fontweight='bold')
    plt.show()
plot_time_final()

#Boxplot Final
def final_boxplot():
    df_sa_nm3_1000_095_300k['algorithm'] = "SA_NM3"
    df_sa_nm6_1000_300k['alpha'] = 0.95
    df_sa_nm6_1000_300k['algorithm'] = "SA_NM3&6"
    box_nm3 = df_sa_nm3_1000_095_300k
    box_nm6 = df_sa_nm6_1000_300k
    box_nm6 = box_nm6.drop('Unnamed: 0', 1)
    box_random = sol_1000
    box_random.columns = ['index','solution','sample_size']
    box_random = box_random.drop('sample_size', 1)
    box_random['seed_iteration'] = 1
    box_random['alpha'] = 1
    box_random['algorithm'] = "Random_Search"
    df_sa_all_final = pd.concat([box_nm3, box_nm6, box_random])
    #Grouped boxplot
    sns.boxplot(x="algorithm", y="solution", data=df_sa_all_final, palette="Set3").set_title('Algorithm Comparison')
final_boxplot()

#end
