**Santa's Stolen Sleigh**

The North Pole is in an uproar over news that Santa's magic sleigh has been stolen. Able to carry all the world's presents in one trip, it was considered crucial to successfully delivering holiday goodies across the globe in one night. Unwilling to cancel Christmas, Santa is determined to deliver toys to all the good girls and boys using his day-to-day, magic-less sleigh. With so little time to pull off this plan, Santa is counting on you and your team to help.

Given the sleigh's antiquated, weight-limited specifications, your challenge is to optimize the routes and loads Santa will take to and from the North Pole. And don't forget about Dasher, Dancer, Prancer, and Vixen; Santa is adamant that the best solutions will minimize the toll of this hectic night on his reindeer friends.

Your goal is to minimize total weighted reindeer weariness (objective function):

![image](https://user-images.githubusercontent.com/76210176/160646275-4b1d92aa-42b1-46f1-ae87-c4878e8b5305.png)

where m is the number of trips, n is the number of gifts for each trip j, w_ij is the weight of i^th gift at trip j, Dist() is calculated with Haversine Distance between two locations, and 〖Loc〗_i is the location of gift i. 〖Loc〗_0 and 〖Loc〗_n are North Pole, and w_nj, a.k.a. the last leg of each trip, is always the base weight of the sleigh. 

For example, if you have 2 gifts A, B to deliver, then the WRW is calculated as:

    Dist(North pole to A)  ×(weight A+weight B+base_weight)+ Dist(A to B)×(weight B+ base_weight)+ Dist(B to North pole)×(base_weight)

You are given a list of gifts with their destinations and their weights. You will plan sleigh trips to deliver all the gifts to their destinations while optimizing the routes. 

All sleighs start from North Pole, then head to each gift in the order that you assign, and then head back to North Pole. 

You may have an unlimited number of sleigh trips.

All the gifts must be traveling with the sleigh at all times until the sleigh delivers it to the destination. A gift may not be dropped off anywhere before it's delivered. 

Sleighs have a base weight of 10, and a maximum weight capacity of 1000 (excluding the sleigh). 

All trips are flying trips, which means you don't need to travel via land. Haversine is used in calculating distance.  

File descriptions

	gifts.csv : Dataset with a list of all gifts that need to be delivered
	GiftId :  the id of the gift
	Latitude/Longitude : the destination of the gift
	Weight : the weight of the gift (unit-less)
	sample_solution.csv : An example solution
	GiftId : the id of the gift, ordered by the order of delivery
	TripId : the id of the trip (should be integer)
	evaluation.py : Example code showing how to evaluate the objective function 

1. Create three smaller datasets by random sampling from the file “gifts.csv”. The number of gifts of each dataset (problem size) shall be 10, 100, and 1000 gifts. 

2. To get a feeling for the problem, you decide to test a simple algorithm (Random Search) and execute it for a number of solution evaluations.
    - Implement the problem and the algorithm in Python (see below for a pseudocode of the algorithm).
    - Apply the algorithm to each of the three datasets using a maximum of 1000×N solution evaluations for each run, where N is the problem size. Each solution evaluation (Line 3 in the Algorithm below) involves computing the value of WRW  (total weighted reindeer weariness) for one candidate solution using the equations provided above.
    - Repeat each run with a different initial random seed 30 times and collect the WRW value of the best solution returned by each run (30 values in total).
    - Report in a table minimum, maximum, mean, and standard deviation of the total weighted reindeer weariness for each dataset.
    - Discuss what can be observed from the results. 
	
  ![image](https://user-images.githubusercontent.com/76210176/160647170-f8983619-e9b0-4791-b98b-3b0cc381fb2d.png)

3. In the hope of obtaining a better result, you found a more advanced heuristic algorithm called Simulated Annealing (SA); see Algorithm 2 for the pseudocode. You decide to combine your SA algorithm with a neighbourhood move assigned to your group. (Use only the neighbourhood move corresponding to your group number.)
    - Implement the algorithm (SA and neighbourhood move) in Python.
    - Apply the algorithm to each of the three datasets using a maximum of 1000×N solution evaluations for each run, where N is the problem size. 
    - Repeat each run with a different initial random seed 30 times and collect the WRW value of the best solution returned by each run (30 values in total).
    - Report in a table minimum, maximum, mean, and standard deviation of the total weighted reindeer weariness for each dataset.
    - Discuss what can be observed from the results. 
	
  ![image](https://user-images.githubusercontent.com/76210176/160647389-6c3356fe-9a98-4d03-a2f1-c9e466edfb0b.png)

4. After analysing the above results, you realise that combining SA with a single neighbourhood move leads to unsatisfactory results. Consequently, you decide to add a second neighbourhood move, NM6, to your algorithm. 
	  - Think about how your algorithm can be extended with the second neighbourhood move, and then implement the algorithm. Provide a short pseudocode outlining the interaction between the two neighbourhood moves within SA. 
	  - Repeat the steps in Question 3 for the updated algorithm. 
	  - Using Python, create one figure (plot) for each input file (i.e. 3 figures in total). Each figure is a boxplot with boxes for the total weighted reindeer weariness values obtained by this algorithm, the algorithm of Question 3, and a third box for the total weighted reindeer weariness values obtained by Random Search (from Question 2). 
	  - Discuss what can be observed from the results.
	  - Given these plots, which algorithm do you think Santa should use in the future and why?

5. After analysing the above results, you realise that the temperature decay (α) is an important parameter in the above algorithm (from Question 3 or 4). You decide to examine how the performance of the algorithm changes when you change this parameter:
	  - Repeat the steps in Q3 or Q4 for each value of α={0.98,0.95,0.90,0.8,0.5}
	  - Use Python (not Excel!) to visualize the effect of varying the above parameter on the WRW values of the datasets. Include the performance of Random Search in your visualization as a reference.
	  - Discuss what can be observed from the results. 
	  - Given these plots, which value of each parameter should be used and why?

6. Calculate the time in seconds required to run each of the algorithms evaluated according to problem size. Given these values, estimate how many solution evaluations you would be able to evaluate in one hour for the original dataset “gifts.csv”. Visualize your data and estimation. Run the best algorithm found for that number of solution evaluations and report the best WRW value found. 
