import numpy
import queue
import math
import time
from Map import Map
from Search import Search


DIM = 100
PROB = 0.00

ITERATIONS = 100
#current_map = Map(DIM, PROB)
total_solutions = 0
average_solved = 0
total_time = 0
total_path_length = 0
expected_path_length = 0
average_path_length = 0
total_cells_visited = 0
average_cells_visited = 0
"""
print("--------------------------------\nImproved DFS")

for x in range(0, ITERATIONS):
	current_map = Map(DIM, PROB)
	start_time = time.time()
	current_map.results = Search(current_map).improvedDFS()
	current_time = round(time.time() - start_time, 20)
	if(current_map.results['Status'] == 'Found Path'):
		total_solutions +=1
		total_path_length += int(current_map.results['Path length'])
	#if(current_map.results['Path length'] != 'N/A'):
		#total_path_length += int(current_map.results['Path length'])
	total_time += current_time
	
	print("Time: ", current_time)
	
average_solved = round(total_solutions / ITERATIONS,7)
if(total_solutions != 0):
	average_path_length = round(total_path_length/total_solutions, 4)


print('p:' + str(PROB))
print('Total Solutions:' + str(total_solutions))
print('Total Time:' + str(total_time))
print('Average Solved:' + str(average_solved))
print('Total Path Length:' + str(total_path_length))
print('Average Path Length:' + str(average_path_length))
"""

print("--------------------------------\nA* Euclidean")

for x in range(0, ITERATIONS):
	current_map = Map(DIM, PROB)
	start_time = time.time()
	current_map.results = Search(current_map).A_star("euclidean")
	current_time = round(time.time() - start_time, 20)
	if(current_map.results['Status'] == 'Found Path'):
		total_solutions +=1
		total_cells_visited += int(current_map.results['# of Visited Cells'])
		
	total_time += current_time
	#if(current_map.results['# of Visited Cells'] != 'n/a'):
		#total_cells_visited += int(current_map.results['# of Visited Cells'])
	
	print("Time: ", current_time)
	
average_solved = round(total_solutions / ITERATIONS,7)
average_cells_visited = round(total_cells_visited/total_solutions,4)


print('p:' + str(PROB))
print('Total Solutions:' + str(total_solutions))
print('Total Time:' + str(total_time))
print('Average Solved:' + str(average_solved))
print('Total cells Visited: ' + str(total_cells_visited))
print('Average Cells Visited: ' + str(average_cells_visited))


"""
print("--------------------------------\nUsing BFS")

for x in range(0, ITERATIONS):
	current_map = Map(DIM, PROB)
	start_time = time.time()
	current_map.results = Search(current_map).bfsAlgo()
	current_time = round(time.time() - start_time, 20)
	if(current_map.results['Status'] == 'Found Path'):
		total_solutions +=1
		
	total_time += current_time
	if(current_map.results['Path length'] != 'N/A'):
		total_path_length += int(current_map.results['Path length'])
	
	print("Time: ", current_time)
	
average_solved = round(total_solutions / ITERATIONS,7)
expected_path_length = round(total_path_length/total_solutions, 4) 

print('p:' + str(PROB))
print('Total Solutions:' + str(total_solutions))
print('Total Time:' + str(total_time))
print('Average Solved:' + str(average_solved))
print('Total Path Length' + str(total_path_length))
print('Expected Path Length:' + str(expected_path_length))

"""
"""
print("--------------------------------\nUsing DFS")

for x in range(0, ITERATIONS):
	current_map = Map(DIM, PROB)
	start_time = time.time()
	current_map.results = Search(current_map).dfsAlgo()
	current_time = round(time.time() - start_time, 20)
	if(current_map.results['Status'] == 'Found Path'):
		total_solutions +=1
	total_time += current_time
	print("Time: ", current_time)
	
average_solved = round(total_solutions / ITERATIONS,7) 

print('Total Solutions:' + str(total_solutions))
print('Total Time:' + str(total_time))
print('Average Solved:' + str(average_solved))
"""
"""
print("--------------------------------\nUsing DFS")

for x in range(0, ITERATIONS):
	current_map = Map(DIM, PROB)
	start_time = time.time()
	current_map.results = Search(current_map).dfs_algo()
	current_time = round(time.time() - start_time, 20)
	if(current_map.results['Status'] == 'Found Path'):
		total_solutions +=1
	total_time += current_time
	print("Time: ", current_time)
	
average_solved = (total_solutions / ITERATIONS) * 100

print('Total Solutions:' + str(total_solutions))
print('Total Time:' + str(total_time))
print('Average Solved:' + str(average_solved))
"""
"""
print("--------------------------------\nUsing BFS Algo")

for x in range(0, ITERATIONS):
	current_map = Map(DIM, PROB)
	start_time = time.time()
	current_map.results = Search(current_map).bfs_algo()
	current_time = round(time.time() - start_time, 20)
	if(current_map.results['Status'] == 'Found Path'):
		total_solutions +=1
	total_time += current_time
	print("Time: ", current_time)
	
average_solved = (total_solutions / ITERATIONS) * 100

print('Total Solutions:' + str(total_solutions))
print('Total Time:' + str(total_time))
print('Average Solved:' + str(average_solved))

"""
"""
print("--------------------------------\nUsing BIBFS Algo")
for x in range(0, ITERATIONS):
	current_map = Map(DIM, PROB)
	start_time = time.time()
	current_map.results = Search(current_map).biBFS()
	current_time = round(time.time() - start_time, 20)
	if(current_map.results['Status'] == 'Found Path'):
		total_solutions +=1
	total_time += current_time
	print("Time: ", current_time)
	
average_solved = (total_solutions / ITERATIONS) * 100

print('Total Solutions:' + str(total_solutions))
print('Total Time:' + str(total_time))
print('Average Solved:' + str(average_solved))

"""
"""
print("--------------------------------\nUsing A* Euclidean")
for x in range(0, ITERATIONS):
	current_map = Map(DIM, PROB)
	start_time = time.time()
	current_map.results = Search(current_map).a_star_algo("euclidean")
	current_time = round(time.time() - start_time, 20)
	if(current_map.results['Status'] == 'Found Path'):
		total_solutions +=1
	total_time += current_time
	print("Time: ", current_time)
	
average_solved = (total_solutions / ITERATIONS) * 100

print('Total Solutions:' + str(total_solutions))
print('Total Time:' + str(total_time))
print('Average Solved:' + str(average_solved))
"""
"""
print("--------------------------------\nUsing A* Manhattan")
for x in range(0, ITERATIONS):
	current_map = Map(DIM, PROB)
	start_time = time.time()
	current_map.results = Search(current_map).a_star_algo("manhattan")
	current_time = round(time.time() - start_time, 20)
	if(current_map.results['Status'] == 'Found Path'):
		total_solutions +=1
	total_time += current_time
	print("Time: ", current_time)
	
average_solved = (total_solutions / ITERATIONS) * 100

print('Total Solutions:' + str(total_solutions))
print('Total Time:' + str(total_time))
print('Average Solved:' + str(average_solved))
"""