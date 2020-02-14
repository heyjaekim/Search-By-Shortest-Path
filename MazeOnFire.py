import numpy as np
import random as rand
import seaborn as sns
import math
import queue as Q
import time
import pprint as pp
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from matplotlib.colors import ListedColormap

# Function to generate maze of given dimension. It takes 'p' as the probability of a particular cell being blocked.


def generate_maze(size, prob):

    mat = np.random.uniform(size=[size, size])
    mat = (mat > prob).astype(int)
    mat[0, 0] = 1
    mat[size - 1, size - 1] = 1
    return mat

# Function to obtain the length of a path


def path_length(prev_list, start, goal):

    prev_node = goal
    path = [prev_node]
    path_len = 0

    if bool(prev_list):
        while(prev_node in prev_list.keys() and prev_list[prev_node] != start):
            current_node = prev_node
            prev_node = prev_list[(current_node)]
            path.append(prev_node)

        path.append(prev_list[prev_node])
        path.reverse()

        path_len = len(path)

    return path_len, path

# Function to prioritize children based on the distance to the goal. This improves DFS search


def prioritize_children(maze, goal, visited, x, y):

    children = []
    node = []
    h = []

    node.append((x+1, y))
    h.append((goal - node[0][0]) + (goal - node[0][1]))

    node.append((x, y+1))
    h.append((goal - node[1][0]) + (goal - node[1][1]))

    node.append((x-1, y))
    h.append((goal - node[2][0]) + (goal - node[2][1]))

    node.append((x, y-1))
    h.append((goal - node[3][0]) + (goal - node[3][1]))

    for i in range(4):
        ind = h.index(min(h))
        h.pop(ind)
        current_child = node.pop(ind)
        if(validPath(maze, current_child, visited)):
            children.append(current_child)

    children.reverse()
    return children

# Improved function of Depth-First Search


def improved_DFS(maze, start, goal):

    fringe = [start]
    visited = [start]
    prev_list = {}    # To store pointers from children to their parents. It is useful for backtracking the path

    count_of_nodes = 0
    max_fringe_size = 0

    while fringe:
        (i, j) = fringe.pop()
        count_of_nodes += 1
        max_fringe_size = max(max_fringe_size, len(fringe))

        if (i, j) == goal:    # to check if the goal state is found
            return 1, prev_list, count_of_nodes, max_fringe_size

        # Generating and adding child nodes in fringe
        children = prioritize_children(
            maze, goal[0]-1, visited, i, j)  # ----> Improvement
        if children:
            for c in children:
                prev_list[c] = (i, j)
                fringe.append(c)
                visited.append(c)

    return 0, prev_list, count_of_nodes, max_fringe_size  # No path found


def findHeuristic(goalCell, cell, heuristic):
    (x1, y1) = cell
    (x2, y2) = goalCell
    if heuristic == "euclidean":
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    elif heuristic == "manhattan":
        return abs(x1 - x2) + abs(y1 - y2)


def in_bounds(cell):
    # Check if a cell is in the map
    x = cell[0]
    y = cell[1]
    return (0 <= x < size) and (0 <= y < size)


def neighborCells(cell):

    x = cell[0]
    y = cell[1]
    neighborCells = set()
    DIRS = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
    for (i, j) in DIRS:
        if in_bounds((i, j)):
            if maze[i, j] == 1:
                neighborCells.add((i, j))
    return neighborCells


def A_star(maze, startCell, goalCell, heuristic):

    dim = maze.shape[0]

    priorityQ = Q.PriorityQueue()
    prevList = {}
    mincost = {}
    cellsVisited = set()
    maxFringe = 0

    priorityQ.put((0, startCell))
    mincost[startCell] = 0
    cellsVisited.add(startCell)

    while not priorityQ.empty():
        cell = priorityQ.get()  # referring to the curernt cell
        cell = cell[1]
        maxFringe = max(maxFringe, priorityQ.qsize())

        if cell == goalCell:
            return 1, prevList, len(cellsVisited), maxFringe, cellsVisited

        for dir in neighborCells(cell):
            newCost = mincost[cell] + 1
            if dir not in cellsVisited:
                mincost[dir] = newCost
                prevList[dir] = cell
                cellsVisited.add(dir)

                priority = newCost + findHeuristic(goalCell, dir, heuristic)
                priorityQ.put((priority, dir))
    return 0, prevList, len(cellsVisited), maxFringe, cellsVisited

# Function to generate neighbors of a cell for fire.


def generate_neighbours(cell):

    neighbours = []

    i = cell[0]
    j = cell[1]

    if(i+1 < size):
        neighbours.append((i+1, j))

    if(i-1 >= 0):
        neighbours.append((i-1, j))

    if(j+1 < size):
        neighbours.append((i, j+1))

    if(j-1 >= 0):
        neighbours.append((i, j-1))

    return neighbours

# Function to check the success rate of the baseline condition


def success_rate(dim, p, n_trials):

    probability_solvable = []

    for q in range(0, 11, 1):

        q = q/10.0
        success = 0

        for i in range(n_trials):

            output = baseline(dim, p, q)

            if(output == 1):
                success = success + 1    # Count the number of succesfully solved mazes
            else:
                continue

        probability_solvable.append(success/n_trials)
        print("Success for q = " + str(q) + " is " + str(success))

    print(probability_solvable)
    x = np.arange(0, 1.1, 0.1)

    plt.clf()
    plt.cla()
    plt.close()

    plt.bar(x, probability_solvable, width=0.05)

    plt.xlabel("q")
    plt.ylabel("Probability of success")
    plt.title("Density vs solvability for dim = " +
              str(dim) + ", #trials = " + str(n_trials))
    plt.xticks(x)
    plt.show()


"""##Fire Runner"""


def validPath(maze, cell, visited):
    x = cell[0]
    y = cell[1]
    if x == -1 or y == -1 or x == size or y == size:
        return False
    elif maze[x, y] == 0 or cell in visited:
        return False
    return True


def getMinDistsFromFire(cell, manhattanDists, fireCells, cells, h):
    for f in fireCells:
        # manhattan distances
        manhattanDists.append((abs(f[0]-cell[0]) + abs(f[1]-cell[1])))
    minDist = 0
    if manhattanDists:
        minDist = min(manhattanDists)
    h.append((size - cell[0]) + (size - cell[1]) - minDist)
    cells.append(cell)
    return cells, h

# Function to prioritize children based on the distance to the goal and distance from fire.


def prioritization(maze, visited, x, y, fireCells):
    children = list()
    cells = list()
    h = list()

    manhattanDists = list()
    cell = ((x, y+1))
    if(validPath(maze, cell, visited)):
        cells, h = getMinDistsFromFire(
            cell, manhattanDists, fireCells, cells, h)

    manhattanDists = list()
    cell = ((x+1, y))
    if(validPath(maze, cell, visited)):
        cells, h = getMinDistsFromFire(
            cell, manhattanDists, fireCells, cells, h)

    manhattanDists = list()
    cell = ((x, y-1))
    if(validPath(maze, cell, visited)):
        cells, h = getMinDistsFromFire(
            cell, manhattanDists, fireCells, cells, h)

    manhattanDists = list()
    cell = ((x-1, y))
    if(validPath(maze, cell, visited)):
        cells, h = getMinDistsFromFire(
            cell, manhattanDists, fireCells, cells, h)

    for i in range(len(h)):
        val = h.index(min(h))
        h.pop(val)
        child = cells.pop(val)
        children.append(child)

    children.reverse()
    return children
# Function to implement search to stay away from all fire cells.


def fire_cell_search(maze, start, goal, fire_start, fire_goal, q):

    fringe = [start]
    visited = [start]
    prev_list = {}    # To store pointers from children to their parents. It is useful for backtracking the path

    count_of_nodes = 0
    max_fringe_size = 0

    maze[0][size-1] = -1  # Intial condition - fire on upper right corner

    fire_fringe = []
    fire_fringe.append((0, size-1))
    not_on_fire_fringe = []
    new_fire = []
    fire_cells = [(0, size-1)]

    while fringe:
        (i, j) = fringe.pop()
        #display_path(maze_temp, prev_list, start, (i,j))

        # Termination Conditions

        if goal in fire_cells:    # Goal is on fire
            return 3, (i, j), prev_list, count_of_nodes, max_fringe_size, fire_cells

        if (i, j) in fire_cells:    # Runner location is on fire
            return 2, (i, j), prev_list, count_of_nodes, max_fringe_size, fire_cells

        count_of_nodes += 1
        max_fringe_size = max(max_fringe_size, len(fringe))

        if (i, j) == goal:    # to check if the goal state is found
            return 1, goal, prev_list, count_of_nodes, max_fringe_size, fire_cells

        # Generating and adding child nodes in fringe
        children = prioritization(
            maze, visited, i, j, fire_fringe)  # ----> Improved DFS
        if children:
            for c in children:
                prev_list[c] = (i, j)
                fringe.append(c)
                visited.append(c)

        # Adding neighbors to the fringe
        if (fire_fringe):
            while fire_fringe:
                neighbours = generate_neighbours(fire_fringe.pop())

                while neighbours:
                    neighbor = neighbours.pop()
                    if maze[neighbor] != -100 and maze[neighbor] != 0 and neighbor not in not_on_fire_fringe:
                        not_on_fire_fringe.append(neighbor)

        not_on_fire_fringe_temp = not_on_fire_fringe.copy()
        while not_on_fire_fringe_temp:
            k = 0
            cell = not_on_fire_fringe_temp.pop()
            neighbours = generate_neighbours(cell)

            for v in neighbours:
                if (maze[v] == -1):
                    k = k+1

            probability = 1 - pow(1-q, k)
            if(rand.uniform(0, 1) < probability):  # If the cell catches fire
                new_fire.append(cell)
                not_on_fire_fringe.remove(cell)

        while new_fire:
            n = new_fire.pop()
            fire_fringe.append(n)
            maze[n] = -1
            fire_cells.append(n)

    return 0, (i, j), prev_list, count_of_nodes, max_fringe_size, fire_cells


# Function to visualize the obtained path

def display_path(canvas, prev_list, start, goal):

    prev_node = goal
    path = [prev_node]
    canvas[prev_node] = 25

    if bool(prev_list):
        while(prev_list[prev_node] != start):
            current_node = prev_node
            prev_node = prev_list[(current_node)]
            path.append(prev_node)
            canvas[prev_node] = 50

        path.append(prev_list[prev_node])
        path.reverse()
        canvas[prev_list[prev_node]] = 25

        path_length = str(len(path))

        plt.figure(figsize=(15, 15))
        sns.heatmap(canvas, cmap=ListedColormap(['black', 'green', 'crimson', 'blue', 'papayawhip']),
                    linewidths=.5,  square=True, cbar=False, xticklabels=False, yticklabels=False)
        plt.show()

        return path_length


start_time = time.time()
#visualise_maze(maze, prev_list_path, start, goal, visited_path)
size = 100
p = 0.2
q = 0.1
start = (0, 0)
goal = (size - 1, size - 1)
fire_start = (0, size-1)
fire_goal = (size-1, 0)

is_goal_reached = 0
is_fire_reached = 0

while not is_goal_reached or not is_fire_reached:
    maze = generate_maze(size, p)
    is_goal_reached, prev_list_path, count_of_nodes_path, max_fringe_size_path, visited_path = A_star(
        maze, start, goal, "manhattan")
    is_fire_reached, prev_list_fire, count_of_nodes_fire, max_fringe_size_fire, visited_fire = A_star(
        maze, fire_start, fire_goal, "manhattan")

is_reached, runner_location, prev_list, count_of_nodes, max_fringe_size, fire_cells = fire_cell_search(
    maze, start, goal, fire_start, fire_goal, q)

print("Time taken for strategy 1" + str(time.time()-start_time))
canvas = maze*100
display_path(canvas, prev_list_path, start, goal)


# Function to implement search to stay away from new cells on fire.

def fire_fringe_search(maze, start, goal, fire_start, fire_goal, q):

    fringe = [start]
    visited = [start]
    prev_list = {}    # To store pointers from children to their parents. It is useful for backtracking the path

    count_of_nodes = 0
    max_fringe_size = 0

    maze[0][dim-1] = -1  # Intial condition - fire on upper right corner

    fire_fringe = []
    fire_fringe.append((0, dim-1))
    not_on_fire_fringe = []
    new_fire = []
    fire_cells = [(0, dim-1)]

    while fringe:
        (i, j) = fringe.pop()
        if (fire_fringe):
            while fire_fringe:
                neighbours = generate_neighbours(fire_fringe.pop())

                while neighbours:
                    neighbor = neighbours.pop()
                    if maze[neighbor] != -1 and maze[neighbor] != 0 and neighbor not in not_on_fire_fringe:
                        not_on_fire_fringe.append(neighbor)

        not_on_fire_fringe_temp = not_on_fire_fringe.copy()
        while not_on_fire_fringe_temp:
            k = 0
            cell = not_on_fire_fringe_temp.pop()
            neighbours = generate_neighbours(cell)

            for v in neighbours:
                if (maze[v] == -1):
                    k = k+1

            probability = 1 - pow(1-q, k)
            if(rand.uniform(0, 1) < probability):  # If the cell catches fire
                new_fire.append(cell)
                not_on_fire_fringe.remove(cell)

        while new_fire:
            n = new_fire.pop()
            fire_fringe.append(n)
            maze[n] = -1
            fire_cells.append(n)

        if goal in fire_cells:
            return 3, (i, j), prev_list, count_of_nodes, max_fringe_size, fire_cells

        if (i, j) in fire_cells:
            return 2, (i, j), prev_list, count_of_nodes, max_fringe_size, fire_cells

        count_of_nodes += 1
        max_fringe_size = max(max_fringe_size, len(fringe))

        if (i, j) == goal:    # to check if the goal state is found
            return 1, goal, prev_list, count_of_nodes, max_fringe_size, fire_cells

        # Generating and adding child nodes in fringe
        children = safe_children(maze, visited, i, j)  # ----> Improvement
        if children:
            for c in children:
                prev_list[c] = (i, j)
                fringe.append(c)
                visited.append(c)

    return 0, (i, j), prev_list, count_of_nodes, max_fringe_size, fire_cells


start_time = time.time()

dim = 100
p = 0.2
q = 0.1
start = (0, 0)
goal = (dim - 1, dim - 1)
fire_start = (0, dim-1)
fire_goal = (dim-1, 0)

is_goal_reached = 0
is_fire_reached = 0

while not is_goal_reached or not is_fire_reached:
    maze = generate_maze(dim, p)
    is_goal_reached, prev_list_path, count_of_nodes_path, max_fringe_size_path, visited_path = A_star(
        maze, start, goal, "manhattan")
    is_fire_reached, prev_list_fire, count_of_nodes_fire, max_fringe_size_fire, visited_fire = A_star(
        maze, fire_start, fire_goal, "manhattan")

is_reached, runner_location, prev_list, count_of_nodes, max_fringe_size, fire_cells = fire_fringe_search(
    maze, start, goal, fire_start, fire_goal, q)

maze_temp = maze*100

print("Time taken for strategy 2: " + str(time.time()-start_time))
display_path(maze_temp, prev_list_path, start, goal)

# Function to implement search to stay from children with more neighbors on fire.

def safe_children(maze, visited, x, y):
  
    children = []
    nodes = []
    n = []
    
    node = ((x+1, y))
    neighbors = [(x+1,y+1), (x+2,y), (x+1,y-1)]
    if(validPath(maze, node, visited)):
      n.append(0)
      for i in neighbors:
        if validPath(maze, i, visited):
          n[len(n)-1] += 1
        n[len(n)-1] = (dim - 1 - node[0]) + (dim - 1 - node[1]) - n[len(n)-1]    # Priority consider Manhattan distance to the goal and the number of neighbors on fire
      nodes.append(node)
    
    node = ((x, y+1))
    neighbors = [(x+1,y+1), (x,y+2), (x-1,y+1)]
    if(validPath(maze, node, visited)):
      n.append(0)
      for i in neighbors:
        if validPath(maze, i, visited):
          n[len(n)-1] += 1
      n[len(n)-1] = (dim - 1 - node[0]) + (dim - 1 - node[1]) - n[len(n)-1]
      nodes.append(node)
    
    node = ((x-1, y))
    neighbors = [(x-1,y+1), (x-2,y), (x-1,y-1)]
    if(validPath(maze, node, visited)):
      n.append(0)
      for i in neighbors:
        if validPath(maze, i, visited):
          n[len(n)-1] += 1
      n[len(n)-1] = (dim - 1 - node[0]) + (dim - 1 - node[1]) - n[len(n)-1]
      nodes.append(node)
    
    node = ((x, y-1))
    neighbors = [(x+1,y-1), (x,y-2), (x-1,y-1)]
    if(validPath(maze, node, visited)):
      n.append(0)
      for i in neighbors:
        if validPath(maze, i, visited):
          n[len(n)-1] += 1
      n[len(n)-1] = (dim - 1 - node[0]) + (dim - 1 - node[1]) - n[len(n)-1]
      nodes.append(node)
          
    for i in range(len(n)):
        ind = n.index(min(n))
        n.pop(ind)
        current_child = nodes.pop(ind)
        children.append(current_child)
    
    
    children.reverse()
    return children

def fire_neighbor_search(maze, start, goal, fire_start, fire_goal, q):

  
  fringe = [start]
  visited = [start]
  prev_list = {}    # To store pointers from children to their parents. It is useful for backtracking the path

  count_of_nodes = 0
  max_fringe_size = 0
  
  maze[fire_start] = -1 # Intial condition - fire on upper right corner
  
  fire_fringe = []
  fire_fringe.append((0, dim-1))
  not_on_fire_fringe = []
  new_fire = []
  fire_cells = [(0, dim-1)]
  
  while fringe:
#       maze_temp = maze*100
      (i, j) = fringe.pop()      
#       display_path(maze_temp, prev_list, start, (i,j))
      
      if goal in fire_cells:
        return 3, (i,j), prev_list, count_of_nodes, max_fringe_size, fire_cells
      
      if (i,j) in fire_cells:
        return 2, (i,j), prev_list, count_of_nodes, max_fringe_size, fire_cells
      
      count_of_nodes+=1
      max_fringe_size = max( max_fringe_size, len(fringe) )

      if (i, j) == goal:    # to check if the goal state is found
          return 1, goal, prev_list, count_of_nodes, max_fringe_size, fire_cells
      
      # Generating and adding child nodes in fringe 
      children = safe_children(maze,visited,i,j)    #----> Improvement
      if children:
        for c in children:
          prev_list[c] = (i,j)
          fringe.append(c)
          visited.append(c)

      if (fire_fringe):
        while fire_fringe:
          neighbours = generate_neighbours(fire_fringe.pop())
          while neighbours:
            neighbor = neighbours.pop()
            if maze[neighbor] != -1 and maze[neighbor] != 0 and neighbor not in not_on_fire_fringe:
              not_on_fire_fringe.append(neighbor)

      not_on_fire_fringe_temp = not_on_fire_fringe.copy()
      while not_on_fire_fringe_temp :
          k = 0
          cell = not_on_fire_fringe_temp.pop()
          neighbours = generate_neighbours(cell)

          for v in neighbours:  
            if (maze[v] == -1):
              k = k+1

          probability = 1 - pow(1-q, k)
          
          if( rand.uniform(0,1) < probability):  ##If the cell catches fire
            new_fire.append(cell) 
            not_on_fire_fringe.remove(cell)
          
      while new_fire:
        n = new_fire.pop()
        fire_fringe.append(n)
        maze[n] = -1
        fire_cells.append(n)
          
  return 0, (i,j), prev_list, count_of_nodes, max_fringe_size, fire_cells

start_time = time.time()

dim = 100
p = 0.2
q = 0.2
start = (0,0)
goal = (dim - 1, dim - 1)
fire_start = (0,dim-1)
fire_goal = (dim-1,0)

is_goal_reached = 0
is_fire_reached = 0

while not is_goal_reached or not is_fire_reached:
  maze = generate_maze(dim, p)
  is_goal_reached, prev_list_path, count_of_nodes_path, max_fringe_size_path, visited_path = A_star(maze, start, goal , "manhattan")
  is_fire_reached, prev_list_fire, count_of_nodes_fire, max_fringe_size_fire, visited_fire = A_star(maze, fire_start, fire_goal, "manhattan")

is_reached, runner_location, prev_list, count_of_nodes, max_fringe_size, fire_cells = fire_neighbor_search(maze, start, goal, fire_start, fire_goal, 0.2)
maze_temp = maze*100

print("Time taken for strategy 3: " + str(time.time()-start_time))
display_path(maze_temp, prev_list_path, start, goal)