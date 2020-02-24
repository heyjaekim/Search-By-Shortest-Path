import numpy as np
import queue as Q
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def renderMaze(size, prob):

    mat = np.random.uniform(size=[size, size])
    mat = (mat > prob).astype(int)
    mat[0, 0] = 1
    mat[size - 1, size - 1] = 1
    return mat

def visualizaeMaze(maze):
    plt.figure(figsize=(15, 15))
    ax = sns.heatmap(maze, linewidths=.5, square=True,
                     cbar=False, xticklabels=False, yticklabels=False)
    plt.show()

"""###A-star function implementation"""

def findHeuristic(goalCell, cell, h):
    # h as heuristic
    (x1, y1) = cell
    (x2, y2) = goalCell
    if h == "euclidean":
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    elif h == "manhattan":
        return abs(x1 - x2) + abs(y1 - y2)

#check if it's a valid inbound
def inBound(cell):

    x = cell[0]
    y = cell[1]
    return (0 <= x < size) and (0 <= y < size)

#append the neighbor cells into the cells list
def neighborCells(maze, cell):

    x = cell[0]
    y = cell[1]
    neighbors = set()
    DIRS = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
    for (i, j) in DIRS:
        if inBound((i, j)):
            if maze[i, j] == 1:
                neighbors.add((i, j))
    return neighbors

#check if the cell is valid and also append into the list
def addPotentialFireCells(cell):

    potentialFire = []

    if(cell[0]+1 < size):
        potentialFire.append((cell[0]+1, cell[1]))
    if(cell[0]-1 >= 0):
        potentialFire.append((cell[0]-1, cell[1]))
    if(cell[1]+1 < size):
        potentialFire.append((cell[0], cell[1]+1))
    if(cell[1]-1 >= 0):
        potentialFire.append((cell[0], cell[1]-1))
    return potentialFire

"""
# A-star function to be implemented 
# parameters: maze -- currentMaze, startCell & goalCell, h -- heuristic
# return values: id_goal - 0(not reached goal) or 1, pathSet - list of path nodes, maxFringe - maximal fringe size 
# """

def A_star(maze, startCell, goalCell, h):

    priorityQ = Q.PriorityQueue()
    pathSet = {}
    mincost = {}
    cellsVisited = set()
    maxFringe = 0

    priorityQ.put((0, startCell))
    mincost[startCell] = 0
    cellsVisited.add(startCell)

    while not priorityQ.empty():
        cell = priorityQ.get() 
        cell = cell[1]
        maxFringe = max(maxFringe, priorityQ.qsize())

        if cell == goalCell:
            return 1, pathSet, len(cellsVisited)

        for dir in neighborCells(maze, cell):
            newCost = mincost[cell] + 1
            if dir not in cellsVisited:
                mincost[dir] = newCost
                pathSet[dir] = cell
                cellsVisited.add(dir)
                priority = newCost + findHeuristic(goalCell, dir, h)
                priorityQ.put((priority, dir))

    return 0, pathSet, len(cellsVisited)

def isValidCell(maze, cell, visited):

    x = cell[0]
    y = cell[1]
    #(0 <= x < self.size) and (0 <= y < self.size)
    if x == -1 or x == size or y == -1 or y == size:
        return False
    elif maze[x, y] == 0 or cell in visited:
        return False
    return True


def visualizePath(canvas, parentSet, start, goal):

    prevCell = goal
    path = [prevCell]
    canvas[prevCell] = 25

    if bool(parentSet):
        while(parentSet[prevCell] != start):
            current_node = prevCell
            prevCell = parentSet[(current_node)]
            path.append(prevCell)
            canvas[prevCell] = 50

        path.append(parentSet[prevCell])
        path.reverse()
        canvas[parentSet[prevCell]] = 25

        path_length = str(len(path))

        plt.figure(figsize=(15, 15))
        sns.heatmap(canvas, cmap=ListedColormap(['black', 'green', 'crimson', 'blue', 'papayawhip']),
                        linewidths=.5,  square=True, cbar=False, xticklabels=False, yticklabels=False)
        plt.show()

        return path_length

def pathLength(parentDict, start, goal):

    parent = goal
    path = [parent]

    if bool(parentDict):
        while(parent in parentDict.keys() and parentDict[parent] != start):
            current_node = parent
            parent = parentDict[(current_node)]
            path.append(parent)

        path.append(parentDict[parent])
        path.reverse()

    return len(path), path


'''DFS function implementation with the maximum fringe size'''

def render_path(pathDict, goal):
        # Build a path (list) from a dictionary of parent cell -> child cell
        current = pathDict[goal]
        path = []
        while current != (0, 0):
            path += (current,)
            current = pathDict[current]
        return path


def dfsAlgo(maze, start, goal):
    dfsStack = Q.LifoQueue()
    dfsStack.put(start)
    parentDict = {}
    cellsVisited = set()
    cellsVisited.add(start)
    maxFringe = 0

    while not dfsStack.empty():
        cell = dfsStack.get()
        maxFringe = max(maxFringe, dfsStack.qsize())
        if cell == goal:
            path = render_path(parentDict, goal)
            return 1, parentDict, maxFringe

        for dir in neighborCells(maze, cell):
            if dir not in cellsVisited:
                parentDict[dir] = cell
                cellsVisited.add(dir)
                dfsStack.put(dir)

    return 0, parentDict, maxFringe


'''IDFS function implementation with the maximum fringe size'''

#Priortizing the neighbor cells and exclude the invalid cells.
def prioritization(maze, goal, visited, x, y):

    neighbors = list()
    cells = list()
    path = list()

    cells.append((x+1, y))
    path.append((goal - cells[0][0]) + (goal - cells[0][1]))

    cells.append((x, y+1))
    path.append((goal - cells[1][0]) + (goal - cells[1][1]))

    cells.append((x-1, y))
    path.append((goal - cells[2][0]) + (goal - cells[2][1]))

    cells.append((x, y-1))
    path.append((goal - cells[3][0]) + (goal - cells[3][1]))

    for i in range(4):
        ind = path.index(min(path))
        path.pop(ind)
        current_child = cells.pop(ind)
        if(isValidCell(maze, current_child, visited)):
            neighbors.append(current_child)

    neighbors.reverse()
    return neighbors

def IDFS(maze, startCell, goalCell):

    fringe = list()
    fringe.append(startCell)
    visitedCells = list()
    visitedCells.append(startCell)
    
    parentDict = {}
    maxFringe = 0

    while fringe:
        (i, j) = fringe.pop()
        maxFringe = max(maxFringe, len(fringe))
        if (i, j) == goalCell:
            return 1, parentDict, maxFringe
        neighbors = prioritization(maze, goalCell[0]-1, visitedCells, i, j)
        if neighbors:
            for n in neighbors:
                parentDict[n] = (i, j)
                visitedCells.append(n)
                fringe.append(n)

    return 0, parentDict, maxFringe


"""computing the cost by adding the path length and the number of the nodes counted"""

def computeCost(maze, start, goal, h):

    if h == "DFS":
        id_arr_goal, parentDict, maxFringe = dfsAlgo(maze, start, goal)
    elif h == "IDFS":
       id_arr_goal, parentDict, maxFringe = IDFS(maze, start, goal)
    else:
       id_arr_goal, parentDict, maxFringe = A_star(maze, start, goal, "manhattan") 
    
    if(id_arr_goal):
        numPathLength, path = pathLength(parentDict, start, goal)
        # computed cost by adding the length of the path and the number of the nodes counted
        cost = numPathLength + maxFringe 
    else:
        cost = 0
        path = [(0, 1)]

    return id_arr_goal, cost, path


"""Fuction for iterating hillClimibing on the same maze"""

def hillClimbing(p, num_loops, h):

    print("INITITALIZE MAZE")
    id_arr_goal = 0
    while not id_arr_goal:
        maze = renderMaze(size, p)
        id_arr_goal, cost, path = computeCost(maze, start, goal, h)

    #save this current maze to track the previous maze
    current_maze = maze
    hardest_maze = np.copy(maze)
    max_cost = 0
    
    if h == "DFS":
        id_arr_goal, parentDict, maxFringe = dfsAlgo(maze, start, goal)
        visualizePath(maze*100, parentDict, start, goal)
        
        # original cost of the first maze
        origin_cost = cost 
        list_of_costs = [origin_cost]
        print("Cost of Original Maze: " + str(cost))
        count = 0

        '''Iteration for num_restarts, this is loop for DFS algorithm by maximal fringe size'''
        while count < num_loops:
            while True:
                # Generate this loop for swapping the cell to the block from the path that is already found to get the hardest maze and cost
                
                swap_cell = random.choice(path)
                
                if swap_cell != start or swap_cell != goal:

                    #current_maze = np.copy(maze)
                    maze[swap_cell] = 0
                    maze[swap_cell[1]][swap_cell[0]] = 1

                    new_dfs_id, new_cost, new_path = computeCost(maze, start, goal, h)

                    if new_dfs_id and new_cost > cost:
                        cost = new_cost
                        path = new_path
                        current_maze = np.copy(maze)

                    else:
                        maze = np.copy(current_maze)
                        if cost > max_cost:         # get the hardest maze's cost
                            max_cost = cost
                            hardest_maze = np.copy(maze)
                        count += 1
                        list_of_costs.append(cost)
                        break

    
    elif h == "IDFS":
        id_arr_goal, parentDict, maxFringe = IDFS(maze, start, goal)
        visualizePath(maze*100, parentDict, start, goal)

        # original cost of the first maze
        origin_cost = cost  
        list_of_costs = [origin_cost]
        print("Cost of Original Maze: " + str(cost))
        count = 0

        ''''#Here we terminate this iteration after we iterate for number of restarts
            # '''

        while count < num_loops:
            while True:
                # Generate this loop for swapping the cell to the block from the path that is already found to get the hardest maze and cost
                
                swap_cell = random.choice(path)
                
                if swap_cell != start or swap_cell != goal:

                    current_maze = np.copy(maze)
                    maze[swap_cell] = 0
                    maze[swap_cell[1]][swap_cell[0]] = 1

                    new_is_reachable, new_cost, new_path = computeCost(maze, start, goal, h) 

                    if new_is_reachable:
                        cost = new_cost
                        path = new_path

                    else:
                        maze = np.copy(current_maze)
                        if cost > max_cost:    
                            max_cost = cost
                            hardest_maze = np.copy(maze)
                        count += 1
                        list_of_costs.append(cost)
                        break
    
    else:
        id_arr_goal, parentDict, maxFringe = A_star(maze, start, goal, "manhattan")
        visualizePath(maze*100, parentDict, start, goal)

        origin_cost = cost 
        list_of_costs = [origin_cost]
        print("Cost of Original Maze: " + str(cost))
        count = 0

        ''''#Here we terminate this iteration after we iterate for number of restarts
            # '''

        while count < num_loops:

            while True:
                # Generate this loop for swapping the cell to the block from the path that is already found to get the hardest maze and cost
                
                swap_cell = random.choice(path)
                if swap_cell != start or swap_cell != goal:
    
                    current_maze = np.copy(maze)
                    maze[swap_cell] = 0
                    maze[swap_cell[1]][swap_cell[0]] = 1

                    new_is_reachable, new_cost, new_path = computeCost(maze, start, goal, h) 

                    if new_is_reachable:
                        cost = new_cost
                        path = new_path

                    else:
                        maze = np.copy(current_maze)
                        if cost > max_cost:        
                            max_cost = cost
                            hardest_maze = np.copy(maze)
                        count += 1
                        list_of_costs.append(cost)
                        break

    return hardest_maze, max_cost, origin_cost, list_of_costs


"""###this function is for iterating hill climbing function multiple times to get the hardest maze."""

def iterateHC(restarts, iterations, h):

    rate_of_increase = []
    final_maze = (renderMaze(size, p), 0, {})

    for i in range(iterations):

        # Return the hardest maze for a given original maze
        hardest_maze, max_cost, origin_cost, y = hillClimbing(p, restarts, h)
        print("HARDEST MAZE")

        if h == "DFS":
            id_arr_goal, parentDict, maxFringe = dfsAlgo(hardest_maze, start, goal)
        elif h == "IDFS":
            id_arr_goal, parentDict, maxFringe = IDFS(hardest_maze, start, goal)
        else:
            id_arr_goal, parentDict, maxFringe = A_star(hardest_maze, start, goal, "manhattan") 
        
        visualizePath(hardest_maze*100, parentDict, start, goal)
        print("Cost of Hardest Maze: " + str(max_cost))

        if max_cost > final_maze[1]:
            final_maze = (hardest_maze, max_cost, parentDict)

        # Display the path in the hardest maze
        sns.set(style="whitegrid", color_codes=True)
        plt.figure(figsize=(15, 10))
        x = np.arange(0, len(y))
        sns.lineplot(x, y)
        plt.xlabel("# of Maximum Local Search")
        plt.ylabel("Cost of the Maze by Maximal Fringe")
        
        if h == "DFS":
            plt.title("Cost of Mazes with DFS algorithm")
        elif h == "IDFS":
            plt.title("Cost of Mazes with IDFS algorithm")
        else:
            plt.title("Cost of Mazes with A_STAR algorithm")
        
        rate_of_increase.append((max_cost - origin_cost) / origin_cost)
        print("Increase rate of the cost: " + str(rate_of_increase[i]+1))

    return final_maze

# Run multiple Hill climbing to find a hard maze:
size = 100
p = 0.2
start = (0, 0)
goal = (size-1, size-1)

restarts = 100
iterations = 1

""""###Please Select Your Algorithm that you wnat to implement
# Options: DFS, IDFS, A_star
# """
algorithm = "A_star"
final_maze = iterateHC(restarts, iterations, algorithm)

print("\nFINAL HARDEST MAZE:")
visualizePath(final_maze[0]*100, final_maze[2], start, goal)
print("Cost of Final Maze: " + str(final_maze[1]))