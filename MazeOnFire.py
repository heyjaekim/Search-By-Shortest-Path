import numpy as np
import random
import seaborn
import math
import queue
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#function for rendering the maze
def renderMaze(size, prob):

    mat = np.random.uniform(size=[size, size])
    mat = (mat > prob).astype(int)
    mat[0, 0] = 1
    mat[size - 1, size - 1] = 1
    return mat

#visualizing the path for runner and the fire cells
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
        seaborn.heatmap(canvas, cmap=ListedColormap(['black', 'green', 'crimson', 'blue', 'papayawhip']),
                        linewidths=.5,  square=True, cbar=False, xticklabels=False, yticklabels=False)
        plt.show()

        return path_length

#computing the path from the start to the end
def computePath(pathList, start, goal):

    path = list()
    path.append(start)

    if bool(pathList):
        while(goal in pathList and pathList[goal] != start):
            pointer = goal
            goal = pathList[(pointer)]
            path.append(goal)

        path.append(pathList[goal])
        path.reverse()

    return len(path), path


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

#check if the cell is a valid value
def checkCellValue(maze, cell, fringe):
    if cell not in fringe and maze[cell] != -1 and maze[cell] != 0:
        fringe.append(cell)
    return fringe

#append the neighbor cells into the cells list
def neighborCells(cell):

    x = cell[0]
    y = cell[1]
    neighborCells = set()
    DIRS = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
    for (i, j) in DIRS:
        if inBound((i, j)):
            if maze[i, j] == 1:
                neighborCells.add((i, j))
    return neighborCells

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


def A_star(maze, startCell, goalCell, h):

    priorityQ = queue.PriorityQueue()
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
            return 1, pathSet

        for dir in neighborCells(cell):
            newCost = mincost[cell] + 1
            if dir not in cellsVisited:
                mincost[dir] = newCost
                pathSet[dir] = cell
                cellsVisited.add(dir)
                priority = newCost + findHeuristic(goalCell, dir, h)
                priorityQ.put((priority, dir))

    return 0, pathSet


'''implementing maze on fire algorithms'''


def validPath(maze, cell, visited):
    x = cell[0]
    y = cell[1]
    if x == -1 or x == size or y == -1 or y == size:
        return False
    elif maze[x, y] == 0 or cell in visited:
        return False
    return True


def getMinDist(cell, manhattanDists, fireCells, cells, safeDist):
    for f in fireCells:
        manhattanDists.append(abs(f[0]-cell[0]) + abs(f[1]-cell[1]))
    minDist = 0
    if manhattanDists:
        minDist = min(manhattanDists)

    safeDist.append((size - cell[0]) + (size - cell[1]) - minDist)
    cells.append(cell)
    return cells, safeDist

def prioritization(maze, visited, x, y, fireCells):
    priorities = list()
    cells = list()
    dist = list()

    manhattanDists = list()
    cell = (x, y+1)
    if(validPath(maze, cell, visited)):
        cells, dist = getMinDist(cell, manhattanDists, fireCells, cells, dist)

    manhattanDists = list()
    cell = (x+1, y)
    if(validPath(maze, cell, visited)):
        cells, dist = getMinDist(cell, manhattanDists, fireCells, cells, dist)

    manhattanDists = list()
    cell = (x, y-1)
    if(validPath(maze, cell, visited)):
        cells, dist = getMinDist(cell, manhattanDists, fireCells, cells, dist)

    manhattanDists = list()
    cell = (x-1, y)
    if(validPath(maze, cell, visited)):
        cells, dist = getMinDist(cell, manhattanDists, fireCells, cells, dist)

    for i in range(len(dist)):
        val = dist.index(min(dist))
        dist.pop(val)
        child = cells.pop(val)
        priorities.append(child)

    priorities.reverse()
    return priorities

def fireStrategyOne(maze, start, goal, fireStart, q):

    fringe = list()
    fringe.append(start)
    visited = list()
    visited.append(start)
    btList = {}  # back tracking list

    maze[fireStart] = -1
    fireCells = list()
    fireCells.append(fireStart)
    potFireCells = list()
    newFire = list()

    while fringe:
        (i, j) = fringe.pop()
        #runner's goal is on fire - error
        if goal in fireCells:    
            return 3, (i, j), btList, fireCells
        #runner's current location is on fire - error
        if (i, j) in fireCells:    
            return 2, (i, j), btList, fireCells

        #valid answer for runner to reach to the goal
        if (i, j) == goal:    
            return 1, goal, btList, fireCells

        # Generating and adding child nodes in fringe
        priorities = prioritization(maze, visited, i, j, fireCells)  
        if priorities:
            for p in priorities:
                btList[p] = (i, j)
                fringe.append(p)
                visited.append(p)

        # Adding neighbors to the fringe
        while fireCells:
            neighbors = addPotentialFireCells(fireCells.pop())
            while neighbors:
                neighbor = neighbors.pop()
                potFireCells = checkCellValue(maze, neighbor, potFireCells)


        listCopy = potFireCells.copy()
        while listCopy:
            k = 0
            cell = listCopy.pop()
            neighbors = addPotentialFireCells(cell)

            for v in neighbors:
                if (maze[v] == -1):
                    k += 1

            prob = 1 - pow(1-q, k)
            if(random.uniform(0, 1) < prob):  # If the cell catches fire
                newFire.append(cell)
                potFireCells.remove(cell)

        while newFire:
            n = newFire.pop()
            fireCells.append(n)
            maze[n] = -1

    return 0, (i, j), btList, fireCells


start_time = time.time()
size = 100
p = 0.2
q = 0.1
start = (0, 0)
goal = (size - 1, size - 1)
fire_start = (random.randint(1,size-1),random.randint(1,size-1))
print(fire_start)
#fire_goal = (size-1, 0)

idAstar = 0
while not idAstar:
    maze = renderMaze(size, p)
    idAstar, pathSet = A_star(maze, start, goal, "manhattan")

idr, runnerCoord, pathSet_fire, fireCells = fireStrategyOne(maze, start, goal, fire_start, q)
print("Time taken for strategy 1" + str(time.time()-start_time))
canvas = maze*100
visualizePath(canvas, pathSet, start, goal)

# Function to implement search to stay away from new cells on fire.
def fireStrategyTwo(maze, start, goal, fire_start, q):

    fringe = list()
    fringe.append(start)
    visited = list()
    visited.append(start)
    btList = {}  
    maze[fire_start] = -1  

    fireCells = list()
    fireCells.append(fire_start)
    potFireFringe = list()
    newFire = list()
    while fringe:
        (i, j) = fringe.pop()
        while fireCells:
            neighbors = addPotentialFireCells(fireCells.pop())

            while neighbors:
                neighbor = neighbors.pop()
                potFireFringe = checkCellValue(maze, neighbor, potFireFringe)

        copyPotFireFringe = potFireFringe.copy()
        while copyPotFireFringe:
            k = 0
            space = copyPotFireFringe.pop()
            neighbors = addPotentialFireCells(space)

            for v in neighbors:
                if (maze[v] == -1):
                    k += 1

            fireProb = 1 - pow(1-q, k)
            if(random.uniform(0, 1) < fireProb):
                newFire.append(space)
                potFireFringe.remove(space)

        while newFire:
            n = newFire.pop()
            fireCells.append(n)
            maze[n] = -1

        if goal in fireCells:
            return 3, (i, j), btList, fireCells

        if (i, j) in fireCells:
            return 2, (i, j), btList, fireCells

        if (i, j) == goal:   
            return 1, goal, btList, fireCells

        # Generating priorities for min distance from firecells and added into priorities to append on fringe, btlist, and visited.
        priorities = prioritization(maze, visited, i, j, fireCells)
        if priorities:
            for p in priorities:
                btList[p] = (i, j)
                fringe.append(p)
                visited.append(p)

    return 0, (i, j), btList, fireCells

start_time = time.time()
size = 100
p = 0.2
q = 0.1
fire_start = (random.randint(1,size-1),random.randint(1,size-1))
print(fire_start)

idAstar = 0

while not idAstar:
    maze = renderMaze(size, p)
    idAstar, pathSet = A_star(maze, start, goal, "manhattan")

idr, runnerCoord, pathSet_fire, fireCells = fireStrategyTwo(maze, start, goal, fire_start, q)
if(idr == 1):
    print("Runner reacehd to the goal without fire.")
else:
    print("Runner either couldn't reach to the goal or kept on fire.")
maze_temp = maze*100

print("Time taken for strategy 2: " + str(time.time()-start_time))
visualizePath(maze_temp, pathSet, start, goal)

#get the min distance from the fire to the cell to get the maximum minimal distance from the fringe
def getMinDistsFromFire(maze, dist, neighbors, cells, cell, visited):
    dist.append(0)
    xCoord = size-1-cell[0]
    yCoord = size-1-cell[1]
    for i in neighbors:
        if validPath(maze, i, visited):
            dist[len(dist)-1] += 1
        dist[len(dist)-1] = xCoord + yCoord - dist[len(dist)-1]
    cells.append(cell)
    return cells, dist    

#another priortization function just for the strategy three
def prioritizationTwo(maze, visited, x, y):

    priorities = list()
    cells = list()
    dist = list()

    cell = (x, y+1)
    neighbors = [(x+1, y+1), (x, y+2), (x-1, y+1)]
    if(validPath(maze, cell, visited)):
        cells, dist = getMinDistsFromFire(maze, dist, neighbors, cells, cell, visited)
    cell = (x+1, y)
    neighbors = [(x+1, y+1), (x+2, y), (x+1, y-1)]
    if(validPath(maze, cell, visited)):
        cells, dist = getMinDistsFromFire(maze, dist, neighbors, cells, cell, visited)
    cell = (x, y-1)
    neighbors = [(x+1, y-1), (x, y-2), (x-1, y-1)]
    if(validPath(maze, cell, visited)):
        cells, dist = getMinDistsFromFire(maze, dist, neighbors, cells, cell, visited)
    cell = (x-1, y)
    neighbors = [(x-1, y+1), (x-2, y), (x-1, y-1)]
    if(validPath(maze, cell, visited)):
        cells, dist = getMinDistsFromFire(maze, dist, neighbors, cells, cell, visited)
    for i in range(len(dist)):
        ind = dist.index(min(dist))
        dist.pop(ind)
        current_child = cells.pop(ind)
        priorities.append(current_child)

    priorities.reverse()
    return priorities

def fireStrategyThree(maze, start, goal, fire_start, q):

    fringe = list()
    fringe.append(start)
    visited = list()
    visited.append(start)
    btList = {}    # backtracking the path

    maxFringe = 0

    maze[fire_start] = -1

    fireCells = list()
    fireCells.append(fire_start)
    potFireFringe = list()
    new_fire = list()

    while fringe:
        (i, j) = fringe.pop()

        if goal in fireCells:
            return 3, (i, j), btList, fireCells

        if (i, j) in fireCells:
            return 2, (i, j), btList, fireCells

        maxFringe = max(maxFringe, len(fringe))

        if (i, j) == goal:    # to check if the goal state is found
            return 1, goal, btList, fireCells

        # Generating and adding child nodes in fringe
        priorities = prioritizationTwo(maze, visited, i, j)
        if priorities:
            for p in priorities:
                btList[p] = (i, j)
                fringe.append(p)
                visited.append(p)

        while fireCells:
            neighbours = addPotentialFireCells(fireCells.pop())
            while neighbours:
                neighbor = neighbours.pop()
                if maze[neighbor] != -1 and maze[neighbor] != 0 and neighbor not in potFireFringe:
                    potFireFringe.append(neighbor)

        copyPotFireFringe = potFireFringe.copy()
        while copyPotFireFringe:
            k = 0
            cell = copyPotFireFringe.pop()
            neighbours = addPotentialFireCells(cell)

            for v in neighbours:
                if (maze[v] == -1):
                    k += 1

            probability = 1 - pow(1-q, k)

            if(random.uniform(0, 1) < probability):  # If the cell catches fire
                new_fire.append(cell)
                potFireFringe.remove(cell)

        while new_fire:
            n = new_fire.pop()
            fireCells.append(n)
            maze[n] = -1

    return 0, (i, j), btList, fireCells


start_time = time.time()

size = 100
p = 0.2
q = 0.2
fire_start = (random.randint(1,size-1),random.randint(1,size-1))
print(fire_start)
idAstar = 0
is_fire_reached = 0

while not idAstar:
    maze = renderMaze(size, p)
    idAstar, pathSet = A_star(maze, start, goal, "manhattan")

idr, runnerCoord, pathSet_fire, fireCells = fireStrategyThree(maze, start, goal, fire_start, q)
if(idr == 1):
    print("Runner reacehd to the goal without fire.")
else:
    print("Runner either couldn't reach to the goal or kept on fire.")
maze_temp = maze*100

print("Time taken for strategy 3: " + str(time.time()-start_time))
visualizePath(maze_temp, pathSet, start, goal)

"""Success rate of new algorithm"""

# Function to check the success rate of the new algorithm

def success_rate_new(size, p, t):
  
    probability_solvable = []
    for q in range(0, 11, 1):
        q = q/10.0
        successCount = 0
        
        for i in range(t):
            idAstar = 0
            while not idAstar:
                maze = renderMaze(size, p)
                idAstar, pathList = A_star(maze, start, goal , "manhattan")
                print(str(idAstar))
            result, runnerCoord, pathList_fire, fireCell = fireStrategyTwo(maze, start, goal, fire_start, q)
            if(result == 1):
                successCount+=1   
            else :
                continue
            
        probability_solvable.append(successCount/t)
        print("Success for q = " + str(q) + " is " + str(successCount))

    print(probability_solvable)
    x = np.arange(0,1.1,0.1)
   
    plt.clf()
    plt.cla()
    plt.close()
    plt.bar(x, probability_solvable, width = 0.05 )
    
    plt.xlabel("q")
    plt.ylabel("Probability of success")
    plt.title("Density vs solvability for dim = " + str(size) + ", #trials = "+ str(t))
    plt.xticks(x)
    plt.show()

size = 60
start = (0, 0)
goal = (size - 1, size - 1)
fire_start = (random.randint(1,size-1),random.randint(1,size-1))
success_rate_new(size, p, 50)