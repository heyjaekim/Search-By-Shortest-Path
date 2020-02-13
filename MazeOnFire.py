import numpy
import queue
import math
import time
import random
from Map import Map

FRAME_WIDTH = 700
DEFAULT_SIZE = 100
DEFAULT_PROB = 0.2

class Search:
    def __init__(self, grid):
        self.map = grid
        self.time = 0
        self.result = "n/a"
        self.startCell = (0, 0)
        self.goalCell = (grid.size - 1, grid.size - 1)

    def render_path(self, pathDict):
        # Build a path (list) from a dictionary of parent cell -> child cell
        current = pathDict[self.goalCell]
        path = []
        while current != (0, 0):
            path += (current,)
            current = pathDict[current]
        return path[::-1]

    def render_path_bidir(self, pathDict, position):
        current = pathDict[position]
        path = []
        while current != (0, 0):
            path += (current,)
            current = pathDict[current]
        return path[::-1]

    def render_path_bidir_fromEnd(self, pathDict, position):
        current = pathDict[position]
        path = []
        while current != (self.grid.size-1, self.grid.size-1):
            path += (current,)
            current = pathDict[current]
        return path[::+1]
    
    def checkValidity(self, cell, visitedCell):
        dim = self.map.shape[0]

        if cell[0] == -1 or cell[1] == -1 or cell[0] == dim or cell[1] == dim:
            return False
        elif self.map[cell[0],cell[1]] == 0 or cell in visitedCell:
            return False
        
        return True

        
    def trackback_neighborCells(self, goalCell, visited, x, y):
        prevCells = list()
        cell = list()
        path = list()

        cell.append((x+1, y))
        path.append((goalCell - cell[0][0]) + (goalCell - cell[0][1]))
        
        cell.append((x, y+1))
        path.append((goalCell - cell[1][0]) + (goalCell - cell[1][1]))
        
        cell.append((x-1, y))
        path.append((goalCell - cell[2][0]) + (goalCell - cell[2][1]))
        
        cell.append((x, y-1))
        path.append((goalCell - cell[3][0]) + (goalCell - cell[3][1]))
        
        for i in range(4):
           low = path.index(min(path))
           path.pop(low)
           currentCell = cell.pop(low)
           if(Search.checkValidity(current_map, currentCell, visited)) :
               prevCells.append(currentCell)
        prevCells.reverse()
        return prevCells
        

    def improved_dfs_algo(self):
        dfsFringe = list()
        dfsFringe.append(self.startCell)
        
        prev_cells = {}
        cellsVisited = list()
        cellsVisited.append(self.startCell)
        maxFringe = 0

        while dfsFringe:
            cell = dfsFringe.pop()
            maxFringe = max(maxFringe, len(dfsFringe))
            if cell == self.goalCell:
                path = self.render_path(prev_cells)
                return {"Status": "Found Path", "Visited cells": cellsVisited,
                        "# of Visited Cells": len(cellsVisited), "Path length": len(path), "Path length from Goal": "", 
                        "Path": path, "Path from Goal": [], "Intersecting Cell": (),
                        "Max fringe size": (maxFringe)}
            children = Search.trackback_neighborCells(self.map, self.goalCell[0] - 1, cellsVisited, cell[0], cell[1])
            if children:
                for dir in children:
                    #if dir not in cellsVisited:
                    prev_cells[dir] = cell
                    cellsVisited.append(dir)
                    dfsFringe.append(dir)

        return {"Status": "Unable to find the path", "Visited cells": cellsVisited,
                "# of Visited Cells": len(cellsVisited), "Path length": "N/A", "Path length from Goal": "", 
                "Path": [], "Path from Goal": [], "Intersecting Cell": (),
                "Max fringe size": "N/A"}

    def a_star_algo(self, heuristic):
        """
        this is for finding the path for A-Star
        return the list of path status, # of visted cells, path, and path length
        """
        priorityQ = queue.PriorityQueue()
        parentCellSet = {}
        mincost = {}
        cellsVisited = set()
        maxFringe = 0

        priorityQ.put((0, self.startCell))
        mincost[self.startCell] = 0
        cellsVisited.add(self.startCell)

        while not priorityQ.empty():
            cell = priorityQ.get()  # referring to the curernt cell
            cell = cell[1]
            maxFringe = max(maxFringe, priorityQ.qsize())

            if cell == self.goalCell:
                path = self.render_path(parentCellSet)
                return path, cellsVisited, maxFringe

            for dir in self.map.neighborCells(cell):
                newCost = mincost[cell] + 1
                # or new_cost < cost_so_far[next_cell]:
                if dir not in cellsVisited:
                    mincost[dir] = newCost
                    parentCellSet[dir] = cell
                    cellsVisited.add(dir)

                    priority = newCost + self.findHeuristic(dir, heuristic)
                    priorityQ.put((priority, dir))

        return cellsVisited

    def findHeuristic(self, cell, heuristic):
        (x1, y1) = cell
        (x2, y2) = self.goalCell
        if heuristic == "euclidean":
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        elif heuristic == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)

    def save_children(self, cellsVisited, newcell, fireCells):
        neighbors = []
        listCells = []
        h = [] #minimum distance from the computation of manhattan distance 
        
        dist = []
        x = newcell[0]+1
        y = newcell[1]
        newcell = ((x,y))
        mindist = 0
        if Map.in_bounds(maze, newcell):
            for i in fireCells:
                dist.append((abs(i[0] - newcell[0]) + abs(i[1]-newcell[1])))    # Manhattan distance to the goal
            if dist:
                mindist = min(dist)
            h.append( (self.grid.size - newcell[0]) + (self.grid.size - newcell[1]) - mindist)    # Factoring in the minimum distance from the fire
            listCells.append(newcell)

        dist = []
        x = newcell[0]
        y = newcell[1]+1
        newcell = ((x,y))
        mindist = 0
        if Map.in_bounds(maze, newcell):
            for i in fireCells:
                dist.append((abs(i[0] - newcell[0]) + abs(i[1]-newcell[1])))    # Manhattan distance to the goal
            if dist:
                mindist = min(dist)
            h.append( (self.grid.size - newcell[0]) + (self.grid.size - newcell[1]) - mindist)    # Factoring in the minimum distance from the fire
            listCells.append(newcell)

        dist = []
        x = newcell[0]-1
        y = newcell[1]
        newcell = ((x,y))
        mindist = 0
        if Map.in_bounds(maze, newcell):
            for i in fireCells:
                dist.append((abs(i[0] - newcell[0]) + abs(i[1]-newcell[1])))    # Manhattan distance to the goal
            if dist:
                mindist = min(dist)
            h.append( (self.grid.size - newcell[0]) + (self.grid.size - newcell[1]) - mindist)    # Factoring in the minimum distance from the fire
            listCells.append(newcell)

        dist = []
        x = newcell[0]
        y = newcell[1]-1
        newcell = ((x,y))
        mindist = 0
        if Map.in_bounds(maze, newcell):
            for i in fireCells:
                dist.append((abs(i[0] - newcell[0]) + abs(i[1]-newcell[1])))    # Manhattan distance to the goal
            if dist:
                mindist = min(dist)
            h.append( (self.grid.size - newcell[0]) + (self.grid.size - newcell[1]) - mindist)    # Factoring in the minimum distance from the fire
            listCells.append(newcell)
        
        for i in range(len(h)):
            temp = h.index(min(h))
            h.pop(temp)
            currentCell = listCells.pop(temp)
            neighbors.append(currentCell)
        
        neighbors.reverse()
        return neighbors



    def fireStrategyOne(self, start, goal, fire_start, fire_goal, q):
        self.map[fire_start] = -1
        prevCells = {}
        cellsVisited = set()

        fringe = list()
        fringe.append(start)
        
        fireFringe = list()
        fireFringe.append(fire_start)
        
        yetFireFringe = list()
        newFireFringe = list()
        fireCells = list()
        fireCells.append((0, DEFAULT_SIZE-1))

        maxFringe = 0
        nodeCounts = 0

        while fringe:
            #tempMaze = self.map*100
            tempCell = fringe.pop()
            
            if goal in fireCells:
                            
                return 3, tempCell, prevCells, nodeCounts, maxFringe, fireCells
            
            if tempCell in fireCells:
                
                return 2, tempCell, prevCells, nodeCounts, maxFringe, fireCells

            nodeCounts += 1
            maxFringe = max(maxFringe, len(fringe))

            if tempCell == goal:
                return 1, tempCell, prevCells, nodeCounts, maxFringe, fireCells
            
            childNodes = save_children(self.grid, cellsVisited, tempCell, fireFringe)
            if childNodes:
                for t in childNodes:
                    prevCells[t] = tempCell
                    fringe.append(t)
                    cellsVisited.add(t)
            
            while fireFringe:
                neighbors = Map.neighborCells(fireFringe.pop())
                while neighbors:
                    nei = neighbors.pop()
                    if self.grid[nei] != -100 and self.grid[nei] != 0 and nei not in yetFireFringe:
                        yetFireFringe.append(nei)
            
            fringeCopy = yetFireFringe.copy()
            while fringeCopy:
                k = 0
                temp = fringeCopy.pop()
                neighbors = Map.neighborCells(temp)

                for t in neighbors:
                    if(self.grid[t] == -1):
                        k += 1
                
                prob = 1 - pow(1-q, k)
                if random.uniform(0, 1) < prob :
                    newFireFringe.append(temp)
                    yetFireFringe.remove(temp)

            while newFireFringe:
                nf = newFireFringe.pop()
                fireFringe.append(nf)
                self.grid[nf] = -1
                fireCells.append(nf)

            return 0, tempCell, prevCells, nodeCounts, maxFringe, fireCells
        
