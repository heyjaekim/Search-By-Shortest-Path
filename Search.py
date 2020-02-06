import numpy
import queue
import math
import Visual
from Map import Map

DEFAULT_SIZE = 100

class FindSolution:
    def __init__(self, grid):
        self.grid = grid
        self.time = 0
        self.result = "n/a"
        self.startCell = (0, 0)
        self.goalCell = (grid.size - 1, grid.size - 1)

    def render_path(self, pathDict):
        #Build a path (list) from a dictionary of parent cell -> child cell
        current = pathDict[self.goalCell]
        path = []
        while current != (0, 0):
            path += (current,)
            current = pathDict[current]
        return path[::-1]

    def render_path_bidir(self, pathDict, position):
        current = pathDict[position]
        path = []
        while current != (0,0):
            path += (current,)
            current = pathDict[current]
        return path[::-1]

    def dfs_algo(self):

        lifoStack = queue.LifoQueue()
        lifoStack.put(self.startCell)
        parentSet = {}
        cellsVisited = set()
        cellsVisited.add(self.startCell)

        while not lifoStack.empty():
            cell = lifoStack.get()
            if cell == self.goalCell:
                path = self.render_path(parentSet)
                return {"Status": "Found Path",
                        "No of visited cells": len(cellsVisited), "Path length": len(path)}

            for dir in self.grid.neighborCells(cell):
                if dir not in cellsVisited:
                    parentSet[dir] = cell
                    cellsVisited.add(dir)
                    lifoStack.put(dir)

        return {"Status": "Path Not Found!!!",
                "No of visited cells": len(cellsVisited), "Path length": "N/A"}

    def bfs_algo(self):
        
        stack = queue.Queue()
        visitedCells = set()
        parentSet = {}
        stack.put(self.startCell)
        visitedCells.add(self.startCell)


        while not stack.empty():
            cell = stack.get()
            if cell == self.goalCell:
                path = self.render_path(parentSet)
                return {"Status": "Found Path",
                        "No of visited cells": len(visitedCells),"Path length": len(path)}

            for dir in self.grid.neighborCells(cell):
                if dir not in visitedCells:
                    parentSet[dir] = cell
                    visitedCells.add(dir)
                    stack.put(dir)

        return {"Status": "Path Not Found!!!",
                "No of visited cells": len(visitedCells), "Path length": "N/A"}

    def isIntersecting(intersectionNode, s_fringe,t_fringe):
        if intersectionNode == (-1,-1):
            for c in s_fringe:
                if(c in t_fringe):
                    return c
            return (-1,-1)
    
        else:
            return intersectionNode

    def biBFS(self):
        
        s_visited = []
        t_visited = []
        s_visited.append(self.startCell)
        t_visited.append(self.goalCell)

        s_fringe = []
        t_fringe = []
        s_fringe.append(self.startCell)
        t_fringe.append(self.goalCell)

        s_parent = {}
        t_parent = {}

        intersectNode = (-1, -1)

        while (s_fringe and t_fringe):
            s_cell = s_fringe.pop(0)
            t_cell = t_fringe.pop(0)
            
            for dir in self.grid.neighborCells(s_cell):
                if dir not in s_visited:
                    s_parent[dir] = s_cell
                    s_visited.append(dir)
                    s_fringe.append(dir)
            
            intersectNode = FindSolution.isIntersecting(intersectNode, s_fringe, t_fringe)
            
            for dir in self.grid.neighborCells(t_cell):
                if dir not in t_visited:
                    t_parent[dir] = t_cell
                    t_visited.append(dir)
                    t_fringe.append(dir)

            intersectNode = FindSolution.isIntersecting(intersectNode, s_fringe, t_fringe)

            if intersectNode != (-1, -1):
                print("intersecting at cell: ")
                print(intersectNode)
                path = self.render_path_bidir(s_parent, intersectNode)
                return {"Status": "Found Path",
                        "No of visited cells": len(s_visited), "Path length": len(path)}

        return {"Status": "Path Not Found!!!",
                        "No of visited cells": len(s_visited), "Path length": "N/A"}

    def a_star_algo(self, heuristic):
        """
        this is for finding the path for A-Star
        return the list of path status, # of visted cells, path, and path length
        """
        priorityQ = queue.PriorityQueue()
        parentCellSet = {}
        mincost = {}
        cellsVisited = set()

        priorityQ.put((0, self.startCell))
        mincost[self.startCell] = 0
        cellsVisited.add(self.startCell)

        while not priorityQ.empty():
            cell = priorityQ.get() #referring to the curernt cell
            cell = cell[1]
            if cell == self.goalCell:
                path = self.render_path(parentCellSet)
                return {"Status": "Found Path",
                        "No of visited cells": len(cellsVisited), "Path length": len(path)}

            for dir in self.grid.neighborCells(cell):
                newCost = mincost[cell] + 1
                if dir not in cellsVisited:  # or new_cost < cost_so_far[next_cell]:
                    mincost[dir] = newCost
                    parentCellSet[dir] = cell
                    cellsVisited.add(dir)

                    priority = newCost + self.findHeuristic(dir, heuristic)
                    priorityQ.put((priority, dir))

        return {"Status": "Path Not Found!!!",
                "No of visited cells": len(cellsVisited), "Path length": "N/A"}

    def findHeuristic(self, cell, heuristic):
        (x1, y1) = cell
        (x2, y2) = self.goalCell
        if heuristic == "euclidean":
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        elif heuristic == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)

current_map = Map(100, 0.2)

print("--------------------------------\nUsing DFS")
current_map.solution = FindSolution(current_map).dfs_algo()
current_map.print_solution()
print("--------------------------------\nUsing BFS")
current_map.solution = FindSolution(current_map).bfs_algo()
current_map.print_solution()
print("--------------------------------\nUsing A* Euclidean")
current_map.solution = FindSolution(current_map).a_star_algo("euclidean")
current_map.print_solution()
print("--------------------------------\nUsing A* Manhattan")
current_map.solution = FindSolution(current_map).a_star_algo("manhattan")
current_map.print_solution()
print("--------------------------------\nUsing Bi Directional BFS")
current_map.solution = FindSolution(current_map).biBFS()
current_map.print_solution()