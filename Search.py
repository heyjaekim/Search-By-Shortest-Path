import numpy
import queue
import math
import time
from Map import Map

class Search:
    def __init__(self, grid):
        self.grid = grid
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

    def dfs_algo(self):

        dfsStack = queue.LifoQueue()
        dfsStack.put(self.startCell)
        prev_cells = {}
        cellsVisited = set()
        cellsVisited.add(self.startCell)
        maxFringe = 0

        while not dfsStack.empty():
            cell = dfsStack.get()
            maxFringe = max(maxFringe, dfsStack.qsize())
            if cell == self.goalCell:
                path = self.render_path(prev_cells)
                return {"Status": "Found Path", "Visited cells": cellsVisited,
                        "# of Visited Cells": len(cellsVisited), "Path length": len(path), "Path length from Goal": "", 
                        "Path": path, "Path from Goal": [], "Intersecting Cell": (),
                        "Max fringe size": (maxFringe)}

            for dir in self.grid.neighborCells(cell):
                if dir not in cellsVisited:
                    prev_cells[dir] = cell
                    cellsVisited.add(dir)
                    dfsStack.put(dir)

        return {"Status": "Unable to find the path", "Visited cells": cellsVisited,
                "# of Visited Cells": len(cellsVisited), "Path length": "N/A", "Path length from Goal": "", 
                "Path": [], "Path from Goal": [], "Intersecting Cell": (),
                "Max fringe size": "N/A"}

    def improved_dfs_algo(self):
        dfsFringe = []
        dfsFringe.append(self.startCell)
        
        prev_cells = {}
        cellsVisited = set()
        cellsVisited.add(self.startCell)
        maxFringe = 0

        while not dfsFringe.empty():
            cell = dfsFringe.get()
            maxFringe = max(maxFringe, dfsFringe.qsize())
            if cell == self.goalCell:
                path = self.render_path(prev_cells)
                return {"Status": "Found Path", "Visited cells": cellsVisited,
                        "# of Visited Cells": len(cellsVisited), "Path length": len(path), "Path length from Goal": "", 
                        "Path": path, "Path from Goal": [], "Intersecting Cell": (),
                        "Max fringe size": (maxFringe)}

            for dir in self.grid.neighborCells(cell):
                if dir not in cellsVisited:
                    prev_cells[dir] = cell
                    cellsVisited.add(dir)
                    dfsFringe.put(dir)

        return {"Status": "Unable to find the path", "Visited cells": cellsVisited,
                "# of Visited Cells": len(cellsVisited), "Path length": "N/A", "Path length from Goal": "", 
                "Path": [], "Path from Goal": [], "Intersecting Cell": (),
                "Max fringe size": "N/A"}

    def bfs_algo(self):

        stack = queue.Queue()
        visitedCells = set()
        parentSet = {}
        stack.put(self.startCell)
        visitedCells.add(self.startCell)
        maxFringe = 0

        while not stack.empty():
            cell = stack.get()
            maxFringe = max(maxFringe, stack.qsize())
            if cell == self.goalCell:
                path = self.render_path(parentSet)
                return {"Status": "Found Path", "Visited cells": visitedCells,
                "# of Visited Cells": len(visitedCells), "Path": path, "Path length": len(path), "Path length from Goal": "", 
                        "Path from Goal": [], "Intersecting Cell": (),
                        "Max fringe size": (maxFringe)}

            for dir in self.grid.neighborCells(cell):
                if dir not in visitedCells:
                    parentSet[dir] = cell
                    visitedCells.add(dir)
                    stack.put(dir)

        return {"Status": "Unable to find the path", "Visited cells": visitedCells,
                "# of Visited Cells": len(visitedCells), "Path length": "N/A", "Path length from Goal": "N/A", 
                "Path": [], "Path from Goal": [], "Intersecting Cell": (),
                "Max fringe size": "n/a"}

    def isIntersecting(intersectionNode, s_fringe, t_fringe):
        if intersectionNode == (-1, -1):
            for c in s_fringe:
                if(c in t_fringe):
                    return c
            return (-1, -1)

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
        maxFringe = 0

        while (s_fringe and t_fringe):
            s_cell = s_fringe.pop(0)
            t_cell = t_fringe.pop(0)
            maxFringe = max(maxFringe, max(len(s_fringe), len(t_fringe)))

            for dir in self.grid.neighborCells(s_cell):
                if dir not in s_visited:
                    s_parent[dir] = s_cell
                    s_visited.append(dir)
                    s_fringe.append(dir)

            intersectNode = Search.isIntersecting(intersectNode, s_fringe, t_fringe)

            for dir in self.grid.neighborCells(t_cell):
                if dir not in t_visited:
                    t_parent[dir] = t_cell
                    t_visited.append(dir)
                    t_fringe.append(dir)

            intersectNode = Search.isIntersecting(intersectNode, s_fringe, t_fringe)

            if intersectNode != (-1, -1):
                print("intersecting at cell:", intersectNode)
                path = self.render_path_bidir(s_parent, intersectNode)
                path2 = self.render_path_bidir_fromEnd(t_parent, intersectNode)
                s_visited.extend(t_visited)
                return {"Status": "Found Path", "Visited cells": s_visited,
                        "# of Visited Cells": len(s_visited), "Path length": len(path), "Path length from Goal": len(path2), 
                        "Path": path, "Path from Goal": path2, "Intersecting Cell": intersectNode,
                        "Max fringe size": (maxFringe)}

        return {"Status": "Unable to find the path", "Visited cells": s_visited,
                "# of Visited Cells": len(s_visited), "Path length": "N/A", "Path length from Goal": "N/A", 
                "Path": [], "Path from Goal": [], "Intersecting Cell": (),
                "Max fringe size": "n/a"}

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
                return {"Status": "Found Path", "Visited cells": cellsVisited,
                        "# of Visited Cells": len(cellsVisited), "Path": path, "Path length": len(path), "Path length from Goal": "", 
                        "Path from Goal": [], "Intersecting Cell": (),
                        "Max fringe size": (maxFringe)}

            for dir in self.grid.neighborCells(cell):
                newCost = mincost[cell] + 1
                # or new_cost < cost_so_far[next_cell]:
                if dir not in cellsVisited:
                    mincost[dir] = newCost
                    parentCellSet[dir] = cell
                    cellsVisited.add(dir)

                    priority = newCost + self.findHeuristic(dir, heuristic)
                    priorityQ.put((priority, dir))

        return {"Status": "Unable to find the path", "Visited cells": cellsVisited,
                "# of Visited Cells": len(cellsVisited), "Path length": "N/A", "Path length from Goal": "N/A", 
                "Path": [], "Path from Goal": [], "Intersecting Cell": (),
                "Max fringe size": "n/a"}

    def findHeuristic(self, cell, heuristic):
        (x1, y1) = cell
        (x2, y2) = self.goalCell
        if heuristic == "euclidean":
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        elif heuristic == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)

"""
current_map = Map(100, 0.2)

#algoLists = ["BFS", "DFS", "A_Manhattan", "A_Euclidean", "BD_BFS"]
#factsLists = ["path_length", "time", "nodes_explored", "max_fringe_size"]

print("--------------------------------\nUsing DFS")
start_time = time.time()
current_map.solution = FindSolution(current_map).dfs_algo()
current_time = round(time.time() - start_time, 4)
current_map.print_solution()
print("Time: ", current_time)

print("--------------------------------\nUsing BFS")
start_time = time.time()
current_map.solution = FindSolution(current_map).bfs_algo()
current_time = round(time.time() - start_time, 4)
current_map.print_solution()
print("Time: ", current_time)

print("--------------------------------\nUsing A* Euclidean")
start_time = time.time()
current_map.solution = FindSolution(current_map).a_star_algo("euclidean")
current_time = round(time.time() - start_time, 4)
current_map.print_solution()
print("Time: ", current_time)

print("--------------------------------\nUsing A* Manhattan")
start_time = time.time()
current_map.solution = FindSolution(current_map).a_star_algo("manhattan")
current_time = round(time.time() - start_time, 4)
current_map.print_solution()
print("Time: ", current_time)

print("--------------------------------\nUsing Bi Directional BFS")
start_time = time.time()
current_map.solution = FindSolution(current_map).biBFS()
current_time = round(time.time() - start_time, 4)
current_map.print_solution()
print("Time: ", current_time)
"""