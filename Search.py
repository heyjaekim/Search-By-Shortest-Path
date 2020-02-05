import numpy
import queue
import math
from Map import Map

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
                return {"Status": "Found Path", "Visited cells": cellsVisited,
                        "No of visited cells": len(cellsVisited), "Path": path, "Path length": len(path)}

            for dir in self.grid.neighborCells(cell):
                if dir not in cellsVisited:
                    parentSet[dir] = cell
                    cellsVisited.add(dir)
                    lifoStack.put(dir)

        return {"Status": "Path Not Found!!!", "Visited cells": cellsVisited,
                "No of visited cells": len(cellsVisited), "Path": [], "Path length": "N/A"}

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
                return {"Status": "Found Path", "Visited cells": visitedCells,
                        "No of visited cells": len(visitedCells), "Path": path, "Path length": len(path)}

            for dir in self.grid.neighborCells(cell):
                if dir not in visitedCells:
                    parentSet[dir] = cell
                    visitedCells.add(dir)
                    stack.put(dir)

        return {"Status": "Path Not Found!!!", "Visited cells": visitedCells,
                "No of visited cells": len(visitedCells), "Path": [], "Path length": "N/A"}

    def biDirSearch(self):
 
        s_queue = queue.Queue()
        t_queue = queue.Queue()

        s_visited = set()
        t_visited = set()

        intersectNode = -1
        parentDict = {}

        s_queue.put(self.startCell)
        s_visited.add(self.startCell)
        t_queue.put(self.goalCell)
        t_visited.add(self.goalCell)


        while not s_queue.empty() and not t_queue.empty():
            s_cell = s_queue.get()
            g_cell = t_queue.get()
            if s_cell == g_cell :
                path = self.render_path(parentDict)
                return {"Status": "Found Path", "Visited cells": s_visited,
                        "No of visited cells": len(s_visited), "Path": path, "Path length": len(path)}
            
            for dir in self.grid.neighborCells(s_cell):
                if dir not in s_visited:
                    parentDict[dir] = s_cell
                    s_visited.add(dir)
                    s_queue.put(dir)
            
            for dir in self.grid.neighborCells(g_cell):
                if dir not in s_visited:
                    parentDict[dir] = g_cell
                    t_visited.add(dir)
                    t_queue.put(dir)

        return {"Status": "Found Path", "Visitedddd cells": s_visited,
                        "No of visited cells": len(s_visited), "Path": [], "Path length": "N/A"}

    def a_star_algo(self, heuristic):
        """
        this is for finding the path for A-Star
        return the list of path status, # of visted cells, path, and path length
        """
        priorityQ = queue.PriorityQueue()
        parentCellSet = {}
        totalcost = {}
        cellsVisited = set()

        priorityQ.put((0, self.startCell))
        totalcost[self.startCell] = 0
        cellsVisited.add(self.startCell)

        while not priorityQ.empty():
            cell = priorityQ.get() #referring to the curernt cell
            cell = cell[1]
            if cell == self.goalCell:
                path = self.render_path(parentCellSet)
                return {"Status": "Found Path", "Visited cells": cellsVisited,
                        "No of visited cells": len(cellsVisited), "Path": path, "Path length": len(path)}

            for dir in self.grid.neighborCells(cell):
                newCost = totalcost[cell] + 1
                if dir not in cellsVisited:  # or new_cost < cost_so_far[next_cell]:
                    totalcost[dir] = newCost
                    parentCellSet[dir] = cell
                    cellsVisited.add(dir)

                    priority = newCost + self.findHeuristic(dir, heuristic)
                    priorityQ.put((priority, dir))

        return {"Status": "Path Not Found!!!", "Visited cells": cellsVisited,
                "No of visited cells": len(cellsVisited), "Path": [], "Path length": "N/A"}

    def findHeuristic(self, cell, heuristic):
        (x1, y1) = cell
        (x2, y2) = self.goalCell
        if heuristic == "euclidean":
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        elif heuristic == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)
        
def render_Maze():
    global current_map = Map(int(input_size.get_text()), float(input_prob.get_text()))
    update()
    
def update():


def render_handler(maze):
    current_map.visualize_maze(maze)

def input_handler():
    pass

def visualize_dfs():

def visualize_bfs():

def visualize_astar():



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
current_map.solution = FindSolution(current_map).biDirSearch()
current_map.print_solution()