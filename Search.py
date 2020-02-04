import numpy
import queue
import math
from Map import Map


class FindSolution:
    def __init__(self, a_map):
        self.a_map = a_map
        self.time = 0
        self.result = "N/A"
        self.start_position = (0, 0)
        self.end_position = (a_map.size - 1, a_map.size - 1)

    def renderPath(self, pathTrack):
        #Build a path (list) from a dictionary of parent cell -> child cell
        path = []
        current = pathTrack[self.end_position]
        while current != (0, 0):
            path += (current,)
            current = pathTrack[current]
        return path[::-1]

    def dfs(self):
        
        s_stack = queue.LifoQueue()
        visitedCells = set()

        s_stack.put(self.start_position)
        visitedCells.add(self.start_position)

        parent_cell = {}

        while not s_stack.empty():
            cell = s_stack.get()
            if cell == self.end_position:
                path = self.renderPath(parent_cell)
                return {"Status": "Found Path", "Visited cells": visitedCells,
                        "No of visited cells": len(visitedCells), "Path": path, "Path length": len(path)}

            for child in self.a_map.neighborCells(cell):
                if child not in visitedCells:
                    parent_cell[child] = cell
                    visitedCells.add(child)
                    s_stack.put(child)

        return {"Status": "Path Not Found!!!", "Visited cells": visitedCells,
                "No of visited cells": len(visitedCells), "Path": [], "Path length": "N/A"}

    def bfs(self):
        
        s_queue = queue.Queue()
        visitedCells = set()

        s_queue.put(self.start_position)
        visitedCells.add(self.start_position)

        parent_cell = {}

        while not s_queue.empty():
            cell = s_queue.get()
            if cell == self.end_position:
                path = self.renderPath(parent_cell)
                return {"Status": "Found Path", "Visited cells": visitedCells,
                        "No of visited cells": len(visitedCells), "Path": path, "Path length": len(path)}

            for child in self.a_map.neighborCells(cell):
                if child not in visitedCells:
                    parent_cell[child] = cell
                    visitedCells.add(child)
                    s_queue.put(child)

        return {"Status": "Path Not Found!!!", "Visited cells": visitedCells,
                "No of visited cells": len(visitedCells), "Path": [], "Path length": "N/A"}

    def biDirectionalBFS(self):

        s_queue = queue.Queue()
        t_queue = queue.Queue()

        visitedCells = set()

        s_queue.put(self.start_position)
        visitedCells.add(self.start_position)
        t_queue.put(self.end_position)
        visitedCells.add(self.end_position)

        parent_cell = {}

        while not s_queue.empty():
            s_cell = s_queue.get()
            t_cell = t_queue.get()
            if s_cell == t_cell:
                path = self.renderPath(parent_cell)
                return {"Status": "Found Path", "Visited cells": visitedCells,
                        "No of visited cells": len(visitedCells), "Path": path, "Path length": len(path)}
            
            for child in self.a_map.neighborCells(s_cell):
                if child not in visitedCells:
                    parent_cell[child] = s_cell
                    visitedCells.add(child)
                    s_queue.put(child)
            return {"Status": "Found Path", "Visited cells": visitedCells,
                        "No of visited cells": len(visitedCells), "Path": [], "Path length": "N/A"}

    def a_star(self, heuristic):
        priorityQ = queue.PriorityQueue()
        visitedCells = set()
        parentCell = {}
        cost = {}

        priorityQ.put((0, self.start_position))
        visitedCells.add(self.start_position)
        cost[self.start_position] = 0

        while not priorityQ.empty():
            currentCell = priorityQ.get()
            currentCell = currentCell[1]
            if currentCell == self.end_position:
                path = self.renderPath(parentCell)
                return {"Status": "Found Path", "Visited cells": visitedCells,
                        "No of visited cells": len(visitedCells), "Path": path, "Path length": len(path)}

            for next_cell in self.a_map.neighborCells(currentCell):
                new_cost = cost[currentCell] + 1
                if next_cell not in visitedCells:  # or new_cost < cost_so_far[next_cell]:
                    cost[next_cell] = new_cost
                    parentCell[next_cell] = currentCell
                    visitedCells.add(next_cell)

                    priority = new_cost + self.findHeuristic(next_cell, heuristic)
                    priorityQ.put((priority, next_cell))

        return {"Status": "Path Not Found!!!", "Visited cells": visitedCells,
                "No of visited cells": len(visitedCells), "Path": [], "Path length": "N/A"}

    def findHeuristic(self, cell, heuristic):
        (x1, y1) = cell
        (x2, y2) = self.end_position
        if heuristic == "euclidean":
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        elif heuristic == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)
        
    
current_map = Map(100, 0.0)

print("--------------------------------\nUsing DFS")
current_map.solution = FindSolution(current_map).dfs()
current_map.print_solution()
print("--------------------------------\nUsing BFS")
current_map.solution = FindSolution(current_map).bfs()
current_map.print_solution()
print("--------------------------------\nUsing A* Euclidean")
current_map.solution = FindSolution(current_map).a_star("euclidean")
current_map.print_solution()
print("--------------------------------\nUsing A* Manhattan")
current_map.solution = FindSolution(current_map).a_star("manhattan")
current_map.print_solution()
print("--------------------------------\nUsing Bi Directional BFS")
current_map.solution = FindSolution(current_map).biDirectionalBFS()
current_map.print_solution()