import numpy
import queue
import math
import time
from Map import Map

class Search:
    def __init__(self, maze):
        self.map = maze
        self.startCell = (0, 0)
        self.goalCell = (maze.size - 1, maze.size - 1)

    def render_path(self, pathDict):
        # Build a path (list) from a dictionary of parent cell -> child cell
        current = pathDict[self.goalCell]
        path = []
        while current != (0, 0):
            path += (current,)
            current = pathDict[current]
        return path

    def render_path_bidir(self, pathDict, position):
        current = pathDict[position]
        path = []
        while current != (0, 0):
            path += (current,)
            current = pathDict[current]
        return path

    def render_path_bidir_fromEnd(self, pathDict, position):
        current = pathDict[position]
        path = []
        while current != (self.map.size-1, self.map.size-1):
            path += (current,)
            current = pathDict[current]
        return path[::+1]

    def dfsAlgo(self):
        dfsStack = queue.LifoQueue()
        dfsStack.put(self.startCell)
        parentDict = {}
        cellsVisited = set()
        cellsVisited.add(self.startCell)
        maxFringe = 0

        while not dfsStack.empty():
            cell = dfsStack.get()
            maxFringe = max(maxFringe, dfsStack.qsize())
            if cell == self.goalCell:
                path = self.render_path(parentDict)
                return {"Status": "Found Path", "Visited cells": cellsVisited,
                        "# of Visited Cells": len(cellsVisited), "Path length": len(path), "Path length from Goal": "", 
                        "Path": path, "Path from Goal": [], "Intersecting Cell": (),
                        "Max fringe size": (maxFringe)}

            for dir in self.map.neighborCells(cell):
                if dir not in cellsVisited:
                    parentDict[dir] = cell
                    cellsVisited.add(dir)
                    dfsStack.put(dir)

        return {"Status": "Unable to find the path", "Visited cells": cellsVisited,
                "# of Visited Cells": len(cellsVisited), "Path length": "N/A", "Path length from Goal": "", 
                "Path": [], "Path from Goal": [], "Intersecting Cell": (),
                "Max fringe size": "N/A"}
        

    def improvedDFS(self):
        fringe = list()
        fringe.append(self.startCell)
        visitedCells = list()
        visitedCells.append(self.startCell)
        
        parentDict = {}
        maxFringe = 0

        while fringe:
            (i, j) = fringe.pop()
            maxFringe = max(maxFringe, len(fringe))
            if (i, j) == self.goalCell:
                path = self.render_path(parentDict)
                return {"Status": "Found Path", "Visited cells": visitedCells,
                        "# of Visited Cells": len(visitedCells), "Path": path, "Path length": len(path), "Path length from Goal": "", 
                        "Path from Goal": [], "Intersecting Cell": (),
                        "Max fringe size": (maxFringe)}
            children = self.map.prioritization(self.goalCell[0] - 1, visitedCells, i, j)
            if children:
                for c in children:
                    #if dir not in cellsVisited:
                    parentDict[c] = (i, j)
                    visitedCells.append(c)
                    fringe.append(c)

        return {"Status": "Unable to find the path", "Visited cells": visitedCells,
                "# of Visited Cells": len(visitedCells), "Path length": "N/A", "Path length from Goal": "", 
                "Path": [], "Path from Goal": [], "Intersecting Cell": (),
                "Max fringe size": "N/A"}

    def bfsAlgo(self):

        stack = queue.Queue()
        visitedCells = set()
        parentDict = {}
        stack.put(self.startCell)
        visitedCells.add(self.startCell)
        maxFringe = 0

        while not stack.empty():
            cell = stack.get()
            maxFringe = max(maxFringe, stack.qsize())
            if cell == self.goalCell:
                path = self.render_path(parentDict)
                return {"Status": "Found Path", "Visited cells": visitedCells,
                "# of Visited Cells": len(visitedCells), "Path": path, "Path length": len(path), "Path length from Goal": "", 
                        "Path from Goal": [], "Intersecting Cell": (),
                        "Max fringe size": (maxFringe)}

            for dir in self.map.neighborCells(cell):
                if dir not in visitedCells:
                    parentDict[dir] = cell
                    visitedCells.add(dir)
                    stack.put(dir)

        return {"Status": "Unable to find the path", "Visited cells": visitedCells,
                "# of Visited Cells": len(visitedCells), "Path length": "N/A", "Path length from Goal": "N/A", 
                "Path": [], "Path from Goal": [], "Intersecting Cell": (),
                "Max fringe size": "n/a"}

    def isIntersecting(self, intersectionNode, s_fringe, t_fringe):
        if intersectionNode == (-1, -1):
            for c in s_fringe:
                if(c in t_fringe):
                    return c
            return (-1, -1)

        else:
            return intersectionNode

    def biBFS(self):

        s_visited = list()
        t_visited = list()
        s_visited.append(self.startCell)
        t_visited.append(self.goalCell)

        s_fringe = list()
        t_fringe = list()
        s_fringe.append(self.startCell)
        t_fringe.append(self.goalCell)

        s_parent = dict()
        t_parent = dict()

        intersectNode = (-1, -1)
        maxFringe = 0

        while (s_fringe and t_fringe):
            s_cell = s_fringe.pop(0)
            t_cell = t_fringe.pop(0)
            maxFringe = max(maxFringe, max(len(s_fringe), len(t_fringe)))

            for dir in self.map.neighborCells(s_cell):
                if dir not in s_visited:
                    s_parent[dir] = s_cell
                    s_visited.append(dir)
                    s_fringe.append(dir)

            intersectNode = Search.isIntersecting(self.map, intersectNode, s_fringe, t_fringe)

            for dir in self.map.neighborCells(t_cell):
                if dir not in t_visited:
                    t_parent[dir] = t_cell
                    t_visited.append(dir)
                    t_fringe.append(dir)

            intersectNode = Search.isIntersecting(self.map, intersectNode, s_fringe, t_fringe)

            if intersectNode != (-1, -1):
                print("intersecting at cell:", intersectNode)
                path = self.render_path_bidir(s_parent, intersectNode)
                path2 = self.render_path_bidir_fromEnd(t_parent, intersectNode)
                s_visited.extend(t_visited)
                intersection = []
                intersection.append(intersectNode)
                return {"Status": "Found Path", "Visited cells": s_visited,
                        "# of Visited Cells": len(s_visited), "Path length": len(path), "Path length from Goal": len(path2), 
                        "Path": path, "Path from Goal": path2, "Intersecting Cell": intersection,
                        "Max fringe size": (maxFringe)}

        return {"Status": "Unable to find the path", "Visited cells": s_visited,
                "# of Visited Cells": len(s_visited), "Path length": "N/A", "Path length from Goal": "N/A", 
                "Path": [], "Path from Goal": [], "Intersecting Cell": (),
                "Max fringe size": "n/a"}

    def A_star(self, heuristic):
        """
        this is for finding the path for A-Star
        return the list of path status, # of visted cells, path, and path length
        """
        priorityQ = queue.PriorityQueue()
        parentDict = {}
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
                path = self.render_path(parentDict)
                return {"Status": "Found Path", "Visited cells": cellsVisited,
                        "# of Visited Cells": len(cellsVisited), 
                        "Path": path, "Path length": len(path), "Path length from Goal": "", 
                        "Path from Goal": [], "Intersecting Cell": (),
                        "Max fringe size": (maxFringe)}

            for dir in self.map.neighborCells(cell):
                newCost = mincost[cell] + 1
                # or new_cost < cost_so_far[next_cell]:
                if dir not in cellsVisited:
                    mincost[dir] = newCost
                    parentDict[dir] = cell
                    cellsVisited.add(dir)

                    priority = newCost + self.findHeuristic(dir, heuristic)
                    priorityQ.put((priority, dir))

        return {"Status": "Unable to find the path", "Visited cells": cellsVisited,
                "# of Visited Cells": len(cellsVisited), 
                "Path length": "N/A", "Path length from Goal": "N/A", 
                "Path": [], "Path from Goal": [], "Intersecting Cell": (),
                "Max fringe size": "n/a"}

    def findHeuristic(self, cell, heuristic):
        (x1, y1) = cell
        (x2, y2) = self.goalCell
        if heuristic == "euclidean":
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        elif heuristic == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)