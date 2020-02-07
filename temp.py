import numpy
import queue
import math
try:
    import simplegui
except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

FRAME_WIDTH = 700
DEFAULT_SIZE = 100
DEFAULT_PROB = 0.2
COLOR_LIST = [["#F44336", "#FFCDD2"], ["#2196F3", "#BBDEFB"], ["#4CAF50", "#C8E6C9"],
              ["#FF9800", "#FFE0B2"], ["#E91E63", "#F8BBD0"], ["#9C27B0", "#E1BEE7"]]
color = []

class Map:
    def __init__(self, size, prob):
        """Rendering a new map 
        - size of map 10 or above.
        - probability(double) is from 0.0 to 1.0
        - 0's all empty cells, 1's all walls
        """
        self.prob = prob
        self.size = size  

        # Check if arguments are valid
        if prob < 0 or prob > 1:
            raise ValueError("probability should be in range 0 and 1")

        # Generate a matrix that is uniformly distrubited
        self.map = numpy.random.uniform(size=[size, size])
        self.map = (self.map < prob).astype(int)
        
        #fix value for start and end
        self.map[0, 0] = 0
        self.map[size - 1, size - 1] = 0

        # Dictionary of solution information
        self.solution = {"Status": "n/a", 
                         "Visited cells": [],
                         "No of visited cells": "n/a",
                         "Path": [],
                         "Path length": "n/a",
                         "Max fringe size": "n/a"}

    def neighborCells(self, cell):
        
        x = cell[0]
        y = cell[1]
        neighborCells = set()
        DIRS = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        for (i, j) in DIRS:
            if self.in_bounds((i, j)):
                if self.map[i, j] == 0:
                    neighborCells.add((i, j))
        return neighborCells

    def in_bounds(self, cell):
        # Check if a cell is in the map
        x = cell[0]
        y = cell[1]
        return (0 <= x < self.size) and (0 <= y < self.size)

    def print_map(self):
        print(self.map)

    def print_solution(self):
        print("Status: " + self.solution["Status"])
        print("No of visited cells: " + str(self.solution["No of visited cells"]))
        print("Path length: " + str(self.solution["Path length"]))
        print("Max fringe size: "  + str(self.solution["Max fringe size"]))

    def visualize_maze(self, maze):
        width = FRAME_WIDTH / self.size
        for x in range(0, self.size):
            for y in range(0, self.size):
                points = [(x * width, y * width), ((x + 1) * width, y * width),
                          ((x + 1) * width, (y + 1) * width), ((x * width), (y + 1) * width)]
                
                if(x,y) == (0,0) or (x,y) == (self.size-1, self.size-1):
                    maze.draw_polygon(points, 1, "Black", "00FF00")
                
                elif (x,y) in self.solution["Path"]:
                    maze.draw_polygon(points, 1, "Black", color[0])
                
                elif (x, y) in self.solution["Visited cells"] and (x, y) not in self.solution["Path"]:
                    maze.draw_polygon(points, 1, "Black", color[1])
                
                elif self.map[x,y] == 0:
                    maze.draw_polygon(points, 1, "Black", "White")
                
                else:
                    maze.draw_polygon(points, 1, "Black", "#464646")

class FindSolution:
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
                        "No of visited cells": len(cellsVisited), "Path": path, "Path length": len(path)}

            for dir in self.grid.neighborCells(cell):
                if dir not in cellsVisited:
                    prev_cells[dir] = cell
                    cellsVisited.add(dir)
                    dfsStack.put(dir)

        return {"Status": "Path Not Found!!!", "Visited cells": cellsVisited,
                "No of visited cells": len(cellsVisited), "Path length": "N/A", "Path": []}

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
                "No of visited cells": len(visitedCells), "Path": path,
                        "Path length": len(path), "Max fringe size": (maxFringe)}

            for dir in self.grid.neighborCells(cell):
                if dir not in visitedCells:
                    parentSet[dir] = cell
                    visitedCells.add(dir)
                    stack.put(dir)

        return {"Status": "Path Not Found!!!", "Visited cells": visitedCells,
                "No of visited cells": len(visitedCells), "Path length": "N/A", "Path": [],
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

            intersectNode = FindSolution.isIntersecting(intersectNode, s_fringe, t_fringe)

            for dir in self.grid.neighborCells(t_cell):
                if dir not in t_visited:
                    t_parent[dir] = t_cell
                    t_visited.append(dir)
                    t_fringe.append(dir)

            intersectNode = FindSolution.isIntersecting(intersectNode, s_fringe, t_fringe)

            if intersectNode != (-1, -1):
                print("intersecting at cell:", intersectNode)
                path = self.render_path_bidir(s_parent, intersectNode)
                return {"Status": "Found Path", "Visited cells": s_visited,
                        "No of visited cells": len(s_visited), "Path length": len(path), "Path": path,
                        "Max fringe size": (maxFringe)}

        return {"Status": "Path Not Found!!!", "Visited cells": s_visited,
                "No of visited cells": len(s_visited), "Path length": "N/A", "Path": [],
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
                        "No of visited cells": len(cellsVisited), "Path length": len(path), "Path": path,
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

        return {"Status": "Path Not Found!!!", "Visited cells": cellsVisited,
                "No of visited cells": len(cellsVisited), "Path length": "N/A", "Path": [],
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

                    
def generate_map():
    global current_map
    current_map = Map(int(input_size.get_text()), float(input_probability.get_text()))
    update()


def draw_handler(maze):
    current_map.visualize_maze(maze)


def input_handler():
    pass


def draw_dfs():
    global current_map, color
    color = COLOR_LIST[0]
    current_map.solution = FindSolution(current_map).dfs_algo()
    algorithm_used.set_text("Algorithm used: DFS")
    update()


def draw_bfs():
    global current_map, color
    color = COLOR_LIST[1]
    current_map.solution = FindSolution(current_map).bfs_algo()
    algorithm_used.set_text("Algorithm used: BFS")
    update()


def draw_euclidean():
    global current_map, color
    color = COLOR_LIST[2]
    current_map.solution = FindSolution(current_map).a_star_algo("euclidean")
    algorithm_used.set_text("Algorithm used: A* Euclidean")
    update()


def draw_manhattan():
    global current_map, color
    color = COLOR_LIST[3]
    current_map.solution = FindSolution(current_map).a_star_algo("manhattan")
    algorithm_used.set_text("Algorithm used: A* Manhattan")
    update()

def draw_biBFS():
    global current_map, color
    color = COLOR_LIST[4]
    current_map.solution = FindSolution(current_map).biBFS()
    algorithm_used.set_text("Algorithm used: BI BFS")
    update()


def update():
    status_label.set_text("STATUS: " + current_map.solution["Status"])
    path_length_label.set_text("PATH LENGTH: " + str(current_map.solution["Path length"]))
    no_of_visited_cells_label.set_text("NO OF VISITED CELLS: " + str(current_map.solution["No of visited cells"]))

current_map = Map(DEFAULT_SIZE, DEFAULT_PROB)

# Create frame and control UI
frame = simplegui.create_frame('Assignment 1', FRAME_WIDTH, FRAME_WIDTH)
frame.add_button("Generate Map", generate_map, 100)
frame.set_draw_handler(draw_handler)
input_size = frame.add_input('Size', input_handler, 50)
input_size.set_text(str(DEFAULT_SIZE))
input_probability = frame.add_input("Probability", input_handler, 50)
input_probability.set_text(str(DEFAULT_PROB))

# Algorithms
frame.add_label("")
frame.add_label("Algorithm")
frame.add_button("DFS", draw_dfs, 100)
frame.add_button("BFS", draw_bfs, 100)
frame.add_button("A* Euclidean", draw_euclidean, 100)
frame.add_button("A* Manhattan", draw_manhattan, 100)
#frame.add_button("BI BFS", draw_biBFS, 100)

# Display status
frame.add_label("")
algorithm_used = frame.add_label("Algorithm used: N/A")
status_label = frame.add_label("STATUS: N/A")
no_of_visited_cells_label = frame.add_label("NO OF VISITED CELLS: N/A")
path_length_label = frame.add_label("PATH LENGTH: N/A")

frame.start()