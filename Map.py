import numpy
import queue
import math
try:
    import simplegui
except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

FRAME_WIDTH = 700

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

    def visualize_maze(self, maze, color):
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
                