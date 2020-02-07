import numpy
import queue
import math
try:
    import simplegui
except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

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

        # fix value for start and end
        self.map[0, 0] = 0
        self.map[size - 1, size - 1] = 0

        # Dictionary of results information
        self.results = {"Status": "",
                         "Visited cells": [],
                         "# of Visited Cells": "",
                         "Path": [],
                         "Path from Goal": [],
                         "Path length": "",
                         "Path length from Goal": "",
                         "Intersecting Cell": (),
                         "Max fringe size": ""}

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
        print("Status: " + self.results["Status"])
        print("# of Visited Cells: " +
              str(self.results["# of Visited Cells"]))
        print("Path length: " + str(self.results["Path length"]))
        print("Max fringe size: " + str(self.results["Max fringe size"]))

    def visualize_maze(self, maze, color, framewidth):
        width = framewidth / self.size
        for x in range(0, self.size):
            for y in range(0, self.size):
                points = [(x * width, y * width), ((x + 1) * width, y * width),
                          ((x + 1) * width, (y + 1) * width), ((x * width), (y + 1) * width)]

                if(x, y) == (0, 0) or (x, y) == (self.size-1, self.size-1):
                    maze.draw_polygon(points, 1, "Black", "00FF00")

                elif (x, y) in self.results["Path"]:
                    maze.draw_polygon(points, 1, "Black", color[0])

                elif (x, y) in self.results["Path from Goal"]:
                    maze.draw_polygon(points, 1, "Black", color[2])

                elif (x, y) in self.results["Intersecting Cell"]:
                    maze.draw_polygon(points, 1, "Black", color[3])

                elif (x, y) in self.results["Visited cells"] and (x, y) not in self.results["Path"] and (x, y) not in self.results["Path from Goal"] and (x, y) not in self.results["Intersecting Cell"]:
                    maze.draw_polygon(points, 1, "Black", color[1])

                elif self.map[x, y] == 0:
                    maze.draw_polygon(points, 1, "Black", "White")

                else:
                    maze.draw_polygon(points, 1, "Black", "#464646")
