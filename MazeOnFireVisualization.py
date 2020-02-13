import numpy
import queue
import math
from MazeOnFire import Search

try:
    import simplegui

except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

FRAME_WIDTH = 700
DEFAULT_SIZE = 100
DEFAULT_PROB = 0.2
G_COLORS = [["#F540FF", "#F8BBEE", "#FFAA00" ]]
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

        # fix value for start and end
        self.map[0, 0] = 0
        self.map[size - 1, size - 1] = 0

        # Dictionary of results information
        self.results = {"Status": "n/a",
                         "Visited cells": [],
                         "# of Visited Cells": "n/a",
                         "Path": [],
                         "Path from Goal": [],
                         "Path length": "n/a",
                         "Path length from Goal": "n/a",
                         "Intersecting Cell": (),
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

    def print_results(self):
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

global a_map 
def generate_map():
    a_map = Map(int(input_size.get_text()), float(input_probability.get_text()))
    update()

def generate_path(maze):
    a_map.visualize_maze(maze, color, FRAME_WIDTH)

def input_handler():
    pass

def pathFireOne():
    global color
    color = G_COLORS[0]    
    a_map.results = Search(a_map).fireStrategyOne()
    currentAlgo.set_text("Algorithm used: Maze On Fire Strategy 1")
    update()

def update():
    currentStatus.set_text("STATUS: " + a_map.results["Status"])
    currentLength.set_text("PATH LENGTH: " + str(a_map.results["Path length"]))
    currentVisitedCells.set_text("NO OF VISITED CELLS: " + str(a_map.results["# of Visited Cells"]))

a_map = Map(DEFAULT_SIZE, DEFAULT_PROB)
fireStart = (0, DEFAULT_SIZE-1)
fireGoal = (DEFAULT_SIZE-1, 0)

fire_start = (0, DEFAULT_SIZE-1)
fire_goal = (DEFAULT_SIZE-1, 0)
# Create frame and control UI
frame = simplegui.create_frame('Assignment 1', FRAME_WIDTH, FRAME_WIDTH)
frame.add_button("Generate Map", generate_map, 100)
frame.set_draw_handler(generate_path)
input_size = frame.add_input('Size',input_handler, 100)
input_size.set_text(str(DEFAULT_SIZE))
input_probability = frame.add_input("Probability", input_handler, 100)
input_probability.set_text(str(DEFAULT_PROB))

start_time = time.time()

dim = 100
p = 0.3
start = (0,0)
goal = (dim - 1, dim - 1)
fire_start = (0,dim-1)
fire_goal = (dim-1,0)

is_goal_reached = 0
is_fire_reached = 0

while not is_goal_reached or not is_fire_reached:
  maze = generate_maze(dim, p)
  #is_goal_reached, prev_list_path, count_of_nodes_path, max_fringe_size_path, visited_path = A_star(maze, start, goal , "manhattan")
  is_fire_reached, prev_list_fire, count_of_nodes_fire, max_fringe_size_fire, visited_fire = A_star(maze, fire_start, fire_goal, "manhattan")

is_reached, runner_location, prev_list, count_of_nodes, max_fringe_size, fire_cells = fire_cell_search(maze, start, goal, fire_start, fire_goal, 0.3)

maze_temp = maze*100
display_path(maze_temp, prev_list_path, start, goal)

print("Time taken: " + str(time.time()-start_time))

frame.add_button("Maze On Fire Strategy 1", pathFireOne, 100)
#frame.add_button("Maze On Fire Strategy 2", pathFireTwo, 100)
#frame.add_button("Maze On Fire Strategy 3", pathFireThree, 100)
'''
# Algorithms
frame.add_label("")
frame.add_label("Algorithm")
frame.add_button("DFS", path_dfs, 100)
frame.add_button("BFS", path_bfs, 100)
frame.add_button("A* Euclidean", path_euclidean, 100)
frame.add_button("A* Manhattan", path_manhattan, 100)
frame.add_button("BI BFS", path_biBFS, 100)
'''

# Display status
frame.add_label("")
currentAlgo = frame.add_label("Algorithm used: N/A")
currentStatus = frame.add_label("STATUS: N/A")
currentVisitedCells = frame.add_label("NO OF VISITED CELLS: N/A")
currentLength = frame.add_label("PATH LENGTH: N/A")

frame.start()