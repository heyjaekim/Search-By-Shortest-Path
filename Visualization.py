import numpy
import queue
import math
from Map import Map
from Search import Search

try:
    import simplegui

except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

FRAME_WIDTH = 700
DEFAULT_SIZE = 100
DEFAULT_PROB = 0.2
G_COLORS = [["#F44336", "#FFCDD2"], ["#2196F3", "#BBDEFB"], ["#4CAF50", "#C8E6C9"],
              ["#FFAA00", "#FFDEAD"], ["#F540FF", "#F8BBEE", "#2DF7D3", "FFF700"]]
color = []
global a_map 

def generate_map():
    global a_map
    a_map = Map(int(input_size.get_text()), float(input_probability.get_text()))
    update()

def generate_path(maze):
    a_map.visualize_maze(maze, color)

def input_handler():
    pass

def path_dfs():
    global color
    color = G_COLORS[0]    
    a_map.results = Search(a_map).dfs_algo()
    currentAlgo.set_text("Algorithm used: DFS")
    update()

def path_bfs():
    global color
    color = G_COLORS[1]
    a_map.results = Search(a_map).bfs_algo()
    currentAlgo.set_text("Algorithm used: BFS")
    update()

def path_euclidean():
    global color
    color = G_COLORS[2]
    a_map.results = Search(a_map).a_star_algo("euclidean")
    currentAlgo.set_text("Algorithm used: A* Euclidean")
    update()

def path_manhattan():
    global color
    color = G_COLORS[3]
    a_map.results = Search(a_map).a_star_algo("manhattan")
    currentAlgo.set_text("Algorithm used: A* Manhattan")
    update()

def path_biBFS():
    global color
    color = G_COLORS[4]
    a_map.results = Search(a_map).biBFS()
    currentAlgo.set_text("Algorithm used: BI BFS")
    update()

def update():
    currentStatus.set_text("STATUS: " + a_map.results["Status"])
    currentLength.set_text("PATH LENGTH: " + str(a_map.results["Path length"]))
    currentVisitedCells.set_text("NO OF VISITED CELLS: " + str(a_map.results["# of Visited Cells"]))

a_map = Map(DEFAULT_SIZE, DEFAULT_PROB)

# Create frame and control UI
frame = simplegui.create_frame('Assignment 1', FRAME_WIDTH, FRAME_WIDTH)
frame.add_button("Generate Map", generate_map, 100)
frame.set_draw_handler(generate_path)
input_size = frame.add_input('Size',input_handler, 100)
input_size.set_text(str(DEFAULT_SIZE))
input_probability = frame.add_input("Probability", input_handler, 100)
input_probability.set_text(str(DEFAULT_PROB))

# Algorithms
frame.add_label("")
frame.add_label("Algorithm")
frame.add_button("DFS", path_dfs, 100)
frame.add_button("BFS", path_bfs, 100)
frame.add_button("A* Euclidean", path_euclidean, 100)
frame.add_button("A* Manhattan", path_manhattan, 100)
frame.add_button("BI BFS", path_biBFS, 100)

# Display status
frame.add_label("")
currentAlgo = frame.add_label("Algorithm used: N/A")
currentStatus = frame.add_label("STATUS: N/A")
currentVisitedCells = frame.add_label("NO OF VISITED CELLS: N/A")
currentLength = frame.add_label("PATH LENGTH: N/A")

frame.start()