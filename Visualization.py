import numpy
import queue
import math
"jaeweon Kim"
from Map import Map
from Search import FindSolution

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
                    
def generate_map():
    global current_map
    current_map = Map(int(input_size.get_text()), float(input_probability.get_text()))
    update()


def draw_handler(maze):
    current_map.visualize_maze(maze, color)


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