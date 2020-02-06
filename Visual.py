import numpy
import queue
import math
from Map import Map

def render_Maze():
    global current_map
    current_map = Map(int(input_size.get_text()), float(input_prob.get_text()))
    update()

#def update():


def render_handler(maze):
    current_map.visualize_maze(maze)

def input_handler():
    pass

#def visualize_dfs():

#def visualize_bfs():

#def visualize_astar():