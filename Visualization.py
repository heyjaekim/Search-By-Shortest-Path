import numpy
import queue
import math
from Map import Map
from Search import Search

try:
    import simplegui
except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

G_COLORS = [["#FFAA00", "#FFCDD2"], 
            ["#2196F3", "#BBDEFB"], 
            ["#4CAF50", "#C8E6C9"],
            ["#FFAA00", "#FFDEAD"], 
            ["#F540FF", "#F8BBEE", "#FFAA00", "#04db41"]]
color = []

def renderingMaze():
    global a_map
    a_map = Map(dim, p)
    update()

def drawingPath(maze):
    a_map.visualize_maze(maze, color)

def input_handler():
    pass

def pathDFS():
    global color
    color = G_COLORS[0]    
    a_map.results = Search(a_map).dfsAlgo()
    update()

def pathBFS():
    global color
    color = G_COLORS[1]
    a_map.results = Search(a_map).bfsAlgo()
    update()

def pathEuclidean():
    global color
    color = G_COLORS[2]
    a_map.results = Search(a_map).A_star("euclidean")
    update()

def pathManhattan():
    global color
    color = G_COLORS[3]
    a_map.results = Search(a_map).A_star("manhattan")
    update()

def path_biBFS():
    global color
    color = G_COLORS[4]
    a_map.results = Search(a_map).biBFS()
    update()

def path_IDFS():
    global color
    color = G_COLORS[0]
    a_map.results = Search(a_map).improvedDFS()
    update()

def update():
    currentStatus.set_text("PATH STATUS: " + a_map.results["Status"])
    currentLength.set_text("PATH LENGTH: " + str(a_map.results["Path length"]))
    currentVisitedCells.set_text("# OF VISITED CELLS: " + str(a_map.results["# of Visited Cells"]))
    if a_map.results["Intersecting Cell"]:
        currentIntersectiong.set_text("INTERSECTING CELL: " + str(a_map.results["Intersecting Cell"]))
    else:
        currentIntersectiong.set_text("INERSECTIING CELL: " )
width = 700
dim = 100
p = 0.2

a_map = Map(dim, p)
# Create frame and control UI
frame = simplegui.create_frame('Part 1', width, width)
frame.set_draw_handler(drawingPath)
frame.add_button("Rendering Maze", renderingMaze, 100)
frame.add_label("")
frame.add_label("Dimension Size: "+str(dim))
frame.add_label("Probability: " + str(p))


# Algorithms
frame.add_label("")
frame.add_label("Algorithm")
frame.add_button("DFS", pathDFS, 100)
frame.add_button("BFS", pathBFS, 100)
frame.add_button("A* Euclidean", pathEuclidean, 100)
frame.add_button("A* Manhattan", pathManhattan, 100)
frame.add_button("BI BFS", path_biBFS, 100)
frame.add_button("IDFS", path_IDFS, 100)

# Display status
frame.add_label("")
currentStatus = frame.add_label("PATH STATUS: N/A")
currentLength = frame.add_label("PATH LENGTH: N/A")
currentVisitedCells = frame.add_label("# OF VISITED CELLS: N/A")
currentIntersectiong = frame.add_label("INTERSECTING CELL: N/A")

frame.start()