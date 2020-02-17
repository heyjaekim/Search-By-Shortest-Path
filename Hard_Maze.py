import Map from Map
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import random
import heapq
import bfsAlgo from Search


class HardMaze():
# These utilities might not be useful.


class Node:
 def __init__(self, i=0, j=0):
        self.i = i
        self.j = j
        self.loc = (self.i, self.j)


class Stack:
    def __init__(self):
        self.stack = []

    def is_empty(self):
        return len(self.stack) == 0

    def pop(self):
        return self.stack.pop()

    def push(self, node):
        self.stack.append(node)

    def top(self):
        return self.stack[-1]


class Node:
    def __init__(self, i=0, j=0):
        self.i = i
        self.j = j
        self.loc = (self.i, self.j)
        # self.loc=[i,j]


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.index = 0

    def push(self, node, priority, path):
        heapq.heappush(self.heap, (priority, self.index, node, path))
        self.index -= 1

    def pop(self):
        return heapq.heappop(self.heap)

    def is_empty(self):
     return self.heap == []


def manhattan_distance(start, goal):
        return abs(goal.i - start.i) + abs(goal.j - start.j)


def astarLength(maze, start=Node(1, 1), heuristic=manhattan_distance):
    goal = Node(len(maze)-2, len(maze)-2)
    mp = maze.copy()
    pq = PriorityQueue()
    pq.push(node=start, priority=heuristic(start, goal), path=[start])
    dirArr = [[1, 0], [0, 1], [0, -1], [-1, 0]]


def bfs(maze):
    start = Node(1, 1)
    goal = Node(len(maze)-2, len(maze)-2)
    mp = maze.copy()
    # queue
    qu = deque()
    # path
    path = []
    dirArr = [[1, 0], [0, 1], [0, -1], [-1, 0]]  # 4 directions

    qu.append((start, 0))
    while len(qu) != 0:
        curNode = qu.popleft()
        path.append(curNode)
        if curNode[0].i == goal.i and curNode[0].j == goal.j:
            output = get_path(maze, path)
            return output
            # return lenghth(output)
        for dirMove in dirArr:
            nextNode = Node(curNode[0].i+dirMove[0], curNode[0].j+dirMove[1])
            if mp[nextNode.loc] == 0 or mp[nextNode.loc] == 1:
                qu.append((nextNode, len(path)))
                mp[nextNode.loc] = 3
    return False

#This is the code to visualize the maze.
  def visualize_Maze(self,maze=None):
        if maze is not None:
            drawMap=maze
        else:
            drawMap=self.mazeMap
            
        plt.figure(figsize=(10,10))
        plt.pcolor(drawMap[::-1],edgecolors='black',cmap='white', linewidths=5)
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()

	# We used a genetic algorithm here
def hard_maze_simulated_annealing(old_maze,new_maze,cur_distance):
  #We need to create a baseline for comparison. 
	original_maze[1,1]=original_maze[10,10]=0

	for i in range(300):
	#We are doing this for 300 cycles in this case
	new_maze=original_maze.copy()
	
	recomb_maze=new_maze[1:10,1:10]
	recomb_maze=recomb_maze.flatten()
	random.shuffle(recomb_maze)
	recomb_maze=recomb_maze.reshape((10,10))
	new_maze[1:10,1:10]=recomb_maze
	recomb_maze[new_maze]=old_maze[new_maze]
	new_maze[1,1]=new_maze[10,10]=1
	
	newDist=astarLength(new_maze)
		if(newDist>cur_distance):
			return new_maze
		
		old_maze[1,1]=original_maze[10,10]=1
		

	return old_maze

	visual=visualize_Maze(10,0.3) 
	

	x=astarLength(visual,manhattan_distance)

	if(a==true):
	visualize_Maze(bfs(x))
	visualize_Maze(bfs(visual)







    

   
    
    
 
 
  
  





