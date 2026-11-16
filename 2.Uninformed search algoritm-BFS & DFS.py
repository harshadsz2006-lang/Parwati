# CELL 1: Import Libraries
import time
from collections import deque

# CELL 2: Define Maze
# Maze: 0 = open path, 1 = wall
maze = [
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]

start = (0, 0)
goal = (4, 4)
rows, cols = len(maze), len(maze[0])

maze

# CELL 3: Get Neighbors
def get_neighbors(cell):
    r, c = cell
    moves = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    return [(nr, nc) for nr, nc in moves
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0]

# CELL 4: Depth First Search (DFS)
def dfs(start, goal):
    stack = [[start]]
    visited = {start}
    nodes_explored = 0
    while stack:
        path = stack.pop()
        node = path[-1]
        nodes_explored += 1
        if node == goal:
            return path, nodes_explored
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(path + [neighbor])
    return None, nodes_explored

# CELL 5: Breadth First Search (BFS)
def bfs(start, goal):
    queue = deque([[start]])
    visited = {start}
    nodes_explored = 0
    while queue:
        path = queue.popleft()
        node = path[-1]
        nodes_explored += 1
        if node == goal:
            return path, nodes_explored
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None, nodes_explored

# CELL 6: Compare Performance
for name, algo in [("BFS", bfs), ("DFS", dfs)]:
    t0 = time.perf_counter()
    path, explored = algo(start, goal)
    t1 = time.perf_counter()
    print(f"{name}: path length={len(path)-1}, nodes explored={explored}, "
          f"time={(t1-t0)*1000:.4f} ms")
    print("Path:", path)
