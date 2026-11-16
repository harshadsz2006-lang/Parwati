import time
from collections import deque

# Maze: 0 = open path, 1 = wall
maze = [
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]
start, goal = (0, 0), (4, 4)
rows, cols = len(maze), len(maze[0])


def get_neighbors(cell):
    r, c = cell
    moves = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [(nr, nc) for nr, nc in moves
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0]


def search(start, goal, depth_first):
    """DFS takes paths from the end of the frontier (stack); BFS from the front (queue)."""
    frontier = deque([[start]])
    visited = {start}
    explored = 0
    while frontier:
        path = frontier.pop() if depth_first else frontier.popleft()
        node = path[-1]
        explored += 1
        if node == goal:
            return path, explored
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(path + [neighbor])
    return None, explored


for name, depth_first in [("BFS", False), ("DFS", True)]:
    t0 = time.perf_counter()
    path, explored = search(start, goal, depth_first)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{name}: path length={len(path) - 1}, nodes explored={explored}, time={ms:.4f} ms")
    print("Path:", path)
