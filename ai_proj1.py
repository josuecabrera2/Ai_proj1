# import math  
# import heapq

# class Node:
#     def __init__(self, position, g=0, h=0, parent=None, parent_move=None):
#         self.position = position  # Position of the node in the grid as (row, col)
#         self.g = g  # Cost from the start node to this node
#         self.h = h  # Heuristic (estimated cost) to reach the goal from this node
#         self.f = g + h  # Total cost (g + h) to determine the node's priority in A*
#         self.parent = parent  # Reference to the previous node in the path
#         self.parent_move = parent_move

#     def __lt__(self, other):
#         return self.f < other.f


# def parse_input(file_path):
#     with open(file_path, 'r') as file:
#         start_x, start_y, goal_x, goal_y = map(int, file.readline().split())
#         start = (start_y, start_x)
#         goal = (goal_y, goal_x)
        
#         grid = []
#         for line in file:
#             grid.append(list(map(int, line.split())))
#     return start, goal, grid


# def get_neighbors(node, grid):
#     directions = [
#         (1, 0), (1, 1), (0, 1), (-1, 1),
#         (-1, 0), (-1, -1), (0, -1), (1, -1)
#     ]
#     neighbors = []
#     for i, (dy, dx) in enumerate(directions):
#         ny, nx = node.position[0] + dy, node.position[1] + dx
#         if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) and grid[ny][nx] != 1:
#             cost = 1 if i % 2 == 0 else math.sqrt(2)
#             neighbors.append((Node((ny, nx), parent=node, parent_move=i), cost, i))
#     return neighbors


# def angle_cost(prev_move, current_move, k):
#     if prev_move is None:
#         return 0  # No angle cost for the initial move
#     angle_diff = abs(current_move - prev_move) * 45  # Each move direction changes by 45 degrees
#     if angle_diff > 180:
#         angle_diff = 360 - angle_diff

#     # Cap the penalty at 90 degrees as the max penalty threshold
#     max_angle = 90
#     effective_angle_diff = min(angle_diff, max_angle)
    
#     # Calculate the penalty as a fraction of k, with 90 degrees being the max penalty
#     return k * (effective_angle_diff / max_angle)


# def heuristic(position, goal):
#     return math.sqrt((position[0] - goal[0])**2 + (position[1] - goal[1])**2)


# def astar_search(start, goal, grid, k):
#     start_node = Node(start, g=0, h=heuristic(start, goal))
#     goal_node = Node(goal)
#     open_set = []
#     heapq.heappush(open_set, start_node)
#     open_set_lookup = {start_node.position: start_node}  # Lookup dictionary for open_set
#     closed_set = set()
#     nodes_generated = 0

#     while open_set:
#         current_node = heapq.heappop(open_set)
#         if current_node.position == goal_node.position:
#             path, moves, costs = reconstruct_path(current_node)
#             return path, moves, costs, nodes_generated  # Path found

#         closed_set.add(current_node.position)

#         for neighbor, move_cost, move in get_neighbors(current_node, grid):
#             if neighbor.position in closed_set:
#                 continue

#             prev_move = current_node.parent_move if current_node.parent else None
#             g_cost = current_node.g + move_cost + angle_cost(prev_move, move, k)
#             h_cost = heuristic(neighbor.position, goal)
#             f_cost = g_cost + h_cost

#             if neighbor.position in open_set_lookup:
#                 existing_node = open_set_lookup[neighbor.position]
#                 if existing_node.f > f_cost:
#                     open_set.remove(existing_node)
#                     heapq.heapify(open_set)  # Reorder the priority queue

#             neighbor.g, neighbor.h, neighbor.f, neighbor.parent = g_cost, h_cost, f_cost, current_node
#             neighbor.parent_move = move
#             heapq.heappush(open_set, neighbor)
#             open_set_lookup[neighbor.position] = neighbor
#             nodes_generated += 1

#     return None, None, None, nodes_generated


# def reconstruct_path(node):
#     path, moves, costs = [], [], []
#     while node:
#         path.append(node.position)
#         moves.append(node.parent.position if node.parent else None)
#         costs.append(node.f)
#         node = node.parent
#     return path[::-1], moves[::-1], costs[::-1]


# def output_result(output_path, path, moves, costs, depth, nodes_generated, grid):
#     with open(output_path, 'w') as file:
#         file.write(f"{depth}\n")
#         file.write(f"{nodes_generated}\n")
#         file.write(" ".join(str(m) for m in moves[1:]) + "\n")
#         file.write(" ".join(f"{cost:.2f}" for cost in costs) + "\n")

#         for position in path[1:-1]:
#             grid[position[0]][position[1]] = 4
        
#         for row in grid:
#             file.write(" ".join(map(str, row)) + "\n")


# if __name__ == "__main__":
#     start, goal, grid = parse_input("Sample_input.txt")
#     k = int(input("Enter the angle penalty factor (k): "))
#     path, moves, costs, nodes_generated = astar_search(start, goal, grid, k)
    
#     if path:
#         output_result("output.txt", path, moves, costs, len(path)-1, len(moves)-1, grid)
#         print("Path found and output saved!")
#     else:
#         print("No path found.")



import math  
import heapq  # Import heapq library to use a priority queue for A* (helps efficiently find the lowest-cost path)

class Node:
    """
    Node class to represent each position in the grid.
    Stores its position, cost to reach (g), heuristic cost (h), and total cost (f).
    Also tracks the parent node to reconstruct the path at the end.
    """
    def __init__(self, position, g=0, h=0, parent=None):
        self.position = position  # Position of the node in the grid as (row, col)
        self.g = g  # Cost from the start node to this node
        self.h = h  # Heuristic (estimated cost) to reach the goal from this node
        self.f = g + h  # Total cost (g + h) to determine the node's priority in A*
        self.parent = parent  # Reference to the previous node in the path

    def __lt__(self, other):
        # Override the < operator to make Node instances comparable based on their f value
        return self.f < other.f


def parse_input(file_path):
    """
    Reads the input file, extracts start and goal positions and the grid.
    Returns start position, goal position, and the grid as a 2D list.
    """
    grid = []
    with open(file_path, "r") as file:
        start_line = [int(i) for i in file.readline().strip().split()]
        start = (start_line[0], start_line[1])
        goal = (start_line[2], start_line[3])
        # start_row, start_col = start_line[0], start_line[1]
        # goal_row, goal_col = start_line[2], start_line[3]
        for line in file:
            row = [int(x) for x in line.strip().split()]
            grid.append(row)
        # Adjust for bottom-left origin 
        grid.reverse()

    return start, goal, grid


def get_neighbors(node: Node, grid):
    """
    Generates neighboring nodes based on 8 possible directions in a 2D grid.
    Only considers positions within grid bounds and avoids obstacles.
    Returns a list of valid neighboring nodes along with their move cost and direction index.

    returns:
    neighbors: List[(Node, cost: int, ith_neighbor: int)], where i = 0 starting from right going clockwise
    """
    directions = [
        (1, 0), (1, -1), (0, -1), (-1, -1),  # right, down-right, down, down-left
        (-1, 0), (-1, 1), (0, 1), (1, 1)  # left, up-left, up, up-right
    ]
    neighbors: list[(Node, int, int)] = []
    for i, (dx, dy) in enumerate(directions):
        # Calculate neighbor's position based on the direction
        nx, ny = node.position[0] + dx, node.position[1] + dy
        # Check if the neighbor's position is within the grid and not an obstacle (1)
        if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid) and grid[ny][nx] != 1:
            # Cost for straight moves is 1, diagonal moves √2
            cost = 1 if i % 2 == 0 else math.sqrt(2)
            neighbors.append((Node((nx, ny), parent=node), cost, i))  # Add neighbor, cost, and direction index
    return neighbors


def angle_cost(prev_move, current_move, k):
    #TODO: this needs work for different values of k.
    """
    Calculates the cost of changing direction based on the difference in angles.
    If there is no previous move (initial move), returns 0.
    Returns the penalty as k times the normalized angle difference.
    """
    # if prev_move is None:
    #     return 0  # No angle cost for the initial move
    # angle_diff = abs(current_move - prev_move[0]) * 45  # Each move direction changes by 45 degrees
    # if angle_diff > 180:  # Use the smallest angle difference if > 180 degrees
    #     angle_diff = 360 - angle_diff
    # return k * (angle_diff / 180)  # Normalize by dividing by 180 to get a fraction of k
    if prev_move is None:
        return 0  # No angle cost for the initial move

    angle_diff = abs(current_move - prev_move[0]) * 45  # Each index represents 45 degrees

    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # Cap the penalty at 90 degrees as the max penalty threshold
    max_angle = 90
    effective_angle_diff = min(angle_diff, max_angle)

    return k * (effective_angle_diff / max_angle)

def heuristic(position: tuple, goal: tuple):
    """
    Calculates the Euclidean distance (straight-line) heuristic to the goal.
    Used to estimate the remaining cost in A*.

    position, goal: tuple, where (x,y)
    """
    x_diff = position[0] - goal[0]
    y_diff = position[1] - goal[1]
    return math.sqrt(x_diff**2 + y_diff**2)


def astar_search(start: tuple, goal: tuple, grid: list, k: int):
    """
    Performs the A* search algorithm with graph search (no repeated states).
    Uses a priority queue to expand nodes with the lowest f-cost.
    Returns the path, moves, costs, and total nodes generated if a path is found.
    
    start, goal: tuple, where (x,y)

    """
    start_node = Node(start, g=0, h=heuristic(start, goal))  # Initialize start node
    goal_node = Node(goal)  # Goal node position for easy comparison
    open_set = []  # Priority queue (min-heap) for the A* frontier
    heapq.heappush(open_set, start_node)  # Push start node into open_set
    closed_set = set()  # Closed set to keep track of visited positions
    nodes_generated = 0  # Counter for generated nodes

    while open_set:
        current_node = heapq.heappop(open_set)  # Pop node with the lowest f-cost
        if current_node.position == goal_node.position:
            res = reconstruct_path(current_node)
            return res[0], res[1], res[2], nodes_generated # path found

        closed_set.add(current_node.position)  # Mark current node as visited

        # Explore neighbors and calculate costs
        for neighbor, move_cost, move in get_neighbors(current_node, grid):
            #TODO: perhaps we could pass in closed_set and check if the neighbor is already visited there?
            if neighbor.position in closed_set:
                continue  # Skip if neighbor is already visited

            # Calculate g-cost, h-cost, and f-cost for the neighbor
            g_cost = current_node.g + move_cost + angle_cost(current_node.parent.position if current_node.parent else None, move, k)
            h_cost = heuristic(neighbor.position, goal)
            f_cost = g_cost + h_cost

            # Only proceed if the neighbor is not already in open set with a lower f-cost
            if any(open_node.position == neighbor.position and open_node.f <= f_cost for open_node in open_set):
                continue

            # Update the neighbor node with calculated costs and parent
            neighbor.g, neighbor.h, neighbor.f, neighbor.parent = g_cost, h_cost, f_cost, current_node
            heapq.heappush(open_set, neighbor)  # Add neighbor to the open set
            nodes_generated += 1  # Increment generated nodes count

    return None, None, None, nodes_generated  # Return None if no path is found


def reconstruct_path(node):
    """
    Reconstructs the path from start to goal by backtracking from the goal node.
    Returns lists of path positions, moves, and costs in reverse order.
    """
    path, moves, costs = [], [], []
    while node:
        path.append(node.position)  # Append the position of the node
        moves.append(node.parent.position if node.parent else None)  # Append the move direction
        costs.append(node.f)  # Append the total cost f of the node
        node = node.parent  # Move to the parent node
    return path[::-1], moves[::-1], costs[::-1]  # Reverse to get start-to-goal order


def output_result(output_path, path, moves, costs, depth, nodes_generated, grid):
    """
    Writes the solution and modified grid to the output file.
    Includes depth, total nodes, move sequence, costs, and the grid with the path marked.
    """
    with open(output_path, 'w') as file:
        file.write(f"{depth}\n")  # Depth level of the goal node
        file.write(f"{nodes_generated}\n")  # Total number of nodes generated
        file.write(" ".join(str(m) for m in moves[1:]) + "\n")  # Move sequence (skip initial None)
        file.write(" ".join(f"{cost:.2f}" for cost in costs) + "\n")  # List of f-costs

        # Mark the path in the grid (except start and goal positions)
        for position in path[1:-1]:
            y = position[1]
            x = position[0]
            grid[y][x] = 4
        
        # Write the grid to the output file
        
        for row in reversed(grid):
            file.write(" ".join(map(str, row)) + "\n")


# Main execution
if __name__ == "__main__":
    start, goal, grid = parse_input("Sample_input.txt")  # Parse input file for start, goal, and grid
    k = int(input("Enter the angle penalty factor (k): "))  # Ask user for angle penalty factor

    path, moves, costs, nodes_generated = astar_search(start, goal, grid, k)  # Run A* search
    
    if path:
        output_result("output.txt", path, moves, costs, len(path)-1, len(moves)-1, grid)  # Save results
        print("Path found and output saved!")
    else:
        print("No path found.")  # Print message if no path found