from random import shuffle, choice
from copy import copy
from time import time

# dimensions = 4      
# CUSTOM_BOARD = [ 1, 2, 3, 0, 4, 5, 6 , 7 , 9 , 10 ,11 , 8 ,13 , 14 , 15 ,12] 
# MAX_DEPTH = 200

class Node(object):
	
	def __init__(self, parent, board, empty_pos, dimensions, h_function):
		self.parent = parent
		self.board = board
		self.empty_pos = empty_pos
		self.dimensions = dimensions                     
		self.h_function = h_function
		self.heuristic = self.calculate_heuristic()
		self.possible_moves = self.calculate_possible_moves()

	def calculate_heuristic(self):
                
		if(self.h_function == 1):

			heuristic = 0
			for dim, val in self.board.items():
				if val is None: continue
				y = int((val-1)/self.dimensions)
				x = (val-1)%self. dimensions
				h = abs(x - dim[0]) + abs(y - dim[1])
				heuristic += h
			return heuristic
                
		if(self.h_function == 2):

			heuristic = 0
			# Loop through each row and column of the board
			for i in range(self.dimensions):
				# Find the tiles that are in their goal row or column
				row_tiles = [self.board[(j, i)] for j in range(self.dimensions) if self.board[(j, i)] and (self.board[(j, i)] - 1) // self.dimensions == i]
				col_tiles = [self.board[(i, j)] for j in range(self.dimensions) if self.board[(i, j)] and (self.board[(i, j)] - 1) % self.dimensions == i]
				# Count the number of linear conflicts in each row or column
				row_conflicts = count_conflicts(row_tiles)
				col_conflicts = count_conflicts(col_tiles)
				# Add the conflicts to the heuristic value
				heuristic += row_conflicts + col_conflicts
			# Add the Manhattan distance to the heuristic value
			heuristic += calculate_manhattan(self.board)
			# Return the heuristic value
			return heuristic

	def calculate_possible_moves(self):
		p = self.empty_pos
		possible_moves = list()
		for pm in [(p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1)]:
			if pm[0] >= self.dimensions or pm[0] < 0 or pm[1] >= self.dimensions or pm[1] < 0: continue
			if self.parent is not None and self.parent.empty_pos == pm: continue
			possible_moves.append(pm)
		return possible_moves
	
	def get_children(self):
		children = list()
		for move in self.possible_moves:
			try:
				new_board = copy(self.board)
				new_board[self.empty_pos], new_board[move] = self.board[move], None
				children.append(Node(self, new_board, move, self.dimensions,self.h_function))
			except KeyError: pass
		return children
	
	def __list__(self):
		result = []
		for y in range(self.dimensions):
			for x in range(self.dimensions):
				result.append(self.board[x,y] or 0)	
		return result




# A helper function to calculate the Manhattan distance
def calculate_manhattan(board):
  # Get the width and height of the board
  width = int(len(board) ** 0.5) # Assume the board is a square
  height = width
  # Initialize the distance value
  distance = 0
  # Loop through each tile of the board
  for i in range(height):
    for j in range(width):
      # Get the value of the tile
      val = board[(j, i)]
      # Ignore the empty tile
      if val == None: continue
      # Calculate the horizontal and vertical distances to the goal position
      x_dist = abs(j - ((val - 1) % width))
      y_dist = abs(i - ((val - 1) // width))
      # Add the distances to the distance value
      distance += x_dist + y_dist
  # Return the distance value
  return distance

# A helper function to count the number of linear conflicts in a list of tiles
def count_conflicts(tiles):
  # Initialize the conflict count
  conflicts = 0
  # Loop through each pair of tiles in the list
  for i in range(len(tiles) - 1):
    for j in range(i + 1, len(tiles)):
      # Check if the tiles are reversed relative to their goal positions
      if tiles[i] > tiles[j]:
        # Increment the conflict count by 2
        conflicts += 2
  # Return the conflict count
  return conflicts





###################################################################################
#khoitao-input

# A function to count the number of inversions in a list
def count_inversions(lst):
  inv_count = 0
  for i in range(len(lst) - 1):
    for j in range(i + 1, len(lst)):
      if lst[j] and lst[i] and lst[i] > lst[j]:
        inv_count += 1
  return inv_count

# A function to swap two adjacent tiles in a list
def swap_tiles(lst, i, j):
  lst[i], lst[j] = lst[j], lst[i]

# A function to make a puzzle solvable by swapping tiles if needed
def make_solvable(puzzle):
  # Get the width and height of the puzzle
  width = int(len(puzzle) ** 0.5) # Assume the puzzle is a square
  height = width
  # Get the number of inversions and the row of the empty space
  inv_count = count_inversions(puzzle)
  empty_row = height - (puzzle.index(0) // width)
  # Check if the puzzle is already solvable
  if width % 2 == 1 and inv_count % 2 == 0:
    return puzzle # Odd width and even inversions
  elif width % 2 == 0 and (inv_count + empty_row) % 2 == 1:
    return puzzle # Even width and odd sum of inversions and empty row
  else:
    # Find two adjacent tiles that are not zero
    i = j = -1
    for k in range(len(puzzle) - 1):
      if puzzle[k] != 0 and puzzle[k + 1] != 0:
        i = k
        j = k + 1
        break
    # Swap them and return the modified puzzle
    swap_tiles(puzzle, i, j)
    return puzzle

# Your original function with some modifications
def create_board(dimensions,board=None):

	solvable = lambda b: sum([sum([1 for i in range(n+1, len(b)) if b[n] > b[i]]) for n in range(0, len(b)-1)]) % 2 == 0
	if board is None:
		board = [i for i in range(1, dimensions**2)]
		shuffle(board)
	else:
		if not solvable(board): board = make_solvable(board)
	
	return {(column, row): board.pop(0) for row in range(dimensions) for column in range(dimensions) if board}

                                           
def get_solution(node):
	solution = list()
	while node is not None:
		solution.append(node)
		node = node.parent
	solution.reverse()
	return solution

def search(node, g, bound):
	f = node.heuristic + g
	if f > bound or node.heuristic == 0: return node,f
	min_f = 9999
	for n in node.get_children():
		nn,t = search(n, g + 1, bound)
		if nn.heuristic == 0: return nn,t
		if t < min_f: min_f = t
	return node,min_f


def solve(board, bound, max_depth):
	if bound > max_depth: return None, None # Unsolvable
	n,t = search(board, 0, bound)
	if n.heuristic == 0: return n,t
	return solve(board, t, max_depth)

def solvePuzzle(d, custom_board , h_function):
    key = None
    print("hfunction", type(h_function))
    h = int(h_function)
    dimensions = int(d)
    board = create_board(dimensions,custom_board)  
    for k,v in board.items():
        if v == 0:
            key = k
            break      
    print(key)
    root = Node(None, board, key, dimensions,h)   
    print("root:",root)
    print("Board:", board)
    print(root.__list__())                       
    start_time = time()
    max_depth = 200;
    end_node, moves = solve(root, root.heuristic, max_depth)
    end_time = time() - start_time
    if end_node is None:
        return "Unsolvable"
    else:
        print("\nSolution (%d moves):" % moves)
        result=[]
        for node in get_solution(end_node):		
            result.append(node.__list__())
        print(result)
        print("\nSolved in %.4f seconds" % end_time)
		
        if(h==1):
            return {
	            "moves": moves,
	            "result": result,
				"mahattan_time": end_time
			}
        else:
            return {
	            "moves": moves,
	            "result": result,
				"linear_time": end_time
			}           
             

solvePuzzle(3, [ 1, 2, 3, 0, 4 , 5, 6 , 7 , 8]  , 1)