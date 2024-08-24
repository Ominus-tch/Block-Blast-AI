import random
import numpy as np
import time

class BlockBlast:
    def __init__(self, grid_size=8):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.shapes = [
            # np.array([[1, 1, 1], [0, 1, 0]]),   #   ###  #
            # np.array([[0, 1, 0], [1, 1, 1]]),   ##   #  ##
            # np.array([[0, 1], [1, 1], [0, 1]]), #    #   #
            # np.array([[1, 0], [1, 1], [1, 0]]),     ###

            np.array([[1, 1], [1, 1]]), # 2x2
            # np.array([[1, 1, 1], [1, 1, 1]]), # 2x3
            # np.array([[1, 1], [1, 1], [1, 1]]), # 3x2
            # np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), # 3x3

            # np.array([[1, 1]]), # 1x2
            # np.array([[1], [1]]), # 2x1

            # np.array([[1, 1, 1]]), #1x3
            # np.array([[1], [1], [1]]), # 3x1

            # np.array([[1, 1, 1, 1]]), # 1x4
            # np.array([[1], [1], [1], [1]]), # 4x1

            # np.array([[1, 1, 1, 1, 1]]), # 1x5
            # np.array([[1], [1], [1], [1], [1]]), # 5x1
        ]
        self.remaining_shapes = self.get_new_shapes()
        self.max_shape_size = 3
        self.score = 0
        self.combo = 0
        self.placed_without_clear = 0
        self.game_iteration = 0

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.get_new_shapes()
        self.score = 0
        self.combo = 0
        self.placed_without_clear = 0
        self.game_iteration = 0
    
    def get_state(self):
        # Flatten the grid
        flattened_grid = self.grid.flatten()

        # Flatten and pad each shape to the maximum shape size
        flattened_shapes = [
            np.pad(shape, ((0, self.max_shape_size - shape.shape[0]), (0, self.max_shape_size - shape.shape[1])), mode='constant').flatten()
            for shape in self.remaining_shapes
        ]

        # If there are less than 3 shapes, pad the flattened_shapes list with zeros
        while len(flattened_shapes) < 3:
            flattened_shapes.append(np.zeros(self.max_shape_size * self.max_shape_size))   

        # Concatenate the grid and all shapes
        state = np.concatenate([flattened_grid] + flattened_shapes)
        return state

    def calculate_lines_cleared_score(self, n):
        if n <= 0:
            return 0
        if n == 1:
            return 10
        return self.calculate_lines_cleared_score(n - 1) * 2

    def calculate_score(self, lines_cleared):
        line_cleared_score = self.calculate_lines_cleared_score(lines_cleared)
        self.score += line_cleared_score * self.combo

    def calculate_reward(self, cells_placed, lines_cleared) -> int:
        line_cleared_score = self.calculate_lines_cleared_score(lines_cleared)

        reward = cells_placed + line_cleared_score * self.combo

        return reward

    def check_fit(self, shape, position):
        if shape is None:
            return False

        shape_height, shape_width = shape.shape
        y, x = position
        if x + shape_height > self.grid_size or y + shape_width > self.grid_size:
            return False
        
        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i, j] == 1 and self.grid[x + i, y + j] == 1:
                    return False
                
        return True

    def place_shape(self, shape, position) -> int:
        shape_height, shape_width = shape.shape
        y, x = position
        num_cells = 0
        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i, j] == 1:
                    self.grid[x + i, y + j] = 1
                    num_cells += 1

        return num_cells

    def clear_lines(self):
        lines_cleared = 0

        new_grid = self.grid
        for i in range(self.grid_size):
            if all(self.grid[i, j] == 1 for j in range(self.grid_size)):
                new_grid[i, :] = 0
                lines_cleared += 1
        for j in range(self.grid_size):
            if all(self.grid[i, j] == 1 for i in range(self.grid_size)):
                new_grid[:, j] = 0
                lines_cleared += 1

        if lines_cleared == 0:
            self.placed_without_clear += 1
        else:
            self.placed_without_clear = 0
            self.combo += 1

        if self.placed_without_clear >= 3:
            self.combo = 0

        self.grid = new_grid
        return lines_cleared

    def is_game_over(self):
        for shape in self.remaining_shapes:
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if self.check_fit(shape, (x, y)):
                        return False
        return True

    def get_new_shapes(self):
        self.remaining_shapes = [random.choice(self.shapes) for _ in range(3)]

    def is_valid_action(self, action) -> bool:
        shape_index, x, y = action // (self.grid_size ** 2), (action % (self.grid_size ** 2)) // self.grid_size, action % self.grid_size

        #print(f"i: {shape_index}, ({x}, {y})")
        #print(len(self.remaining_shapes))

        #time.sleep(3)

        if shape_index < 0 or shape_index > len(self.remaining_shapes) - 1:
            return False
        
        if x < 0 or y < 0:
            return False
        
        if x > self.grid_size - 1 or y > self.grid_size - 1:
            return False
        
        return True

    def step(self, action):
        shape_index, x, y = action // (self.grid_size ** 2), (action % (self.grid_size ** 2)) // self.grid_size, action % self.grid_size
        
        shape = self.remaining_shapes[shape_index]
        
        reward = 0
        if self.check_fit(shape, (x, y)):
            cells_placed = self.place_shape(shape, (x, y))
            lines_cleared = self.clear_lines()
            reward = self.calculate_reward(cells_placed, lines_cleared)
            self.score += reward

            reward += 10

            if len(self.remaining_shapes) == 0:
                self.get_new_shapes()

            valid_action = True
            next_state = self.get_state()
        else:
            #print("Failed to invalid spot")
            reward = -10
            valid_action = False
            next_state = None

        return next_state, reward, valid_action, self.score