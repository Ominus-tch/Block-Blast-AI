import pygame
import os

def clear_console(self):
    os.system('cls' if os.name == 'nt' else 'clear')

def display_grid(score, grid):
    print(f"Score: {score}")
    for row in grid:
        print(' '.join('■' if cell else '□' for cell in row))

def display_shapes(remaining_shapes):
    for shape in remaining_shapes:
        shape_height, shape_width = shape.shape
        for row in range(shape_height):
            print(' '.join('■' if cell else ' ' for cell in shape[row]))
        print()