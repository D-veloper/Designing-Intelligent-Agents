# This code builds on a tutorial by NeuralNine
# Tutorial Title: Conway's Game of Life in Python
# YouTube URL: https://www.youtube.com/watch?v=cRWg2SWuXtM
# This script extensively modifies and extends the original code presented in the tutorial to suit specific needs.

import numpy as np
import pygame

class Environment:
    COLOUR_BG = (10, 10, 10)
    COLOUR_GRID = (40, 40, 40)
    COLOUR_DIE_NEXT = (170, 170, 170)
    COLOUR_ALIVE_NEXT = (255, 255, 255)
    COLOUR_DIE_ALICE = (150, 75, 0)
    COLOUR_ALIVE_ALICE = (255, 255, 0)
    COLOUR_DIE_BOB = (255, 0, 0)
    COLOUR_ALIVE_BOB = (0, 0, 255)
    COLOUR_DIE_BOTH = (128, 0, 0)
    COLOUR_ALIVE_BOTH = (0, 255, 0)
    SIZE = 25

    # max_steps is the max number of steps rl_agent should take per desired pattern
    def __init__(self, width, height, max_steps):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.target = np.zeros((height, width))
        self.max_steps = max_steps
        self.previous_grid = np.zeros((height, width))

    def initialize_block(self):
        center_x, center_y = (self.width // 2) - 1, (self.height // 2) - 1
        self.grid[center_y:center_y+2, center_x:center_x+2] = 1  # a 2x2 block pattern for a simple still life start

    def initialize_blinker(self, target):
        center_x, center_y = (self.width // 2) - 1, (self.height // 2) - 1
        target[center_y, center_x:center_x + 3] = 1  # Set three center cells horizontally for a simple blinker
        # note the positioning of the blinker is important otherwise model's job is hard

    def update_grid(self, screen, size, goto_next_generation=False):
        new_grid = np.zeros((self.height, self.width))  # the new grid is the same size as the original

        for row, col in np.ndindex(self.grid.shape):  # iterate through each cell in the grid
            # calculate the cell state (alive or dead). Each cell is either alive (1) or dead (0)
            # by summing up the values of the neighbours of each cell, we determine if it survives, dies or gets born.

            # pick a cell. Go to the cell directly above it and to the left. sum starts here.
            # then go to the cell directly below it and to the left. sum ends there.
            # sum therefore consists of all 8 (max) of the current cell's neighbours.
            alive = np.sum(self.grid[row-1:row+2, col-1:col+2]) - self.grid[row, col]
            colour = self.COLOUR_BG if self.grid[row, col] == 0 else self.COLOUR_ALIVE_NEXT

            # apply the game rules

            if self.grid[row, col] == 1:  # if the current cell is alive
                if alive < 2 or alive > 3:  # if the alive cell has too few or too many neighbours, it dies.
                    if goto_next_generation:  # only apply game rule when we want to go to the next generation.
                        # no need to set the cell to 0 because new_grid is already initialised to 0
                        colour = self.COLOUR_DIE_NEXT

                elif 2 <= alive <= 3:  # if the alive cell has 2 or 3 neighbours, it survives.
                    if goto_next_generation:
                        new_grid[row, col] = 1
                        colour = self.COLOUR_ALIVE_NEXT

            else:  # If the current cell is dead
                if alive == 3:  # if the dead cell has exactly three neighbours, it gets born on the next generation.
                    if goto_next_generation:
                        new_grid[row, col] = 1
                        colour = self.COLOUR_ALIVE_NEXT

            pygame.draw.rect(screen, colour, (col * size, row * size, size - 1, size - 1))

        return new_grid if goto_next_generation else self.grid

    def effect_agent_change(self, action_type, action_index, agent_responsible, screen, size, goto_next_generation=False):
        cell_value = 1
        colour = self.COLOUR_BG
        self.previous_grid = self.grid  # keep a record of what the grid was previously before doing agents bidding
        # later we can penalise agent for unnecessary actions, you can't 'kill' a cell that's already dead, for instance

        if action_type == "nothing":
            return
        elif action_type == "kill":
            cell_value = 0
            self.grid[action_index[0], action_index[1]] = cell_value
            if agent_responsible == 'alice':
                colour = self.COLOUR_DIE_ALICE
            elif agent_responsible == 'bob':
                colour = self.COLOUR_DIE_BOB
            elif agent_responsible == 'both':
                colour = self.COLOUR_DIE_BOTH
            colour = self.COLOUR_BG
        elif action_type == "revive":
            self.grid[action_index[0], action_index[1]] = cell_value
            if agent_responsible == 'alice':
                colour = self.COLOUR_ALIVE_ALICE
            elif agent_responsible == 'bob':
                colour = self.COLOUR_ALIVE_BOB
            elif agent_responsible == 'both':
                colour = self.COLOUR_ALIVE_BOTH

        pygame.draw.rect(screen, colour, (action_index[1] * size, action_index[0] * size, size - 1, size - 1))

        return self.grid

    # the environment computes the rl_agent's reward
    def compute_reward(self, current_grid, target_grid, steps_taken, action_type, action_coordinates):

        difference = np.sum(current_grid != target_grid)  # calculate hamming distance between current and target grid

        # print("hamming score is", difference)

        similarity_score = max(0, len(current_grid.flatten()))

        base_reward = similarity_score  # max base reward is the number of cells in the grid (perfect match)

        # calculate time penalty because we want agent to solve as quickly as possible
        # time penalty ensures positive reward reduces but negative reward increases as time passes
        time_penalty = (steps_taken / self.max_steps) * similarity_score  # time penalty is scaled linearly

        # if agent can't generate pattern after, max_steps, fail condition is reached
        # once agent finds pattern, win condition is reached
        final_reward = base_reward - time_penalty

        # check the state of the grid at the action coordinates before the action was taken
        cell_state = self.previous_grid[action_coordinates]

        # Apply penalties for unreasonable actions
        if action_type == 'kill' and cell_state == 0:  # Trying to kill an already dead cell
            final_reward -= 50  # penalty for redundant kill action
        elif action_type == 'revive' and cell_state == 1:  # trying to revive an already living cell
            final_reward -= 50  # penalty for unreasonable 'revive' action

        # check for pattern completion
        if np.array_equal(current_grid, target_grid):
            print("Target pattern achieved.")
            final_reward += 100  # bonus points for achieving desired pattern
        elif steps_taken > self.max_steps:
            print("Failed to achieve target pattern within the allowed number of generations")
            final_reward -= 50  # additional penalty for failing

        return final_reward
