# This code builds on a tutorial by NeuralNine
# Tutorial Title: Conway's Game of Life in Python
# YouTube URL: https://www.youtube.com/watch?v=cRWg2SWuXtM
# This code extensively modifies and extends the original code presented in the tutorial to suit specific needs.

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

    def effect_agent_change(self, action_type, action_index, agent_responsible, screen, size):
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

    def penalty_calculation(self, max_reward, steps_taken, action_type, action_coordinates):
        # calculate time penalty because we want agent to solve as quickly as possible
        # time penalty ensures positive reward reduces  as time passes
        time_penalty_reward = max_reward - steps_taken

        final_reward = time_penalty_reward

        # Apply rewards for actions closer to the center of the grid where the block is
        center_x, center_y = (self.width // 2) - 1, (self.height // 2) - 1

        if action_coordinates is not None:
            row = action_coordinates[0]
            col = action_coordinates[1]
            distance_to_center = abs(row - center_y) + abs(col - center_x)
            if distance_to_center <= 5:
                final_reward += 50

        # check the state of the grid at the action coordinates before the action was taken
        cell_state = self.previous_grid[action_coordinates]

        # Apply penalties for unreasonable actions
        if action_type == 'kill' and cell_state == 0:  # Trying to kill an already dead cell
            final_reward -= 100  # penalty for redundant kill action
        elif action_type == 'revive' and cell_state == 1:  # trying to revive an already living cell
            final_reward -= 100  # penalty for unreasonable 'revive' action

        # if agent can't generate pattern after, max_steps, fail condition is reached
        # if steps_taken >= max_steps:
        #     # print("Failed to achieve target pattern within the allowed number of generations")
        #     final_reward -= 50  # additional penalty for failing

        return final_reward, False

    # the environment computes the rl_agent's reward
    def compute_reward(self, max_reward, current_grid, target_grid, steps_taken, max_steps, action_type, action_coordinates):
        # should return the reward and a flag to know if learning is complete
        difference = np.sum(current_grid != target_grid)  # calculate hamming distance between current and target grid

        if difference == 0:  # if the target has been achieved precisely
            return max_reward, True

        else:  # check if target was achieved but at a different location in the grid
            max_reward = 50
            reward, learning_completed = self.penalty_calculation(max_reward, steps_taken, action_type, action_coordinates)
            return reward, learning_completed

        # # check for pattern completion
        # if np.array_equal(current_grid, target_grid):
        #     print("Target pattern achieved.")
        #     final_reward += 100  # bonus points for achieving desired pattern
        # el

        # return final_reward
        #
        # rows, cols = current_grid.shape
        # found_blinker = False
        #
        # # check for blinker pattern horizontally for each row
        # for i in range(rows):
        #     for j in range(cols - 2):  # On each row, we don't look for blinker past the second to last cell
        #         # check if there are three consecutive 1s and if those 1s are surrounded by zeros
        #         if (current_grid[i, j:j + 3] == 1).all():  # checking for the blinker pattern
        #             # Ensure it's surrounded by zeros horizontally or at boundary
        #             horizontal_clear = (j == 0 or current_grid[i, j - 1] == 0) and \
        #                                (j + 3 == cols or current_grid[i, j + 3] == 0)
        #             # Ensure it's surrounded by zeros vertically or at boundary of grid
        #             vertical_clear = (i == 0 or np.all(current_grid[i - 1, j:j + 3] == 0)) and \
        #                              (i + 1 == rows or np.all(current_grid[i + 1, j:j + 3] == 0))
        #             if horizontal_clear and vertical_clear:
        #                 found_blinker = True
        #
        # # check for blinker pattern vertically for each column
        # for j in range(cols):
        #     for i in range(rows - 2):  # Avoid checking beyond the valid range
        #         # Check if there are three consecutive 1s
        #         if (current_grid[i:i + 3, j] == 1).all():
        #             # Ensure it's surrounded by zeros vertically
        #             vertical_clear = (i == 0 or current_grid[i - 1, j] == 0) and \
        #                              (i + 3 == rows or current_grid[i + 3, j] == 0)
        #             # Ensure it's surrounded by zeros horizontally
        #             horizontal_clear = (j == 0 or np.all(current_grid[i:i + 3, j - 1] == 0)) and \
        #                                (j + 1 == cols or np.all(current_grid[i:i + 3, j + 1] == 0))
        #             if vertical_clear and horizontal_clear:
        #                 found_blinker = True
        #
        # if found_blinker:
        #     max_reward = 50 * max_steps
        # else:
        #     max_reward = 50
