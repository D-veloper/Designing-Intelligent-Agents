# This code builds on a tutorial by NeuralNine
# Tutorial Title: Conway's Game of Life in Python
# YouTube URL: https://www.youtube.com/watch?v=cRWg2SWuXtM
# This script extensively modifies and extends the original code presented in the tutorial to suit specific needs.

import pygame
import numpy as np
import time
from life_environment import Environment
from agent_bob import RL_Agent
import matplotlib.pyplot as plt

width, height = 20, 20
def main():
    # create pygame window
    pygame.init()
    screen = pygame.display.set_mode((width * Environment.SIZE, height * Environment.SIZE))
    pygame.display.set_caption("Game of Life: Bob's Training")

    # Todo: create lists of patterns we want to create and maintain
    # patterns_list = ["blinker"]

    # create simulation modes
    # there are two modes, manual (press space) and auto (press a)
    mode = 'manual'  # default mode is manual, user steps through generation by pressing space
    continue_sim = True

    # create default instance of environment
    max_steps = 10
    env = Environment(width, height, max_steps)

    # create instance of rl_agent and training parameters
    num_episodes = 50
    scaler = num_episodes * max_steps
    agent_Bob = RL_Agent(width, height, scaler)

    accuracy_scores = {}  # Dictionary to store accuracy scores for each episode
    mean_accuracies = []  # list to store the mean accuracy of each episode

    # Todo: loop through pattern list, changing target pattern and max_steps in environment depending on pattern
    # for...

    # Start Training loop
    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")
        episode_scores = []  # list to store scores for the current episode
        env.initialize_block()  # start with a simple still life block
        env.initialize_blinker(env.target)  # initialise the target pattern.
        env.update_grid(screen, Environment.SIZE)
        pygame.display.flip()

        step = 0
        while step < max_steps:  # agent has a limited number of generations to figure stuff out
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:  # if a button is pressed, mode is auto
                        mode = 'auto' if mode == 'manual' else 'manual'  # toggle between modes

                    elif event.key == pygame.K_SPACE:
                        if mode == 'auto':  # switch to manual if currently in auto mode
                            mode = 'manual'
                        else:  # if in manual mode, step through one generation per space press
                            continue_sim = True
                            print(f"This is generation {step + 1}")

                #Uncomment below to add or remove cells in real time:
                # if pygame.mouse.get_pressed()[0]:
                #     pos = pygame.mouse.get_pos()
                #     if env.grid[pos[1] // env.SIZE, pos[0] // env.SIZE] == 1:
                #         env.grid[pos[1] // env.SIZE, pos[0] // env.SIZE] = 0
                #     else:
                #         env.grid[pos[1] // env.SIZE, pos[0] // env.SIZE] = 1
                #     env.update_grid(screen, env.SIZE)
                #     pygame.display.update()

            if mode == 'auto' or continue_sim:
                # agent assesses the environment and decides on an action
                print(f"This is generation {step + 1}")
                grid_state = env.grid
                action_type, coordinates, probabilities = agent_Bob.get_action(grid_state)

                # next, apply the agents chosen action onto the environment
                env.effect_agent_change(action_type, coordinates, 'bob', screen, Environment.SIZE)
                pygame.display.update()

                if coordinates is not None:
                    print(f"agent wants to {action_type} cell at ({coordinates[0], coordinates[1]})")
                else:
                    print("The agent wants to do nothing")

                # next,update the grid for the next generation
                env.grid = env.update_grid(screen, env.SIZE, goto_next_generation=True)
                pygame.display.update()

                # then we compute the agents reward based on the new grid state
                reward = env.compute_reward(env.grid, env.target, step + 1, action_type, coordinates)

                # then we train the agent based on feedback from the environment, i.e. reward
                if coordinates is not None:  # if the agent didn't choose to do nothing
                    action_index = coordinates[0] * width + coordinates[1]
                else:
                    # if grid is 10 * 10, for example, that's 100 cells going from 0 to 99.
                    # the output layer has an extra node, e.g. 101, to represent do nothing
                    # so 101 node is at position 100. action_index at do nothing is width * height
                    action_index = width * height  # index of 'do nothing'.
                agent_Bob.train(probabilities, action_index, reward)

                step += 1
                agent_Bob.steps_taken += 1
                continue_sim = False  # so that simulation stops if we are in manual mode

                # calculate accuracy based on reward:
                accuracy_score = calculate_accuracy_score(reward)
                episode_scores.append(accuracy_score)

        # we have reached the desired generation limit
        print("Max steps reached. Resetting environment.")

        # save all the accuracy scores for that episode
        accuracy_scores[episode] = episode_scores
        mean_accuracy = np.mean(episode_scores)
        mean_accuracies.append(mean_accuracy)

        if episode == num_episodes - 1:
            plot_accuracy(accuracy_scores)
            plot_mean_accuracies(mean_accuracies)

        env.grid = np.zeros((env.height, env.width))
        env.initialize_block()
        env.update_grid(screen, Environment.SIZE)
        pygame.display.flip()

        time.sleep(0.001)


def calculate_accuracy_score(reward):
    base_reward = width * height
    bonus = 100
    max_reward = base_reward + bonus
    return (reward / max_reward) * 100


def plot_accuracy(accuracy_scores):
    num_episodes = len(accuracy_scores)

    first_idx = 0
    middle_idx = num_episodes // 2
    last_idx = num_episodes - 1

    selected_indices = [first_idx, middle_idx, last_idx]

    # Create a figure and subplots; one subplot per episode in a single column
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 15))
    fig.suptitle("Accuracy Score Over Each Generation for First, Middle, and Last Episodes")

    # Loop through each subplot and plot the accuracy scores for each episode
    for plot_idx, episode_idx in enumerate(selected_indices):
        episode = list(accuracy_scores.keys())[episode_idx]  # Get episode number from index
        scores = accuracy_scores[episode]  # Get scores for the selected episode

        ax = axes[plot_idx]  # Select the appropriate subplot
        ax.plot(scores, label=f'Episode {episode + 1}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Accuracy Score')
        ax.set_ylim(0, 100)
        ax.set_title(f'Episode {episode + 1}')
        ax.legend()

    # Adjust the layout to make room for the main title
    plt.tight_layout(pad=4.0)

    plt.show()


def plot_mean_accuracies(mean_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(mean_accuracies, label='Mean Accuracy per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Accuracy')
    plt.ylim(0, 100)
    plt.title('Mean Accuracy Across All Episodes')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()


    # env.initialize_blinker(env.target)
    # target_pattern = env.target
    # env.initialize_block()
    # env.update_grid(screen, Environment.SIZE)
    # pygame.display.flip()