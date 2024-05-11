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

width, height = 4, 4  # make sure width and height are equal. some calculations break down if grid is not a square


def main():

    # create pygame window
    pygame.init()
    screen = pygame.display.set_mode((width * Environment.SIZE, height * Environment.SIZE))
    pygame.display.set_caption("Game of Life Pattern Stabilization")

    # Todo: create lists of patterns we want to create and maintain
    # patterns_list = ["blinker"]

    # create simulation modes
    # there are two modes, manual (press space) and auto (press a)
    mode = 'manual'  # default mode is manual, user steps through generation by pressing space
    continue_sim = True  # boolean to help toggle simulation between modes

    # create default instance of environment
    max_steps = 5  # the desired number of steps agent should achieve goal under
    env = Environment(width, height, max_steps)  # instance of the environment

    # create instance of rl_agent and training parameters
    num_episodes = 100  # number of training episodes
    num_generations = 6  # limit of an episode. Once we reach here, episode terminates
    input_size = width * height  # our input layer is the grid as a flat list so its size must be width * height
    output_size = (2 * input_size) + 1  # output layer is a node to represent each action that can be done on each cell
    agent_bob = RL_Agent(gamma=0.99, epsilon=1, lr=0.0042, input_layer=[input_size], batch_size=12,
                         output_layer=output_size)

    # getting information on agent performance
    scores, eps_history = [], []
    # Todo: loop through pattern list, changing target pattern and max_steps in environment depending on pattern
    # for...

    # Start Training loop
    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")

        max_reward = 104

        env.initialize_block()  # start with a simple still life block
        env.initialize_blinker(env.target)  # initialise the target pattern.
        # env.update_grid(screen, Environment.SIZE)
        pygame.display.flip()

        step = 0  # this tracks the number of steps we have taken
        learning_complete = False  # this tracks if agent succeeded to learn how to stabilize pattern
        score = 0

        # agent has a limited number of generations to figure stuff out before episode ends
        while not learning_complete and step < num_generations:
            for event in pygame.event.get():  # player quits, close window.
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:  # if 'a' button is pressed, mode is auto
                        mode = 'auto' if mode == 'manual' else 'manual'  # toggle between modes

                    elif event.key == pygame.K_SPACE:
                        if mode == 'auto':  # switch to manual if currently in auto mode
                            mode = 'manual'
                        else:  # if in manual mode, step through one generation per space press
                            continue_sim = True
                            print(f"This is generation {step + 1}")

                # Note: Uncomment below to add or remove cells in real time:
                # if pygame.mouse.get_pressed()[0]:
                #     pos = pygame.mouse.get_pos()
                #     if env.grid[pos[1] // env.SIZE, pos[0] // env.SIZE] == 1:
                #         env.grid[pos[1] // env.SIZE, pos[0] // env.SIZE] = 0
                #     else:
                #         env.grid[pos[1] // env.SIZE, pos[0] // env.SIZE] = 1
                #     env.update_grid(screen, env.SIZE)
                #     pygame.display.update()

            # if we are in manual mode and agent messes up the grid, reset it
            if mode == 'manual' and np.sum(env.grid) <= 1:
                env.initialize_block()
                pygame.display.update()

            elif mode == 'auto' or continue_sim:
                # reset the grid but not the episode if agent's action messes everything completely.
                if np.sum(env.grid) <= 1:
                    env.initialize_block()
                    pygame.display.update()

                # agent assesses the environment and decides on an action
                grid_state = env.grid
                best_action, action_type, coordinates = agent_bob.get_action(grid_state)

                # next, apply the agents chosen action onto the environment
                env.effect_agent_change(action_type, coordinates, 'bob', screen, Environment.SIZE)
                pygame.display.update()

                # get some debugging information
                if coordinates is not None:
                    print(f"agent wants to {action_type} cell at ({coordinates[0], coordinates[1]})")
                else:
                    print("The agent wants to do nothing")

                # next, update the grid for the next generation
                env.grid = env.update_grid(screen, env.SIZE, goto_next_generation=True)
                pygame.display.update()

                # then we compute the agents reward based on the new grid state
                reward, learning_complete = env.compute_reward(max_reward, env.grid, env.target, step + 1,
                                                               max_steps, action_type, coordinates)

                score += reward  # update the score
                #scores.append(reward)

                new_state = env.grid  # save the new state

                # if learning_complete:  # pause the simulation, so we can see the correct pattern has been achieved
                #     mode = 'manual'

                # store current results in agents memory and train it
                agent_bob.store_transition(grid_state, best_action, reward, new_state, learning_complete)
                agent_bob.train()

                step += 1
                max_reward -= 1
                continue_sim = False  # so that simulation stops if we are in manual mode

        # note the scores at the end of each episode
        scores.append(score)
        eps_history.append(agent_bob.epsilon)
        # avg_score = np.mean(scores)

        # we have reached the desired generation limit
        print("Max steps reached. Resetting environment.")
        if mode == 'auto' or continue_sim:
            env.grid = np.zeros((env.height, env.width))
            env.initialize_block()
            env.update_grid(screen, Environment.SIZE)
            pygame.display.flip()

        time.sleep(0.001)

    # plot learning curve
    plot_learning_curve(scores)

def plot_learning_curve(scores):
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Plot of reward model earns per episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Accuracy')
    #plt.ylim(-100, 100)
    plt.title('Plot of Total Reward Earned Per Episode')
    plt.legend()
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
