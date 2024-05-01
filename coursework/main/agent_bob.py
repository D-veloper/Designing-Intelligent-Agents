import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import  torch.nn.functional as F

class RL_Agent:
    def __init__(self, width, height, scaler):
        self.scaler = scaler
        self.steps_taken = 0  # keeps track of how many steps agent has taken
        self.input_size = width * height  # input layer takes entire grid as a flattened list
        # the output is the same size as the input but with one additional neuron to mean "do nothing"
        self.output_size = self.input_size + 1

        # create the neural network architecture
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 128),  # the first hidden layer
            nn.ReLU(),  # relu activation function chosen arbitrarily based on usual conventions
            nn.Linear(128, 256),  # the second hidden layer
            nn.ReLU(),
            nn.Linear(256, 128),  # the second hidden layer
            nn.ReLU(),
            nn.Linear(128, 128),  # the second hidden layer
            nn.ReLU(),
            nn.Linear(128, self.output_size)  # the last/output layer
        )

        self.loss_function = nn.MSELoss()  # using Mean Squared Error for loss function

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.042)  # adam optimizer and learning rate chosen based on conventions

    def forward_propagation(self, training_data):
        probabilities = self.model(training_data)
        return F.softmax(probabilities, dim=-1)  # applying softmax to the output layer

    # Agent was not very exploratory so we add greedy function
    def get_action(self, grid_state):
        e = max(0.1, 1 - (self.steps_taken / self.scaler))  # decrease e over time
        grid_state = torch.tensor(grid_state, dtype=torch.float32)  # convert the grid into a PyTorch tensor
        grid_state = grid_state.view(1, -1)  # flatten our grid, so we have 1 batch with the number of cells in our grid

        possible_outcomes = self.forward_propagation(grid_state)  # get network's predictions on various actions to take
        action_moves = possible_outcomes[0][:-1]  # exclude "do nothing" from list of actions
        do_nothing_probability = possible_outcomes[0][-1].item()  # get value representing the choice to do nothing

        # the idea is thus: if an outcome is closer to 0.33, the agent wants the cell dead
        # if it is closer to 0.66 then alive, closer to 1 then do nothing
        # but the agent can only make one move per generation
        # so, we calculate the distance each action probability is from our kill and revive signals

        # distances returns a list (tensor) of all the smallest distances between kill signal and revive signal
        distances_to_kill = torch.abs(action_moves - 0.33)
        distances_to_revive = torch.abs(action_moves - 0.66)
        distance_do_nothing = abs(1 - do_nothing_probability)

        # we need to find which distance is smaller for each action and track the source of that distance
        # is_kill_closer returns boolean tensor that's true at each cell if kill distance < revive distance at that cell
        is_kill_closer = distances_to_kill < distances_to_revive
        min_distances = torch.where(is_kill_closer, distances_to_kill, distances_to_revive)

        # the smaller the distance, the stronger our confidence so we calculate a minimum
        min_distance, min_index = torch.min(min_distances, 0)  # find the closest distance and the index of the action move
        min_distance = min_distance.item()
        min_index = min_index.item()

        # I'm not sure how likely it is...
        # but it is possible that multiple cells will have the same distance from kill and revive signal...
        tied_indices = (min_distances == min_distance).nonzero(as_tuple=False).squeeze().tolist()

        # to resolve TypeError: object of type 'int' has no len() at line 61, we make sure tied_indices is a list
        if isinstance(tied_indices, int):
            tied_indices = [tied_indices]

        # Make Action Decisions

        if np.random.random() < e:
            # Randomly choose between all possible actions, even do nothing
            chosen_index = np.random.randint(0, self.input_size + 1)
            if chosen_index == self.input_size:
                return 'nothing', None, possible_outcomes
            else:
                row, col = divmod(chosen_index, int(np.sqrt(self.input_size)))
                i = random.randint(0, 1)
                if i == 1:
                    return 'kill', (row, col), possible_outcomes
                else:
                    return 'revive', (row, col), possible_outcomes

        else:
            if distance_do_nothing < min_distance:
                return 'nothing', None, possible_outcomes

            # in there are ties, one is chosen at random
            if len(tied_indices) > 1:
                chosen_index = np.random.choice(tied_indices)
            else:
                chosen_index = min_index

        action_type = 'kill' if is_kill_closer[chosen_index] else 'revive'

        # calculate the cell coordinates to kill or revive
        row, col = divmod(chosen_index, int(np.sqrt(self.input_size)))

        return action_type, (row, col), possible_outcomes

    def train(self, probabilities, action_index, reward):
        # use the probabilities from forward propagation to see what action agent thought most rewarding
        rewarding_action = probabilities[0, action_index]

        # convert the reward to a tensor and calculate loss
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        # Loss calculation using negative log likelihood
        # our loss function will penalise the model when it assigns high probability of reward to actions that received
        # low rewards or even penalties.
        # it will encourage the probability of the rewarding action to be high if the reward is high, and low if the
        # reward is low or negative.

        epsilon = 1e-8  # a very tiny constant to prevent log(0)
        loss = -torch.log(rewarding_action + epsilon) * reward_tensor

        # zero the gradients before calculating them for this batch
        # to prevent stacking of gradients from multiple forward and backward passes causing errors in gradient updates
        self.optimizer.zero_grad()
        loss.backward()  # perform back propagation to calculate new gradients
        self.optimizer.step()  # perform a single optimization step to update model params

        # return loss.item()




