# This code builds on a tutorial by "Machine Learning with Phil"
# Tutorial Title: Deep Q Learning is Simple with PyTorch
# YouTube URL: https://www.youtube.com/watch?v=wc-FxNENg9U
# This code extensively modifies and extends the original code presented in the tutorial to suit specific needs.

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


class DeepQNetwork(nn.Module):
    # hl stands for hidden layer. hence hl_1 is hidden layer 1
    # Optimizer is Adam optimizer based on convention
    def __init__(self, lr, input_layer, hl_1, hl_2, output_layer):
        super(DeepQNetwork, self).__init__()
        self.input_layer = input_layer
        self.hl_1 = hl_1
        self.hl_2 = hl_2
        self.output_layer = output_layer
        self.func_1 = nn.Linear(*self.input_layer, self.hl_1)
        self.func_2 = nn.Linear(self.hl_1, self.hl_2)
        self.func_3 = nn.Linear(self.hl_2, self.output_layer)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use gpu if possible
        self.to(self.device)

    def forward(self, state):
        # activation function is ReLu based on convention
        x = f.relu_(self.func_1(state))
        x = f.relu_(self.func_2(x))
        actions = self.func_3(x)  # let's get the raw values of the agents reward estimates

        return actions


class RL_Agent:
    def __init__(self, gamma, epsilon, lr, input_layer, batch_size, output_layer,
                 max_mem_size=100000, epsilon_end=0.01, eps_dec=5e-6):
        self.input_size = input_layer
        self.output_size = output_layer

        self.gamma = gamma   # gamma determines weight of future rewards
        self.epsilon = epsilon  # epsilon for exploration
        self.eps_min = epsilon_end  # when to stop decrementing epsilon
        self.eps_dec = eps_dec  # factor to decrement epsilon by
        self.learning_rate = lr  # influences how drastically agent changes/learns new strategy

        self.action_space = [i for i in range(output_layer)]  # list of available actions
        self.memory_size = max_mem_size
        self.mem_count = 0  # to keep track of available memory
        self.batch_size = batch_size

        # the evaluation function or network
        self.Q_eval = DeepQNetwork(self.learning_rate, input_layer=input_layer, hl_1=256,
                                   hl_2=256, output_layer=output_layer)
        self.state_memory = np.zeros((self.memory_size, *input_layer), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_layer), dtype=np.float32)

        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_count % self.memory_size
        self.state_memory[index] = state.flatten()
        self.new_state_memory[index] = state_.flatten()
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_count += 1

    # Agent was not very exploratory so we add greedy function
    def get_action(self, grid_state):

        # the new idea is to unambiguously represent the agents decision
        # the outputs 0 - input_size - 1, maps on to the grid but for kill only
        # the outputs input_size - (2 * input_size-1) map on to the grid but for revive only

        if np.random.random() > self.epsilon:  # Exploitation
            current_state = torch.tensor([grid_state.flatten()], dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(current_state)
            best_action = torch.argmax(actions).item()
        else:  # Exploration
            best_action = np.random.choice(self.action_space)

        # return actions
        print("chosen index is", best_action)
        if best_action == self.output_size - 1:  # if agent decides to do nothing
            return best_action, 'nothing', None

        if best_action < self.input_size[0]:  # if the agent decided to kill
            action_type = 'kill'
            cell_index = best_action
        else:  # if agent decided to revive
            action_type = 'revive'
            cell_index = best_action - self.input_size[0]

        print("calculated cell index is", cell_index)
        # calculate the cell coordinates to kill or revive
        row, col = divmod(cell_index, int(np.sqrt(self.input_size)))

        coordinates = (row, col)

        print(f"coordinate is ({row}, {col})")

        return best_action, action_type, coordinates

    def train(self):
        if self.mem_count < self.batch_size:
            return

        # zero the gradients before calculating them for this batch
        # to prevent stacking of gradients from multiple forward and backward passes causing errors in gradient updates
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_count, self.memory_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]  # greedy action

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    # def train(self, probabilities, action_index, reward):
    #     # max reward is perfect score (grid size) + action close to center + bonus for correct pattern
    #     max_reward = 0
    #     min_reward = -1 * (self.input_size + 10)
    #     # normalise the reward so that it fits in the same scale as predictions
    #     scaled_reward = (reward - min_reward) / (max_reward - min_reward)
    #
    #     # use the probabilities from forward propagation to see what action agent thought most rewarding
    #     predicted_reward = probabilities[0, action_index]
    #     predicted_reward_tensor = torch.tensor([predicted_reward], dtype=torch.float32, requires_grad=True)
    #
    #     # convert the reward to a tensor and calculate loss
    #     target_reward_tensor = torch.tensor([scaled_reward], dtype=torch.float32)
    #
    #     # Loss calculation using mean squared error function
    #     loss = self.loss_function(predicted_reward_tensor, target_reward_tensor)
    #
    #     self.optimizer.zero_grad()
    #     loss.backward()  # perform back propagation to calculate new gradients
    #     self.optimizer.step()  # perform a single optimization step to update model params
    #
    #     # return loss.item()
