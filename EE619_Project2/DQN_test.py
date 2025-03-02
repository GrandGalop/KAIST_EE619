
import numpy as np
import torch
import torch.nn as nn
import pdb
import random
from copy import deepcopy

from collections import deque

np.random.seed(1)

class DeepQLearning:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            discount_factor=0.99,
            e_greedy=0.9,
            e_decay=1,
            replace_target_iter=10,
            memory_size=1000,
            batch_size=40
        ):

        # Initialize variables
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = 32 ## temporally setted
        self.learning_rate = learning_rate
        self.e_greedy = e_greedy
        self.e_decay = e_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = discount_factor

        self.replace_iter = replace_target_iter
        self.copy_stack = 0

        self.memory = []
        self.mstack = 0
        self.flag = False

        self.construct_network() 

    def construct_network(self):
        self.criterion = nn.MSELoss()
        self.model = nn.Sequential(nn.Linear(self.n_features, self.n_hidden),
                                   nn.LeakyReLU(),
                                   nn.Linear(self.n_hidden, self.n_hidden*2),
                                   nn.LeakyReLU(),
                                   nn.Linear(self.n_hidden*2, self.n_actions))
        # self.model = nn.Sequential(nn.Linear(self.n_features, self.n_hidden),
        #                            nn.LeakyReLU(),
        #                            nn.Linear(self.n_hidden, self.n_actions))

        self.model_fix = deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def store_transition(self, s, a, r, next_s, terminal):
        if not self.flag:
            self.memory.append([s, a, r, next_s, terminal])
        else:
            self.memory[self.mstack] = [s, a, r, next_s, terminal]
        self.mstack += 1

        if self.mstack == self.memory_size:
            self.mstack = 0
            self.flag = True

    def choose_action(self, state):
        if np.random.rand() < self.e_greedy:
            return np.random.randint(0, self.n_actions)

        q = self.pred_qval(self.model, state)
        return np.argmax(q)
        
    def pred_qval(self, model, state):
        with torch.no_grad():
            return model(torch.Tensor(np.array(state)))
        
    def update_epsilon(self):
        self.e_greedy = max([self.e_greedy*self.e_decay, 0.01])

    def learn(self):
        # retrieve the memories
        nid_samples = np.random.choice(len(self.memory), self.batch_size, replace=True)    
        samples = [self.memory[n] for n in nid_samples]
        states = [s[0] for s in samples]
        next_states = [s[3] for s in samples]

        q_pred = self.model(torch.Tensor(np.array(states)))
        q_next_pred = self.pred_qval(self.model_fix, next_states)
        targets = q_pred.clone().detach()

        for n in range(self.batch_size):
            _, a, r, _, terminal = samples[n]
            if terminal:
                targets[n][a] = r
            else:
                targets[n][a] = r + self.gamma * q_next_pred[n].max()

        # update
        loss = self.criterion(q_pred, targets)
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.copy_stack += 1
        if self.copy_stack == self.replace_iter:
            self.model_fix = deepcopy(self.model)
            self.copy_stack = 0

# # 모델 구성
# model = nn.Sequential(
#     nn.Linear(2,8),  # 입력층
#     nn.ReLU(),                           # 활성화 함수
#     nn.Linear(8, 4)  # 출력층
# )

# # 입력 데이터
# input_data = torch.randn(10, 2)

# # forward propagation
# output = model(input_data)
# print(output)

dqn = DeepQLearning(4, 2)
dqn.store_transition(np.array([0, 1]), 2, 3, np.array([1, 1]), False)
dqn.store_transition(np.array([0, 1]), 2, 3, np.array([1, 1]), False)
dqn.store_transition(np.array([0, 1]), 2, 3, np.array([1, 1]), False)
dqn.store_transition(np.array([0, 1]), 2, 3, np.array([1, 1]), False)
dqn.store_transition(np.array([0, 1]), 2, 3, np.array([1, 1]), False)
dqn.learn()