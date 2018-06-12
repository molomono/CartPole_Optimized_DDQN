import random
import math
import gym
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, Embedding #need to practice with this never used it before
from keras.optimizers import SGD, Adam

############### Author ################
## Brian Armstrong, github.com/molomono

DDQN = True
#DQN:   Y_t := Rt+1 + gamma*maxQ(St+1,a;theta_t)
#DDQN:  Y_t := Rt+1 + gamma*Q(St+1, argmaxQ(St+1,a;theta_t);theta_t)

class DDQN_Agent:
    """ Deep Q-learning Network

    This is my first attempt at an implementation of a DeepMind inspired DQN.
    .. members:
        * memory: Class that allows acces to past experiences, this member is used to take samples from past experiences to use in optimization.
        * Qout: The neural network function, where the output is the actions. The input is the observed states aquired from the environment
    """
    def __init__(self, number_inputs, number_outputs, environment, hidden_layers={'layers':[16,32], 'activation':['tanh','tanh']}, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, gamma=0.99, learning_rate=0.01, lr_decay=0.01):
        #Greedy Epsilon initializaiton
        self.environment    = environment
        self.epsilon        = epsilon
        self.epsilon_temp   = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay

        #Experience Replay learning parametes
        self.batch_size     = batch_size
        self.gamma          = gamma
        self.adam           = Adam(lr=learning_rate, decay=lr_decay)
        #Initialize memory and model
        self.memory     = deque(maxlen=50000)

        #Setup the 'gradient descending' algorithms.
        adam = Adam(lr=learning_rate, decay=lr_decay)
        #Construct the NN archetecture###############################
        self.model = Sequential()
        for i in range(len(hidden_layers['layers'])):
            if i is 0:
                self.model.add(Dense(hidden_layers['layers'][i], input_dim=number_inputs, activation=hidden_layers['activation'][i], kernel_initializer=hidden_layers['initializer'][i]))
            else:
                self.model.add(Dense(hidden_layers['layers'][i], activation=hidden_layers['activation'][i], kernel_initializer=hidden_layers['initializer'][i]))
        self.model.add(Dense(number_outputs, activation='linear'))

        #Compile the NN
        self.model.compile(optimizer=adam, loss='mse')

        if DDQN:
            self.target_model = Sequential()
            for i in range(len(hidden_layers['layers'])):
                if i is 0:
                    self.target_model.add(Dense(hidden_layers['layers'][i], input_dim=number_inputs, activation=hidden_layers['activation'][i], kernel_initializer=hidden_layers['initializer'][i]))
                else:
                    self.target_model.add(Dense(hidden_layers['layers'][i], activation=hidden_layers['activation'][i],  kernel_initializer=hidden_layers['initializer'][i]))
            self.target_model.add(Dense(number_outputs, activation='linear'))
            #Compile the NN
            self.target_model.compile(optimizer=adam, loss='mse')
            print 'run the xxx.update_target() command after initialization'


    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append( (state, action, reward, next_state, done) )

    def replay(self, batch_size = 0):
        if batch_size is 0:
            batch_size = self.batch_size

        X, Y = [], []

        if len(self.memory) < batch_size:
            minibatch = len(self.memory)
        else:
            minibatch = batch_size

        minibatch = random.sample(self.memory, minibatch)
        for state, action, reward, next_state, done in minibatch:
            y_i = self.model.predict(state)

            if not DDQN:
                #Vanilla DQN
                y_i[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])*np.invert(done)
            else:
                #Double DQN
                action_index = np.argmax(self.model.predict(next_state)[0])
                y_i[0][action] = reward + self.gamma * self.target_model.predict(next_state)[0][action_index] * np.invert(done)

            X.append(state[0])
            Y.append(y_i[0])

        self.model.fit(np.array(X), np.array(Y), batch_size=len(X), verbose=0)

    def take_action(self, states, argmax_action = True):
        #Takes either a random action using greedy-e or uses the argmax of the predicted action Q values
        if np.random.rand(1) < self.epsilon:
            return self.environment.action_space.sample()
        else:
            if argmax_action: return np.argmax(self.model.predict(states))
            else: return self.model.predict(states)

    def update_epsilon(self, return_epsilon = False):
        #This Function decay's the epsilon ever time it's run (which is once after each episode)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return self.epsilon

    def reset_epsilon(self, episode, e_reset, reset_decay=0.85):
            #this function resets epsilon to a decreased max epsilon value using a counter
        if episode is 0: self.reset_counter = 0
        if episode % e_reset is 0 and episode > 0:
            self.reset_counter += 1
            self.epsilon = 1.0 * reset_decay**self.reset_counter

    def reset_epsilon_2(self, episode, reset_decay=0.85):
        #This function resets epsilon to a decreased max epsilon value using a threshold
        if episode is 0: self.reset_counter = 0
        if self.epsilon < 1.2*self.epsilon_min:
            self.reset_counter += 1
            self.epsilon = 1.0 * reset_decay**self.reset_counter

    def save_model(self, file_name):
        self.model.save_weights(file_name)

    def load_model(self, file_name):
        self.model.load_weights(file_name)
