import gym
import numpy as np
import random
import tensorflow as tf
import math
import time
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.optimizers import SGD, Adam

import GPyOpt

############### Author ################
## Brian Armstrong, github.com/molomono

plot = True                     #Turn off plot when tuning hyperparameters or you will have to kill all the plots every iteration of the optimizer
tune_hyperparameters = False    #If True the optimizer will be constructed and run. DISCLAMER; Optimizing will take around 4 hours

'''
This script is an implementation of the GPyOpt algorithm to tune the hyperparameters of a Double Deep Q-Learning network,
The way it works:
1. The DDQN is an agent class that can be constructed using select hyperparameters,
2. The Agent is wrapped and trained in a function that uses the CartPole-V0 enviroment NN_wrapper(X) = -score, for X is a tuple of hyperparameters defined the domain of the GPyOptimizer
the score is the performance of the NN (i'm simply using the average of the past 100 samples at the end of each training session, How the score is defined can be greatly improved)
3. The NN_wrapper function is passed to the GPyOpt.method as the function to optimize.
'''

'''
Hyper parameter space box constraints
e_decay: 0.999-0.9: Real
e_min: 0.05 - 0.0:  Real
BATCH_max: 2^x: X=integer, 3-8
h_layers['layers']=[8-48, 8-48], integers
h_layers['activation']: tanh, relu, linear
gamma = 1-0.8: real
learning_rate: 0.05-0: real
lr_decay: 0.05-0: real

Evaluate Outputs:
Average score
How quicly the AI breaks threshold score(180) with the smoothed score
'''

from Agent_DDQN_V0_8_0 import *


def set_dtypes(X):
    #TEST THIS FUNCTION, should be used to change the datatypes of the domain to the correct datatypes to be used by the wrapper algorithm
    [e_decay,e_min,gamma,lr,lr_d,BATCH_max,layer1,layer2,layer1_a,layer2_a] = X

    BATCH_max   =   int(BATCH_max)
    layer1      =   int(layer1)
    layer2      =   int(layer2)
    layer1_a    =   int(layer1_a)
    layer2_a    =   int(layer2_a)
    layer_activations = ['tanh', 'relu', 'linear']

    h_layers = {'layers':[layer1,layer2], 'activation':[layer_activations[layer1_a], layer_activations[layer2_a]], 'initializer':['he_normal', 'he_normal']}

    return [e_decay,e_min,gamma,lr,lr_d,BATCH_max,layer1,layer2,layer1_a,layer2_a], h_layers



def movingaverage(r_list, n):
    r_mean = []
    for i in range(n/2):
        r_mean.append(0)
    for i in range(len(r_list) - n):
        r_mean.append( sum(r_list[i:i+n])/n)
    return r_mean

def NN_wrapper(X, num_episodes = 500):
    #inital parameters
    env = gym.make('CartPole-v0')
    #num_episodes = 100
    h_layers = {'layers':[24,48], 'activation':['tanh','tanh'], 'initializer':['he_normal', 'he_normal']}

    #Set learning parameters
    gamma = 0.99
    e = 1
    e_decay = 0.95#95
    e_min = 0.01
    e_reset_decay = 0.825
    epsilon_reset_interval = 40
    BATCH_max = 32
    episodes_till_target_update = 5

    '''
    print X[0]
    #Overwrite the initial parameters with parameters passed through the Optimizer equation
    [e_decay,e_min,gamma,lr,lr_d,BATCH_max,layer1,layer2,layer1_a,layer2_a] = X[0]

    BATCH_max   =   int(BATCH_max)
    layer1      =   int(layer1)
    layer2      =   int(layer2)
    layer1_a    =   int(layer1_a)
    layer2_a    =   int(layer2_a)
    layer_activations = ['tanh', 'relu', 'linear']

    h_layers = {'layers':[layer1,layer2], 'activation':[layer_activations[layer1_a], layer_activations[layer2_a]], 'initializer':['he_normal', 'he_normal']}
    '''

    [X, h_layers] = set_dtypes(X[0])
    [e_decay,e_min,gamma,lr,lr_d,BATCH_max,layer1,layer2,layer1_a,layer2_a] = X

    #construct nn
    qnet = DDQN_Agent(4, 2, environment=env, hidden_layers=h_layers,
                    epsilon=e, epsilon_min=e_min, epsilon_decay=e_decay,
                    batch_size=BATCH_max, gamma=gamma, learning_rate=lr, lr_decay=lr_d)

    load = False
    simulation_mode = False
    train = True

    jList = []
    r_list = []
    e_list = []

    loss = 0
    completed = False

    with tf.Session() as sess:
        from keras import backend as K
        K.set_session(sess)
        qnet.update_target()

        if load:
            qnet.load_model('solved.h5')
            qnet.update_target()

        if simulation_mode:
            qnet.epsilon = 0.01
            states = env.reset()
            states = np.reshape(states, [1, states.size])
            completed = True
            for _ in range(1000):
                time.sleep(0.01)
                action = qnet.take_action(states)
                next_states, reward, done, _ = env.step(action)
                env.render()
                next_states = np.reshape(next_states, [1, states.size])
                qnet.add_experience(states, action, reward, next_states, done)
                states = next_states

        while completed or not train:
            print crash
            break

        j_total = 0
        for i in range(num_episodes):
            #Reset environment
            states = env.reset()
            states = np.reshape(states, [1, states.size])
            #print states.shape
            r_sum = 0
            done = False
            j = 0

            #Let the simulation run:
            while j < 199:
                action = qnet.take_action(states)
                next_states, reward, done, _ = env.step(action)
                next_states = np.reshape(next_states, [1, states.size])
                qnet.add_experience(states, action, reward, next_states, done)

                states = next_states

                r_sum += reward
                j+=1
                j_total+=1

                if done:
                    break

            #Experience replay after ever episode, this is where the leraning happens
            qnet.replay()
            #update the epsilon value every episode
            e = qnet.update_epsilon(return_epsilon=True)
            #This equation addes discontinues behavior in hte form of a saw blade descent of epsilon. drastically improving the convergance speed and reliability
            qnet.reset_epsilon(i, epsilon_reset_interval, e_reset_decay)

            if i % episodes_till_target_update: qnet.update_target()

            e_list.append(e)
            #print(r_sum) 
            r_list.append(r_sum)

            if i > 100:
                r_list_smooth_100 = movingaverage(r_list, 100)
                for k in range(len(r_list_smooth_100)):
                    if r_list_smooth_100[k] > 195:
                        print 'The Agent has solved the enviroment'
                        completed = True
                        #qnet.save_model('solved.h5')
                        #print crash

            if completed:
                break

    if plot:
        print completed
        plt.figure(1)
        plt.title('EndScore Per Episode')
        plt.plot(range(len(r_list)),r_list)

        plt.figure(2)
        plt.title('Smoothed end Score Per Episode')
        #still need to add a offset to the moving average so it doesn't cut the front of the list off
        r_list_smooth = movingaverage(r_list, 10)
        r_list_smooth2 = movingaverage(r_list, 40)
        plt.plot(r_list_smooth)
        plt.plot(r_list_smooth2)


        plt.figure(3)
        plt.title('Greedy epsilon value')
        plt.plot(e_list)

        plt.show()

    #admittedly this is a very poor performance metric, but this is my first time tuning AI hyperparameters
    #and this metric works well enough to evaluate the optimizer, in future work i'll be using better methods
    ai_performance = r_list_smooth_100[len(r_list_smooth_100)-1]
    print "Score: ", ai_preformance
    print "Using X: ", X

    return -ai_performance




if __name__=="__main__":
    #define the bounding box for the hyperparameters
    bounds = [  {'name': 'e_d',     'type': 'continuous', 'domain': (0.9,0.9999)},
                {'name': 'e_m',     'type': 'continuous', 'domain': (0.0,0.05)},
                {'name': 'gamma',   'type': 'continuous', 'domain': (0.8,1.0)},
                {'name': 'lr',      'type': 'continuous', 'domain': (0.0,0.05)},
                {'name': 'lr_d',    'type': 'continuous', 'domain': (0.0,0.05)},
                {'name': 'b_m',     'type': 'discrete',   'domain': (16, 32, 64, 128, 256)},
                {'name': 'layer1',  'type': 'discrete',   'domain': (20,24,28,32,36,40)},
                {'name': 'layer2',  'type': 'discrete',   'domain': (20,24,28,32,36,40)},
                {'name': 'layer1_a','type': 'categorical','domain': (0,1,2)}, #['tanh','relu','linear']
                {'name': 'layer2_a','type': 'categorical','domain': (0,1,2)}]

    if tune_hyperparameters:
        #Configure optimizer and set the number of optimization steps
        max_iter = 25
        ai_optimizer = GPyOpt.methods.BayesianOptimization(NN_wrapper, domain=bounds,
                                                        initial_design_numdata = 8,   # Number data initial design
                                                        Initial_design_type = 'latin',
                                                        model_type= 'GP_MCMC',
                                                        acquisition_type='EI_MCMC',
                                                        normalize_Y = True) #http://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/devel/manual/GPyOpt_mixed_domain.ipynb
        #Run optimizer
        ai_optimizer.run_optimization(max_iter)
        # Evaluate using ai_optimizer.plot_convergence()
        ai_optimizer.plot_convergence()
        # All the hyperparameters come out of the optimization as float64 set_dtypes corrects the datatypes
        [x_optimum, h_layers] = set_dtypes(ai_optimizer.x_opt)

        #The estimated optimum is printed and saved to a file
        print "Best performance: ", x_optimum
        #Save best performance to a file
        file_name = 'Optimzed_performance_variables.csv'
        df = pd.DataFrame([x_optimum], columns=["epsilon_decay", "epsilon_min", "gamma", "learning_rate", "learning_rate_decay", "Batch_size", "layer1", "layer2", "layer1_activation", "layer2_activation"])
        df.to_csv(file_name, index=False)

    else:
        #Function for evaluation, Using pandas this can be automated but for now i manual input the estimated optimum hyperparameters
        X = np.array([0.9427935559203139, 0.03627109780242037, 0.9593081233219476, 0.023057169655664934, 0.011163645947286423, 128, 36, 28, 0, 1])
        NN_wrapper([X])
