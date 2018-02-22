"""
Created on Sep 04, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import datetime
import random

import numpy as np
import tensorflow as tf

# Local imports
import simengine
import config

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RNNConfig(object):
    h_size = 512  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
    alpha = 0.05  # Learning rate
    beta = 0.001  # Average reward learning rate
    gamma = 0.9   # Reward decay rate


class RNNAgent(object):
    total_drone_num = 0
    max_people_tracked = 4
    static_goal_num = 0

    r_bar = 0

    lstm = None
    session = None
    train = None
    saver = None

    state_size = 0
    input_size = 0
    max_goal_size = 0
    inputs = None
    q_values_tf = None
    r_tf = None
    delta_tf = None

    test = True

    if not test:
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        os.mkdir(os.path.join(config.Paths().tmp_root, 'rnn', timestamp))

    def __init__(self, index, handle):
        self.__class__.total_drone_num += 1
        self._index = index
        self._handle = handle

        self.epsilon = 0.0 if self.__class__.test else 0.1

        self.last_state = None
        self.last_q = 0
        self.last_action = -1

    @classmethod
    def set_static_goals(cls, goals):
        cls.static_goal_num = len(goals[0]) + len(goals[1]) + len(goals[2])

    @classmethod
    def init_lstm(cls):
        cls.max_goal_size = cls.static_goal_num + cls.max_people_tracked

        cls.state_size = 2 * (cls.total_drone_num + cls.max_people_tracked - 1)
        cls.input_size = cls.state_size + cls.max_goal_size
        cls.inputs = tf.placeholder(tf.float32, [None, None, cls.input_size], name='input')

        lstm = tf.contrib.rnn.LSTMCell(num_units=RNNConfig.h_size, state_is_tuple=True)
        batch_size = tf.shape(cls.inputs)[1]
        initial_state = lstm.zero_state(batch_size, tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(lstm, cls.inputs, initial_state=initial_state, time_major=True)
        cls.q_values_tf = tf.contrib.layers.fully_connected(rnn_outputs, num_outputs=128, activation_fn=tf.nn.relu)
        cls.q_values_tf = tf.contrib.layers.fully_connected(cls.q_values_tf, num_outputs=32, activation_fn=tf.nn.relu)
        # cls.q_values_tf = tf.contrib.layers.fully_connected(cls.q_values_tf, num_outputs=1, activation_fn=tf.nn.tanh)
        cls.q_values_tf = tf.contrib.layers.fully_connected(cls.q_values_tf, num_outputs=1, activation_fn=None)

        cls.delta_tf = tf.placeholder(tf.float32, name='delta')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=RNNConfig.alpha)
        grads_and_vars = optimizer.compute_gradients(cls.q_values_tf)
        grads_and_vars = [(- gv[0] * cls.delta_tf, gv[1]) for gv in grads_and_vars]
        cls.train = optimizer.apply_gradients(grads_and_vars)

        # # Direct approximation of reward
        # cls.r_tf = tf.placeholder(tf.float32, name='r')
        # loss = tf.square(cls.q_values_tf - cls.r_tf)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=RNNConfig.alpha)
        # grads_and_vars = optimizer.compute_gradients(loss)
        # cls.train = optimizer.apply_gradients(grads_and_vars)

        cls.session = tf.Session()
        cls.session.run(tf.global_variables_initializer())

        # file_writer = tf.summary.FileWriter('/home/siyuan/Downloads/test/log', cls.session.graph)
        cls.saver = tf.train.Saver()

        if cls.test:
            cls.saver.restore(cls.session, os.path.join(config.Paths().tmp_root, 'rnn', '2017-09-08 18:24:57', 'model_0190000_249_1016.ckpt'), )

    def act(self, goals, state, reward, episode_done = False):
        if reward == 0 and np.random.randint(500) != 1 and self.last_action != -1:
            return self.last_action

        goals_list = [k for i in range(len(goals)) for k, v in goals[i].items()]

        agent_state = state[1][:]
        # Swap the current agent's position to the first
        agent_state[0], agent_state[self._index*2] = agent_state[self._index*2], agent_state[0]
        agent_state[1], agent_state[self._index*2+1] = agent_state[self._index*2+1], agent_state[1]

        lstm_input = np.zeros((1, 1, self.__class__.input_size))
        if len(agent_state) < self.__class__.state_size:
            lstm_input[0, 0, :len(agent_state)] = np.array(agent_state)
        else:
            lstm_input[0, 0, :self.__class__.state_size] = np.array(agent_state)[:self.__class__.state_size]
        lstm_input /= 210.

        # RNN outputs Q(s, a)
        q_values = np.zeros(len(goals_list)) if len(goals_list) < self.__class__.max_goal_size else np.zeros(self.__class__.max_goal_size)
        for i in range(len(q_values)):
            lstm_input[0, 0, self.__class__.state_size:] = 0
            lstm_input[0, 0, self.__class__.state_size+i] = 1
            q_values[i] = self.__class__.session.run(self.__class__.q_values_tf, {self.__class__.inputs: lstm_input})[0][0]

        # Epsilon Greedy
        if np.random.random() < self.epsilon:
            a = np.random.choice(len(q_values))
        else:
            a = random.choice(np.argwhere(q_values == np.max(q_values)).flatten())
        # lstm_input[0, 0, -1] = goals_list[a]
        lstm_input[0, 0, self.__class__.state_size:] = 0
        lstm_input[0, 0, self.__class__.state_size + a] = 1

        # # Boltzmann distribution
        # softmax_q_values = np.exp(q_values)
        # if np.isinf(sum(softmax_q_values)):
        #     softmax_q_values = [1./len(q_values) for _ in range(len(q_values))]
        # else:
        #     softmax_q_values /= sum(softmax_q_values)
        # a = np.random.choice(len(softmax_q_values), p=softmax_q_values)
        # lstm_input[0, 0, -1] = goals_list[a]

        # Update network
        if reward != 0 and self.epsilon != 0:
            # delta = reward - self.__class__.r_bar + q_values[a] - self.last_q  # Continuing
            if episode_done:
                print 'Episode done!'
                delta = reward - q_values[a]
            else:
                delta = reward - RNNConfig.gamma * q_values[a] - self.last_q  # Episodic
            self.__class__.r_bar += RNNConfig.beta * delta
            self.__class__.session.run(self.__class__.train, {self.__class__.inputs: self.last_state, self.__class__.delta_tf: delta})
            # self.__class__.session.run(self.__class__.train, {self.__class__.inputs: self.last_state, self.__class__.r_tf: reward})
            print 'reward', reward, 'r_bar', self.__class__.r_bar, 'a', a, 'q', q_values[a], 'last_q', self.last_q, 'delta', delta, self.last_state
            print q_values

        self.last_state = lstm_input
        self.last_q = q_values[a]
        self.last_action = goals_list[a]
        print self._index, 'choose action', self.last_action
        return goals_list[a]

    @classmethod
    def save_model(cls, iteration, performance):
        if not cls.test:
            paths = config.Paths()
            cls.saver.save(cls.session, os.path.join(paths.tmp_root, 'rnn', cls.timestamp, 'model_{:07d}_{:03d}_{:03d}.ckpt'.format(iteration, performance[0], performance[1])))


def main():
    # # Testing RNN agent
    # tf.set_random_seed(0)
    # np.random.seed(0)
    # static_goals = [{i: i for i in range(4)}, {i+4: i+4 for i in range(4)}, {i+8: i+8 for i in range(1)}]
    # RNNAgent.set_static_goals(static_goals)
    # agent = RNNAgent(0, 20)
    # RNNAgent.init_lstm()
    #
    # reward = 0
    # for _ in range(10000):
    #     print 'iteration', _
    #     # a = agent.act(static_goals, [[20, 100], [10, 10, 10, 10]], reward)
    #     a = agent.act(static_goals, [[20, 100], [0, 0, 0, 0]], reward)
    #     reward = 1 if a == 3 else -1

    # Simulation for learning
    engine = simengine.Engine()
    engine.connect()
    if engine.connect():
        engine.start()

        # Initialization
        RNNAgent.set_static_goals(engine.get_goals())
        agents = [RNNAgent(i, handler) for i, handler in enumerate(engine.get_agents())]
        agent_num = len(agents)
        RNNAgent.init_lstm()

        iteration = 0
        i_episode = 0
        episode_done = False

        goals, state, reward = engine.step([-1 for _ in range(agent_num)])
        while not state:
            goals, state, reward = engine.step([-1 for _ in range(agent_num)])
        actions = [agents[i].act(goals, state, reward[i]) for i in range(agent_num)]

        while True:
            performance = engine.get_performance()
            if state and performance:
                if performance[1] == 10 * (i_episode + 1):
                    i_episode += 1
                    episode_done = True
                else:
                    episode_done = False

                actions = [agents[i].act(goals, state, reward[i], episode_done) for i in range(agent_num)]

                if iteration % 500 == 0 or not all(r == 0 for r in reward):
                    print 'iteration:', iteration, 'reward:', reward, 'performance', performance, 'state:', state
                    print 'actions', actions

            if iteration % 5000 == 0:
                RNNAgent.save_model(iteration, performance)
            iteration += 1

            goals, state, reward = engine.step(actions)
            actions = None

    engine.disconnect()


if __name__ == '__main__':
    main()
