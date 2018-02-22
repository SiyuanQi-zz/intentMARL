"""
Created on Sep 13, 2017

@author: Siyuan Qi

Description of the file.

"""


import os
import time
import datetime
import bisect
import random
import Queue
import numpy as np
import scipy.ndimage
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt

# For ROS
import cv2
import rospy
import sensor_msgs.msg
import cv_bridge

# Local imports
import simengine
import config
import path_utils


class IntentAgent(object):
    total_drone_num = 0
    human_goal_num = 0
    static_goal_num = 0
    # static_goals = None

    agent_type_num = 3
    goal_type_num = 4

    # Modified dynamically
    intent_probability_matrix = None
    goals_list = None
    goal_types = None

    # Reinforcement learning parameters
    # Agent type 0: self, 1: drones, 2: humans
    # Goal type 0: doors, 1: exits, 2: monitors, 3:humans
    theta = np.zeros((agent_type_num, goal_type_num, goal_type_num*2))
    # theta = np.random.rand(agent_type_num, goal_type_num, goal_type_num*2)
    r_bar = 0

    decay_rate = 0.95
    alpha = 0.05
    beta = 0.05
    epsilon = 0.2

    paths = config.Paths()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    os.mkdir(os.path.join(paths.tmp_root, 'theta', timestamp))

    visualize = True
    if visualize:
        all_cost_maps = np.load(os.path.join(paths.tmp_root, 'cost_maps', 'city_all.npy'))

    height_map = np.load(os.path.join(paths.tmp_root, 'height_maps', 'city.npy'))
    height_map_color = matplotlib.cm.Blues(height_map)
    height_map_color = height_map_color[:, :, :3]
    height_map_color *= 255
    height_map_color = height_map_color.astype('uint8')

    rospy.init_node('agents', anonymous=False)
    detection_pub = rospy.Publisher("detection", sensor_msgs.msg.Image, queue_size=5)
    map_pub = rospy.Publisher("map", sensor_msgs.msg.Image, queue_size=5)
    bridge = cv_bridge.CvBridge()

    def __init__(self, index, handle):
        print 'Initializing agent', index, handle
        self.__class__.total_drone_num += 1
        self._index = index
        self._handle = handle

        self.last_action = -1
        self.last_action_type = -1
        self.last_gradient = np.zeros_like(self.__class__.theta)

        sarsa_n = 5
        self.gamma = 1 - sarsa_n
        self.reward_list = list()
        self.action_type_queue = Queue.Queue()
        self.gradient_queue = Queue.Queue()

        # For visualization
        self.height_thresh = 0.02
        self.dilate_size = 5
        self.blur_sigma = 4

    @classmethod
    def set_static_goals(cls, goals):
        # cls.static_goals = goals
        cls.human_goal_num = len(goals[0]) + len(goals[1])
        cls.static_goal_num = cls.human_goal_num + len(goals[2])

    @classmethod
    def compute_intent_probability_matrix(cls, goals, state):
        goal_prob = 1.0
        agent_num = len(state[0])/2

        cls.goals_list = [k for i in range(len(goals)) for k, v in goals[i].items()]
        cls.goal_types = [i for i in range(len(goals)) for _, _, in goals[i].items()]
        cls.intent_probability_matrix = np.zeros((cls.static_goal_num + agent_num - cls.total_drone_num, agent_num))

        position_list = [[int(v[0]), int(v[1])] for i in range(len(goals)) for k, v in goals[i].items()]
        for i in range(agent_num):
            goal = state[0][2*i+1]

            agent_pos = [int(state[1][2 * i]), int(state[1][2 * i + 1])]
            true_goal_pos = position_list[cls.goals_list.index(goal)] if goal != -1 else agent_pos
            probs = np.zeros(len(position_list))
            for g, goal_pos in enumerate(position_list):
                if i >= cls.total_drone_num and g >= cls.human_goal_num:
                    # TODO: clip when normalizing
                    break

                probs[g] -= np.sqrt((true_goal_pos[0]-goal_pos[0])**2 + (true_goal_pos[1]-goal_pos[1])**2)/10
                probs[g] -= cls.all_cost_maps[agent_pos[0], agent_pos[1], goal_pos[0], goal_pos[1]]
            probs = np.exp(probs/50)
            if i >= cls.total_drone_num:
                probs[cls.human_goal_num:] = 0
            probs /= np.sum(probs)
            cls.intent_probability_matrix[:, i] = probs

    def compute_q_gradient(self, goals, state, chosen_goal):
        gradient = np.zeros_like(self.__class__.theta)

        chosen_goal_type = self.__class__.goal_types[chosen_goal]
        for goal_i in range(self.__class__.intent_probability_matrix.shape[0]):
            for agent_i in range(self.__class__.intent_probability_matrix.shape[1]):
                goal_type = self.__class__.goal_types[goal_i]

                if agent_i == self._index:
                    if chosen_goal == goal_i:
                        gradient[0, goal_type, goal_type] = 1
                    continue
                if agent_i < self.__class__.total_drone_num:
                    agent_type = 1
                else:
                    agent_type = 2

                if chosen_goal == goal_i:
                    gradient[agent_type, chosen_goal_type, self.goal_type_num+goal_type] += self.__class__.intent_probability_matrix[goal_i, agent_i]
                else:
                    gradient[agent_type, chosen_goal_type, goal_type] += self.__class__.intent_probability_matrix[goal_i, agent_i]

        return gradient

    def compute_q(self, goals, state):
        agent_num = len(state[0])/2

        current_position = np.array((state[1][2*self._index], state[1][2*self._index+1]))
        q = np.zeros(len(self.__class__.goals_list))
        costs = np.zeros_like(q)
        for i in range(q.shape[0]):
            goal_type = self.__class__.goal_types[i]

            # Goal reward
            q[i] = self.__class__.theta[0, goal_type, goal_type]

            # Reward influence by other agents
            for agent_i in range(agent_num):
                if agent_i == self._index:
                    continue
                if agent_i < self.__class__.total_drone_num:
                    agent_type = 1
                else:
                    agent_type = 2

                weight = [self.__class__.theta[agent_type, goal_type, other_agent_goal_type] for other_agent_goal_type in
                          self.__class__.goal_types]
                weight[i] = self.__class__.theta[agent_type, goal_type, self.__class__.goal_type_num+goal_type]
                q[i] += np.dot(np.array(weight), self.__class__.intent_probability_matrix[:, agent_i])

            # Compute cost
            goal_position = goals[goal_type][self.__class__.goals_list[i]]
            costs[i] = np.linalg.norm(current_position - goal_position)/100

        return q, costs

    def publish_detection(self, goals, state):
        agent_num = len(state[0])/2

        intent_prob_matrix = np.copy(self.__class__.intent_probability_matrix)
        for i in range(agent_num):
            intent_prob_matrix[:, i] /= np.max(intent_prob_matrix[:, i])
            intent_prob_matrix[:, i] *= 2

        position_list = [[int(v[0]), int(v[1])] for i in range(len(goals)) for k, v in goals[i].items()]
        heat_map = np.zeros_like(self.__class__.height_map)
        for i in range(agent_num):
            if i == self._index:
                continue

            agent_pos = [int(state[1][2*i]), int(state[1][2*i+1])]

            for g, goal_pos in enumerate(position_list):
                prob = intent_prob_matrix[g, i]
                path = path_utils.path_backtrace(self.__class__.all_cost_maps[agent_pos[0], agent_pos[1], :, :], goal_pos)

                for point in path:
                    heat_map[point[0], heat_map.shape[1]-1-point[1]] = prob
            heat_map[agent_pos[0], heat_map.shape[1]-1-agent_pos[1]] += 1.5

        heat_map = scipy.ndimage.grey_dilation(heat_map, size=(self.dilate_size, self.dilate_size))
        heat_map = scipy.ndimage.gaussian_filter(heat_map, sigma=(self.blur_sigma, self.blur_sigma))
        heat_map[self.__class__.height_map > 0.5] = 0
        heat_map_color = matplotlib.cm.BuGn(heat_map)[:, :, :3] * 255
        heat_map_color = heat_map_color.astype('uint8')
        height_map_color = np.copy(self.__class__.height_map_color)
        height_map_color[heat_map > self.height_thresh, :] = heat_map_color[heat_map > self.height_thresh, :]

        for i in range(agent_num):
            if i == self._index:
                continue
            agent_pos = [int(state[1][2*i]), int(state[1][2*i+1])]
            height_map_color = cv2.circle(height_map_color, (heat_map.shape[1]-1-agent_pos[1], agent_pos[0]), 5, (0, 0, 0), 2)

        img_msg = self.__class__.bridge.cv2_to_imgmsg(height_map_color, "rgb8")
        img_msg.header.frame_id = str(self._index-1)
        self.__class__.detection_pub.publish(img_msg)

    def publish_q(self, goals, state, q, costs):
        q -= np.min(q)
        q = q * 4 / np.max(q)

        position_list = [[int(v[0]), int(v[1])] for i in range(len(goals)) for k, v in goals[i].items()]

        agent_pos = [int(state[1][2 * self._index]), int(state[1][2 * self._index + 1])]
        best_action = np.argmax(q-costs)
        goal_pos = position_list[best_action]
        path_map = np.zeros_like(self.__class__.height_map)
        path = path_utils.path_backtrace(self.__class__.all_cost_maps[agent_pos[0], agent_pos[1], :, :], goal_pos)
        for point in path:
            path_map[point[0], path_map.shape[1]-1-point[1]] = q[best_action]
        path_map = scipy.ndimage.grey_dilation(path_map, size=(self.dilate_size/2, self.dilate_size/2))
        path_map = scipy.ndimage.gaussian_filter(path_map, sigma=(self.blur_sigma, self.blur_sigma))
        path_map[self.__class__.height_map > 0.5] = 0
        path_map_color = matplotlib.cm.Reds(path_map)[:, :, :3] * 255
        path_map_color = path_map_color.astype('uint8')

        heat_map = np.zeros_like(self.__class__.height_map)
        for i, pos in enumerate(position_list):
            heat_map[pos[0], heat_map.shape[1]-1-pos[1]] = q[i]

        heat_map = scipy.ndimage.grey_dilation(heat_map, size=(self.dilate_size, self.dilate_size))
        heat_map = scipy.ndimage.gaussian_filter(heat_map, sigma=(self.blur_sigma, self.blur_sigma))
        heat_map[self.__class__.height_map > 0.5] = 0
        heat_map_color = matplotlib.cm.Reds(heat_map)[:, :, :3] * 255
        heat_map_color = heat_map_color.astype('uint8')

        height_map_color = np.copy(self.__class__.height_map_color)
        height_map_color[heat_map > self.height_thresh, :] = heat_map_color[heat_map > self.height_thresh, :]
        height_map_color[path_map > self.height_thresh, :] = path_map_color[path_map > self.height_thresh, :]

        height_map_color = cv2.circle(height_map_color, (heat_map.shape[1]-1-agent_pos[1], agent_pos[0]), 5, (0, 0, 0), 2)

        # scipy.misc.imsave(os.path.join(self.__class__.paths.tmp_root, 'intents', '{:05d}_{:01d}.png'.format(iteration, self._index)), height_map_color)

        img_msg = self.__class__.bridge.cv2_to_imgmsg(height_map_color, "rgb8")
        img_msg.header.frame_id = str(self._index-1)
        self.__class__.map_pub.publish(img_msg)

    def act(self, goals, state, reward):
        if self.__class__.visualize:
            q, costs = self.compute_q(goals, state)
            self.publish_detection(goals, state)
            self.publish_q(goals, state, q, costs)

        if reward == 0 and np.random.randint(50) != 1 and self.last_action != -1:
            return self.last_action

        q, costs = self.compute_q(goals, state)
        q -= costs

        # ****** Eplison Greedy ******
        if random.random() < self.__class__.epsilon:
            print 'random action with prob:', self.__class__.epsilon
            a = random.choice(range(len(q)))
        else:
            a = random.choice(np.argwhere(q == np.max(q)).flatten())

        print 'queue size:', len(self.reward_list), self.action_type_queue.qsize(), self.gradient_queue.qsize()

        if reward != 0:
            self.reward_list.append(reward)
            self.action_type_queue.put(self.last_action_type)
            self.gradient_queue.put(self.last_gradient)
            self.gamma += 1
            if self.gamma > 0:
                print 'reward', reward, 'r_bar', self.__class__.r_bar, self.__class__.alpha, self.__class__.beta
                print 'reward_list', self.reward_list
                reward_diff = 0
                for r in self.reward_list:
                    reward_diff += (r - self.__class__.r_bar)
                self.reward_list.pop(0)

                gradient_gamma = self.gradient_queue.get()
                action_type_gamma = self.action_type_queue.get()
                delta = reward_diff + q[a] - np.sum(np.sum(np.multiply(self.__class__.theta, gradient_gamma), 0), 1)[action_type_gamma]
                self.__class__.r_bar += self.__class__.beta * delta
                self.__class__.theta += self.__class__.alpha * delta * gradient_gamma
                print 'average reward:', self.__class__.r_bar, 'reward_diff', reward_diff, 'delta', delta, 'a', a, 'q', q, 'type', action_type_gamma, 'gradient', np.sum(np.sum(np.multiply(self.__class__.theta, gradient_gamma), 0), 1)

        self.last_action = self.__class__.goals_list[a]
        self.last_action_type = self.__class__.goal_types[a]
        self.last_gradient = self.compute_q_gradient(goals, state, a)
        print self._index, 'choose action', self.last_action
        print q[a], self.__class__.goal_types[a], np.sum(np.sum(np.multiply(self.__class__.theta, self.last_gradient), 0), 1)

        return self.__class__.goals_list[a]

    @classmethod
    def decrease_learning_rate(cls):
        cls.alpha *= cls.decay_rate
        cls.beta *= cls.decay_rate
        cls.epsilon *= cls.decay_rate

    @classmethod
    def save_model(cls, iteration, performance):
        paths = config.Paths()
        np.save(os.path.join(paths.tmp_root, 'theta', cls.timestamp, 'theta_{:07d}'.format(iteration)), cls.theta)
        with open(os.path.join(paths.tmp_root, 'theta', cls.timestamp, 'theta_{:07d}.csv'.format(iteration)), 'w') as f:
            for agent_theta in cls.theta:
                np.savetxt(f, agent_theta, fmt='%.5f', footer='====')
            np.savetxt(f, performance, fmt='%d')
            np.savetxt(f, [cls.r_bar], fmt='%.5f')


def main():
    np.random.seed(0)
    engine = simengine.Engine()
    engine.connect()
    if engine.connect():
        engine.start()

        # Initialization
        agents = [IntentAgent(i, handler) for i, handler in enumerate(engine.get_agents())]
        agent_num = len(agents)
        IntentAgent.set_static_goals(engine.get_goals())

        iteration = 0
        total_steps = 0

        goals, state, reward = engine.step([-1 for _ in range(agent_num)])
        while not state:
            goals, state, reward = engine.step([-1 for _ in range(agent_num)])
        IntentAgent.compute_intent_probability_matrix(goals, state)
        actions = [agents[i].act(goals, state, reward[i]) for i in range(agent_num)]

        # # Static strategy testing
        # actions = [-1 for _ in range(agent_num)]
        # actions[0] = agents[0].act(goals, state, reward[0])

        goal_num = len([k for i in range(len(goals)) for k, v in goals[i].items()])
        last_goal_num = goal_num

        while True:
            performance = engine.get_performance()
            if state:
                IntentAgent.compute_intent_probability_matrix(goals, state)
                # print IntentAgent.goal_list, IntentAgent.goal_types
                # print IntentAgent.intent_probability_matrix
                actions = [agents[i].act(goals, state, reward[i]) for i in range(agent_num)]

                # # Static strategy testing
                # actions = [-1 for _ in range(agent_num)]
                # actions[0] = agents[0].act(goals, state, reward[0])

                total_steps += 1

                if iteration % 500 == 0 or not all(r == 0 for r in reward):
                    print 'iteration:', iteration, 'steps:', total_steps, 'reward:', reward, 'performance', performance, 'state:', state
                    print 'actions', actions

                if total_steps % 1000 == 0:
                    IntentAgent.decrease_learning_rate()

            if not all(r == 0 for r in reward) or iteration % 5000 == 0:
                IntentAgent.save_model(iteration, performance)
            iteration += 1

            last_goal_num = goal_num
            goals, state, reward = engine.step(actions)
            goal_num = len([k for i in range(len(goals)) for k, v in goals[i].items()])
            actions = None

    engine.disconnect()


if __name__ == '__main__':
    main()

