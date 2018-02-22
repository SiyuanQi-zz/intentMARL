"""
Created on Aug 20, 2017

@author: Siyuan Qi

Description of the file.

"""

try:
    import vrep
except ImportError:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')

import numpy as np


class Engine(object):
    def __init__(self, simulation_rounds_each_action=1):
        self._rounds = simulation_rounds_each_action

        self._client_id = -1
        self._agents = None
        self._doors = None
        self._exits = None
        self._monitors = None
        self._static_goals = None

    # @property
    # def client_id(self):
    #     return self._client_id

    def connect(self, ip='127.0.0.1', socket=19997):
        vrep.simxFinish(-1)  # just in case, close all opened connections
        self._client_id = vrep.simxStart(ip, socket, True, True, 5000, 5)  # Connect to V-REP
        if self._client_id != -1:
            return True
        else:
            return False

    def disconnect(self):
        if self._client_id != -1:
            vrep.simxStopSimulation(self._client_id, vrep.simx_opmode_oneshot)

            # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive.
            vrep.simxGetPingTime(self._client_id)

        # Now close the connection to V-REP:
        vrep.simxFinish(self._client_id)
        self._client_id = -1
        self._agents = None
        self._doors = None
        self._exits = None
        self._monitors = None
        self._static_goals = None

    def reset(self, synchronous=True):
        vrep.simxStopSimulation(self._client_id, vrep.simx_opmode_oneshot)
        vrep.simxGetPingTime(self._client_id)
        vrep.simxSynchronous(self._client_id, synchronous)
        vrep.simxStartSimulation(self._client_id, vrep.simx_opmode_oneshot)

    def start(self, synchronous=True):
        vrep.simxSynchronous(self._client_id, synchronous)
        vrep.simxStartSimulation(self._client_id, vrep.simx_opmode_oneshot)
        self.get_agents()
        self.get_doors()
        self.get_exits()
        self.get_monitors()
        self._static_goals = [self._doors, self._exits, self._monitors]

    def get_agents(self):
        if not self._agents:
            res, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self._client_id, 'engine', vrep.sim_scripttype_childscript, 'getDrones', [], [], [], bytearray(), vrep.simx_opmode_blocking)
            if res == 0:
                self._agents = ret_ints
        return self._agents

    def get_doors(self):
        if not self._doors:
            res, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self._client_id, 'engine', vrep.sim_scripttype_childscript, 'getDoors', [], [], [], bytearray(), vrep.simx_opmode_blocking)
            if res == 0:
                self._doors = dict()
                for i in range(0, len(ret_ints)):
                    self._doors[ret_ints[i]] = np.array((ret_floats[2*i], ret_floats[2*i+1]))
        return self._doors

    def get_exits(self):
        if not self._exits:
            res, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self._client_id, 'engine', vrep.sim_scripttype_childscript, 'getExits', [], [], [], bytearray(), vrep.simx_opmode_blocking)
            if res == 0:
                self._exits = dict()
                for i in range(0, len(ret_ints)):
                    self._exits[ret_ints[i]] = np.array((ret_floats[2*i], ret_floats[2*i+1]))
        return self._exits

    def get_monitors(self):
        if not self._monitors:
            res, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self._client_id, 'engine', vrep.sim_scripttype_childscript, 'getMonitors', [], [], [], bytearray(), vrep.simx_opmode_blocking)
            if res == 0:
                self._monitors = dict()
                for i in range(0, len(ret_ints)):
                    self._monitors[ret_ints[i]] = np.array((ret_floats[2*i], ret_floats[2*i+1]))
        return self._monitors

    def get_goals(self, state=None):
        """

        :param state: ([agent1, goal_handle1, agent2, goal_handle2, ... ], [position_x1, position_y1, position_x2, position_y2, ...])
        :return: [door, exits, monitors, humans]
        """
        goals = self._static_goals[:]
        if state:
            moving_goals = dict()
            for i in range(2*len(self._agents), len(state[0]), 2):
                moving_goals[state[0][i]] = np.array((state[1][i], state[1][i+1]))
            goals.append(moving_goals)
        return goals

    def get_state(self):
        state = None
        res, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self._client_id, 'engine', vrep.sim_scripttype_childscript, 'getState', [], [], [], bytearray(), vrep.simx_opmode_oneshot)
        if res == 0:
            # print 'State return:', ret_ints, ret_floats
            state = ret_ints, ret_floats
        return state

    def get_reward(self):
        res, ret_ints, ret_floats, ret_strings, ret_buffer = \
            vrep.simxCallScriptFunction(self._client_id, 'engine', vrep.sim_scripttype_childscript, 'getReward', [], [], [], bytearray(), vrep.simx_opmode_oneshot)
        reward = ret_floats if res == 0 else None
        return reward

    def step(self, actions):
        if actions:
            vrep.simxCallScriptFunction(self._client_id, 'engine', vrep.sim_scripttype_childscript, 'act', actions, [], [], bytearray(), vrep.simx_opmode_blocking)

        vrep.simxSynchronousTrigger(self._client_id)
        vrep.simxGetPingTime(self._client_id)
        state = self.get_state()
        reward = self.get_reward()
        goals = self.get_goals(state)
        return goals, state, reward

    def get_performance(self):
        res, ret_ints, ret_floats, ret_strings, ret_buffer = \
            vrep.simxCallScriptFunction(self._client_id, 'engine', vrep.sim_scripttype_childscript, 'getPerformance', [], [], [], bytearray(), vrep.simx_opmode_blocking)
        performance = ret_ints if res == 0 else None
        return performance


def main():
    engine = Engine()
    engine.connect()
    if engine.connect():
        engine.start()

    engine.disconnect()


if __name__ == '__main__':
    main()
