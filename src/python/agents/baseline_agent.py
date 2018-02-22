"""
Created on Aug 21, 2017

@author: Siyuan Qi

Description of the file.

"""

import random
import numpy as np

import simengine


class RandomAgent(object):
    total_agent_num = 0

    """The world's simplest agent!"""
    def __init__(self, index, handle):
        print 'Initializing agent', index, handle
        self.__class__.total_agent_num += 1
        self._index = index
        self._handle = handle

        self.last_action = -1

    def act(self, goals, state, reward):
        if self.last_action != -1 and np.random.randint(500) != 1:
            return self.last_action

        goals_list = [k for i in range(len(goals)) for k, v in goals[i].items()]

        # # Random agent
        # self.last_action = random.choice(goals_list)

        # Greedy agent
        if goals[3]:
            self.last_action = random.choice(goals[3].keys())
            goals[3].pop(self.last_action, None)
        else:
            self.last_action = random.choice(goals_list)

        return self.last_action


def main():
    engine = simengine.Engine()
    engine.connect()
    if engine.connect():
        engine.start()
        agents = [RandomAgent(i, handler) for i, handler in enumerate(engine.get_agents())]
        agent_num = len(agents)
        actions = [-1 for _ in range(agent_num)]
        goals, state, reward = engine.step(actions)
        iteration = 0
        while True:
            if iteration % 500 == 0:
                performance = engine.get_performance()
                print 'iteration:', iteration, 'reward:', reward, 'performance', performance, 'state:', state
            if state:
                actions = [agents[i].act(goals, state, reward) for i in range(agent_num)]
                iteration += 1
            goals, state, reward = engine.step(actions)

    engine.disconnect()


if __name__ == '__main__':
    main()
