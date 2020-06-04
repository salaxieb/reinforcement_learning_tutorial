import numpy as np
import random
from time import sleep

SMALL_VALUE = 0.001
GAMMA = 0.9

class Environment():
    def __init__(self):
        self.possible_actions = {
            (0,0): ['U', 'R'],
            (0,1): ['D', 'U'],
            (0,2): ['D', 'R'],
            (1,0): ['L', 'R'],
            (1,2): ['L', 'R'],
            (2,0): ['L', 'U', 'R'],
            (2,1): ['D', 'U', 'R'],
            (2,2): ['L', 'D', 'R'],
            (3,0): ['U', 'L'],
            (3,2): None,
            (3,1): None
        }

    def draw_value_function(self, value_funcition):
        for j in range(3):
            j = 2 - j
            print('-'*58)
            s = ''
            for i in range(4):
                value = 0
                if (i,j) in value_funcition:
                    value = value_funcition[(i,j)]
                s += ' %3.2f \t |' % value
            print(s)
        print('-'*58)

    def draw_policy(self, policy):
        for j in range(3):
            j = 2 - j
            print('-'*34)
            s = ''
            for i in range(4):
                p = ' '
                if (i,j) in policy:
                    p = policy[(i,j)]
                s += ' %s \t |' % p
            print(s)
        print('-'*34)

    def state_reward(self, state):
        if state == (3,2):
            return 1
        if state == (3,1):
            return -1
        return -0.1

    def next_state_after_action(self, state, action):
        #valid action
        assert action in ['U', 'D', 'L', 'R']
        if action == 'U':
            state = (state[0], state[1]+1)
        if action == 'D':
            state = (state[0], state[1]-1)
        if action == 'L':
            state = (state[0]-1, state[1])
        if action == 'R':
            state = (state[0]+1, state[1])
        #new state is valid
        #assert state in self.possible_actions
        return state


class Agent():
    def __init__(self, env):
        self.value_function = {}
        #self.policy = {state:random.sample(env.possible_actions[state], 1)[0] if len(env.possible_actions[state]) else None for state in env.possible_actions}
        self.policy = {
            (0,2): 'R', (1,2): 'R', (2,2): 'R', (3,2): None,
            (0,1): 'U',             (2,1): 'R', (3,1): None,
            (0,0): 'U', (1,0): 'R', (2,0): 'U', (3,0): 'L'
        }

    def get_value_functions(self, state):
        if state not in self.value_function:
            self.value_function[state] = 0
        return self.value_function[state]

    def iteratively_find_value_function_for_policy(self, env):
        value_function_changed_flag = True
        while value_function_changed_flag:
            value_function_changed_flag = False
            for state in env.possible_actions:
                before = self.get_value_functions(state)
                optimal_action = self.policy[state]
                if env.possible_actions[state]:
                    v = 0
                    for action in env.possible_actions[state]:
                        if action == optimal_action:
                            p = 0.7
                        else:
                            p = 0.3/(len(env.possible_actions[state])-1)
                        new_state = env.next_state_after_action(state, action)
                        v += p * (env.state_reward(new_state) + GAMMA*self.get_value_functions(new_state))
                    after = v
                    self.value_function[state] = v
                    if abs(before-after) > SMALL_VALUE: value_function_changed_flag=True

    def optimize_policy(self, env):
        policy_changed_flag = True
        while policy_changed_flag:
            env.draw_policy(self.policy)
            env.draw_value_function(self.value_function)
            #sleep(3)
            policy_changed_flag = False
            self.iteratively_find_value_function_for_policy(env)
            for state in self.policy:
                current_action = self.policy[state]
                current_expected_value = self.get_value_functions(state)
                if env.possible_actions[state]:
                    for possible_action in env.possible_actions[state]:
                        #action = self.get_random_policy(state, env)
                        next_state = env.next_state_after_action(state, possible_action)
                        v = 0
                        for action in env.possible_actions[state]:
                            if action == possible_action:
                                p = 0.7
                            else:
                                p = 0.3/(len(env.possible_actions[state])-1)
                            v += p * (env.state_reward(next_state) + GAMMA*self.get_value_functions(next_state))

                        if v>current_expected_value and possible_action!=self.policy[state]:
                            print(v, current_expected_value)
                            self.policy[state] = possible_action
                            current_expected_value = v
                            policy_changed_flag = True

env = Environment()
agent = Agent(env)
# env.draw_value_function(agent.value_function)
# agent.iteratively_find_value_function_for_policy(env)
# env.draw_value_function(agent.value_function)
env.draw_policy(agent.policy)
agent.optimize_policy(env)
env.draw_value_function(agent.value_function)
env.draw_policy(agent.policy)
