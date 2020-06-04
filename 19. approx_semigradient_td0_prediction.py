import numpy as np
import random
from time import sleep

SMALL_VALUE = 0.001
GAMMA = 0.9
ALPHA = 1
LR = 0.01

class Environment():
    def __init__(self):
        self.possible_actions = {
            (0,2): ['D', 'R'],   (1,2): ['L', 'R'],     (2,2): ['L', 'D', 'R'],     (3,1): None,
            (0,1): ['D', 'U'],                          (2,1): ['D', 'U', 'R'],     (3,2): None,
            (0,0): ['U', 'R'],   (1,0): ['L', 'R'],     (2,0): ['L', 'U', 'R'],     (3,0): ['U', 'L']
        }

    def draw_value_function(self, agent):
        for j in range(3):
            j = 2 - j
            print('-'*58)
            s = ''
            for i in range(4):
                value = agent.get_value_functions((i,j))
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
        assert state in self.possible_actions
        return state


class Agent():
    def __init__(self, env):
        self.action_value_estimation = {}
        #self.policy = {state:random.sample(env.possible_actions[state], 1)[0] if len(env.possible_actions[state]) else None for state in env.possible_actions}
        self.policy = {
            (0,2): 'R', (1,2): 'R', (2,2): 'R', (3,2): None,
            (0,1): 'U',             (2,1): 'U', (3,1): None,
            (0,0): 'U', (1,0): 'R', (2,0): 'U', (3,0): 'L'
        }
        self.state_actions_counter = {}
        self.theta = np.random.rand(5)
        self.b = np.random.rand()

    def features_generator(self, state):
        x = []
        x.append(state[0] - 1.5)
        x.append(state[1] - 1.5)
        x.append((state[0]**2 - 4.5)/3)
        x.append((state[1]**2 - 4.5)/3)
        x.append((state[0]*state[1] - 4.5)/3)
        return np.array(x)

    def get_value_functions(self, state):
        x = self.features_generator(state)
        value_function = np.dot(self.theta, x.T) + self.b
        return value_function

    def backpropogate_value_function(self, state, expected_value):
        x = self.features_generator(state)
        v_f = self.get_value_functions(state)
        error = np.mean((expected_value - v_f)**2)
        self.theta = self.theta + LR * ((expected_value - v_f) * x)
        self.b = self.b + LR * (expected_value - v_f)

    def temporal_dofference_learning(self, env, N=30):
        for i in range(5000):
            state = (0, 0)
            n = 0
            p = 0.7
            states_and_rewards = []
            while env.possible_actions[state] and n<N:
                if np.random.rand() < p:
                    action = self.policy[state]
                else:
                    action = random.sample(env.possible_actions[state], 1)[0]
                new_state = env.next_state_after_action(state, action)
                reward = env.state_reward(new_state)
                if env.possible_actions[new_state]:
                    G = self.get_value_functions(state) + (0.5 + 0.5/(i+1))*(reward + GAMMA*self.get_value_functions(new_state) - self.get_value_functions(state))
                else:
                    G = reward
                #print(G)
                self.backpropogate_value_function(state, G)
                state = new_state


env = Environment()
agent = Agent(env)
env.draw_policy(agent.policy)
agent.temporal_dofference_learning(env)
env.draw_value_function(agent)
env.draw_policy(agent.policy)
