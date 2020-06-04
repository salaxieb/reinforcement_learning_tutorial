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
                if (i,j) in agent.policy:
                    value = agent.get_value_functions((i,j), agent.policy[(i,j)])
                else:
                    value = 0
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
        self.theta = np.random.rand(len(self.features_generator((0,0), 'U')))

    def features_generator(self, state, action):
        x = []
        x.append(state[0] - 1.5)
        x.append(state[1] - 2.5)
        x.append((state[0]**2 - 4.5)/3)
        x.append((state[1]**2 - 4.5)/3)
        x.append((state[0]*state[1] - 4.5)/3)
        x.append(state[0] * ('U'==action))
        x.append(state[0] * ('D'==action))
        x.append(state[0] * ('L'==action))
        x.append(state[0] * ('R'==action))
        x.append(state[1] * ('U'==action))
        x.append(state[1] * ('D'==action))
        x.append(state[1] * ('L'==action))
        x.append(state[1] * ('R'==action))
        x.append(1)
        return np.array(x)

    def get_value_functions(self, state, action):
        x = self.features_generator(state, action)
        value_function = np.dot(self.theta, x.T)
        return value_function

    def backpropogate_value_function(self, state, action, expected_value):
        x = self.features_generator(state, action)
        v_f = self.get_value_functions(state, action)
        error = np.mean((expected_value - v_f)**2)
        self.theta = self.theta + LR * ((expected_value - v_f) * x)

    def temporal_dofference_learning(self, env, N=30):
        for i in range(15000):
            state = (0, 0)
            n = 0
            p = 0.7
            states_and_rewards = []
            action = self.policy[state]
            while env.possible_actions[state] and n<N:
                new_state = env.next_state_after_action(state, action)
                reward = env.state_reward(new_state)
                if env.possible_actions[new_state]:
                    if np.random.rand() < p:
                        new_action = self.policy[new_state]
                    else:
                        new_action = random.sample(env.possible_actions[new_state], 1)[0]
                    G = self.get_value_functions(state, action) + (0.5 + 0.5/(i+1))*(reward + GAMMA*self.get_value_functions(new_state, new_action) - self.get_value_functions(state, action))
                else:
                    G = reward
                    #print(G)
                self.backpropogate_value_function(state, action, G)
                state = new_state
                action = new_action
            else:
                self.backpropogate_value_function(state, None, 0)


env = Environment()
agent = Agent(env)
env.draw_policy(agent.policy)
agent.temporal_dofference_learning(env)
env.draw_value_function(agent)
env.draw_policy(agent.policy)
