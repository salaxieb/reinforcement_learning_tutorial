import numpy as np
import random
from time import sleep

SMALL_VALUE = 0.001
GAMMA = 0.9
ALPHA = 1

class Environment():
    def __init__(self):
        self.possible_actions = {
            (0,2): ['D', 'R'],   (1,2): ['L', 'R'],     (2,2): ['L', 'D', 'R'],     (3,1): None,
            (0,1): ['D', 'U'],                          (2,1): ['D', 'U', 'R'],     (3,2): None,
            (0,0): ['U', 'R'],   (1,0): ['L', 'R'],     (2,0): ['L', 'U', 'R'],     (3,0): ['U', 'L']
        }

    def draw_value_function(self, action_value_estimation):
        for j in range(3):
            j = 2 - j
            print('-'*58)
            s = ''
            for i in range(4):
                value = 0.0
                if (i,j) in action_value_estimation:
                    value = np.mean([action_value_estimation[(i,j)][action] for action in action_value_estimation[(i,j)]])
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

    def from_states_and_rewards_make_states_and_returns(self, states_and_rewards):
        G = 0
        states_and_returns = []
        for s, a, r in reversed(states_and_rewards):
            G = r + GAMMA * G
            states_and_returns.append((s, a, G))
        return reversed(states_and_returns)

    def get_action_value_estimation(self, state, action):
        if state not in self.action_value_estimation:
            self.state_actions_counter[state] = {}
            self.action_value_estimation[state] = {}
        if action not in self.action_value_estimation[state]:
            self.action_value_estimation[state][action] = 0
            self.state_actions_counter[state][action] = 1
        return self.action_value_estimation[state][action]

    def first_visit_monte_carlo_prediction(self, env, N=30):
        for _ in range(5000):
            state = (0,0)
            n = 0
            epsilon = 0.9
            states_and_rewards = []
            if np.random.rand() < epsilon:
                action_index = np.argmax([self.get_action_value_estimation(state, action) for action in env.possible_actions[state]])
                action = env.possible_actions[state][action_index]
            else:
                action = random.sample(env.possible_actions[state], 1)[0]

            while env.possible_actions[state] and n<N:
                n += 1
                next_state = env.next_state_after_action(state, action)
                if env.possible_actions[next_state]: #if state is not terminal
                    if np.random.rand() < epsilon:
                        next_action_index = np.argmax([self.get_action_value_estimation(next_state, action) for action in env.possible_actions[next_state]])
                        next_action = env.possible_actions[next_state][next_action_index]
                        #print(next_state, next_action)
                    else:
                        next_action = random.sample(env.possible_actions[next_state], 1)[0]
                else:
                    next_action = None #for terminal states
                reward = env.state_reward(next_state)
                self.action_value_estimation[state][action] += 0.1*(reward+GAMMA*self.get_action_value_estimation(next_state, next_action) - self.get_action_value_estimation(state, action))
                self.state_actions_counter[state][action] += 1
                state = next_state
                action = next_action


env = Environment()
agent = Agent(env)
env.draw_policy(agent.policy)
agent.first_visit_monte_carlo_prediction(env)
env.draw_value_function(agent.action_value_estimation)
env.draw_policy(agent.policy)
print(agent.state_actions_counter)
