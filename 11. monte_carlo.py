import numpy as np
import random
from time import sleep

SMALL_VALUE = 0.001
GAMMA = 0.9

class Environment():
    def __init__(self):
        self.possible_actions = {
            (0,2): ['D', 'R'],   (1,2): ['L', 'R'],     (2,2): ['L', 'D', 'R'],     (3,1): None,
            (0,1): ['D', 'U'],                          (2,1): ['D', 'U', 'R'],     (3,2): None,
            (0,0): ['U', 'R'],   (1,0): ['L', 'R'],     (2,0): ['L', 'U', 'R'],     (3,0): ['U', 'L']
        }

    def draw_value_function(self, value_funcition):
        for j in range(3):
            j = 2 - j
            print('-'*58)
            s = ''
            for i in range(4):
                value = 0.0
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
        return -0.0

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
        self.value_function = {}
        #self.policy = {state:random.sample(env.possible_actions[state], 1)[0] if len(env.possible_actions[state]) else None for state in env.possible_actions}
        self.policy = {
            (0,2): 'R', (1,2): 'R', (2,2): 'R', (3,2): None,
            (0,1): 'U',             (2,1): 'R', (3,1): None,
            (0,0): 'U', (1,0): 'L', (2,0): 'U', (3,0): 'L'
        }
        self.states_counter = {}

    def play_game(self, env, N=30):
        state = (0, 0)
        n = 0
        p = 0.5
        states_and_rewards = []
        while env.possible_actions[state] and n<N:
            n += 1
            if np.random.rand() > p:
                action = self.policy[state]
            else:
                action = random.sample(env.possible_actions[state], 1)[0]
            new_state = env.next_state_after_action(state, action)
            reward = env.state_reward(new_state)
            states_and_rewards.append((state, reward))
            state = new_state
        return states_and_rewards

    def from_states_and_rewards_make_states_and_returns(self, states_and_rewards):
        G = 0
        states_and_returns = []
        for s, r in reversed(states_and_rewards):
            G = r + GAMMA * G
            states_and_returns.append((s, G))
        return reversed(states_and_returns)

    def get_value_functions(self, state):
        if state not in self.value_function:
            self.states_counter[state] = 1
            self.value_function[state] = 0
        return self.value_function[state]


    def first_visit_monte_carlo_prediction(self, states_and_returns):
        for _ in range(5000):
            states_and_rewards = self.play_game(env)
            states_and_returns = self.from_states_and_rewards_make_states_and_returns(states_and_rewards)
            #print(list(states_and_returns))
            already_seen = set()
            for state, return_ in states_and_returns:
                if state not in already_seen:
                    self.value_function[state] = self.get_value_functions(state) + 1/self.states_counter[state]*(return_-self.get_value_functions(state))
                    self.states_counter[state] += 1
                    already_seen.add(state)


env = Environment()
agent = Agent(env)
env.draw_policy(agent.policy)
agent.first_visit_monte_carlo_prediction(env)
env.draw_value_function(agent.value_function)
env.draw_policy(agent.policy)
