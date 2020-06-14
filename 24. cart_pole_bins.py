import gym
import numpy as np

ALPHA = 0.01
GAMMA = 0.9

class Model:
    def __init__(self):
        self.steps_for_each_observation = 10
        self.limits = [[-0.1, 0.1],
                       [-0.1, 0.1],
                       [-0.1, 0.1],
                       [-0.1, 0.1]]
        self.Q = np.zeros((self.steps_for_each_observation**len(self.limits), 2))

    def transform(self, observation):
        box = []
        for parameter, limit_i in zip(observation, range(len(self.limits))):
            if parameter < self.limits[limit_i][0]:
                self.limits[limit_i][0] = parameter - 0.01
            if parameter > self.limits[limit_i][1]:
                self.limits[limit_i][1] = parameter + 0.01

            box.append(int((parameter - self.limits[limit_i][0])/(self.limits[limit_i][1]-self.limits[limit_i][0])*self.steps_for_each_observation))
        encoded = self.encode(box)
        return encoded

    def encode(self, indexes):
        encoded = 0
        for i, index_v in enumerate(indexes):
            encoded += index_v*self.steps_for_each_observation**i
        return encoded

    def predict(self, state):
        return self.Q[state]

    def fit(self, state, reward, action, next_state):
        self.Q[state][action] += ALPHA*(reward + GAMMA*np.max(self.Q[next_state]) - self.Q[state][action])


def play_episode(model, env, p=0.6):
    observation = env.reset()
    done = False
    nb_steps = 0
    while not done and nb_steps < 200:
        nb_steps +=1
        state = model.transform(observation)
        action = np.argmax(model.predict(state))
        if np.random.rand() > p:
            action = 1 - action #choosing another action
        new_observation, reward, done, info = env.step(action)
        if done:
            reward = -100
        model.fit(state, reward, action, model.transform(new_observation))
        observation = new_observation
    return nb_steps


env = gym.make('CartPole-v0')
# print(dir(env.action_space))
# print(help(env.action_space))
model = Model()
lengths = []
from tqdm import tqdm
for i in tqdm(range(10000)):
    lengths.append(play_episode(model, env))

for row in model.Q:
    print(row)
import matplotlib.pyplot as plt
plt.plot(lengths)
plt.show()
