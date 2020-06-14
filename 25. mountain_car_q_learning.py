from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import gym
import numpy as np
from gym import wrappers

ALPHA = 0.01
GAMMA = 0.9

class Model:
    def __init__(self, samples):
        self.limits = [[-1.2 , 0.6],
                       [-0.07, 0.07]]

        self.scaler = StandardScaler().fit(samples)
        self.features_extractor = RBFSampler().fit(self.scaler.transform(samples))
        self.Q = [SGDRegressor(), SGDRegressor(), SGDRegressor()]
        features = self.transform([0, 0])
        [sgdregressor.fit(features, [0]) for sgdregressor in self.Q]

    def transform(self, observation):
        features = self.features_extractor.transform(self.scaler.transform([observation]))
        return features

    def predict(self, state):
        t_state = self.transform(state)
        return [sgdregressor.predict(t_state) for sgdregressor in self.Q]

    def fit(self, state, action, G):
        t_state = self.transform(state)
        old_Q = self.Q[action].predict(t_state)
        self.Q[action].partial_fit(t_state, old_Q + ALPHA * (G - old_Q))

def play_episode(model, env):
    observation = env.reset()
    done = False
    nb_steps = 0
    p = 1#0.8
    while not done and nb_steps<300:
        nb_steps +=1
        action = np.argmax(model.predict(observation))
        # if np.random.rand() > p:
        #     action = np.random.choice([0, 1, 2])
        new_observation, reward, done, info = env.step(action)
        reward += 300 * (abs(new_observation[1]) - abs(observation[1]))
        if done:
            reward += 300
        G = reward + np.max(model.predict(new_observation))
        model.fit(observation, action, G)
        observation = new_observation
    return nb_steps

env = gym.make('MountainCar-v0').env

samples = [env.observation_space.sample() for i in range(10)]
model = Model(samples)

rewards = []
from tqdm import tqdm
for i in tqdm(range(3000)):
    r = np.mean([play_episode(model, env) for i in range(1)])
    rewards.append(r)
    print(r)

import matplotlib.pyplot as plt
plt.plot(rewards)
plt.show()

env = wrappers.Monitor(env, 'videos', force=True)
print(play_episode(model, env))
