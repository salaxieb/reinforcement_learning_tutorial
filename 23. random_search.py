import gym
from gym import wrappers
import numpy as np

env = gym.make('CartPole-v0')

def run_episode(env, weights):
    done = False
    observation = env.reset()
    episode_steps = 0
    while not done:
        #env.render()
        if np.dot(observation, weights) > 0:
            action = 0
        else:
            action = 1
        observation, reward, done, info = env.step(action)
        episode_steps += 1
    return episode_steps

optimal_weights = None
best_average_play = -1
for i in range(100):
    weights = (np.random.rand(4) - 0.5)/1
    steps_till_death = []
    for i in range(10):
        steps_till_death.append(run_episode(env, weights))

    if np.mean(steps_till_death) > best_average_play:
        best_average_play = np.mean(steps_till_death)
        optimal_weights = weights

    print(best_average_play)

env = wrappers.Monitor(env, 'videos', force=True)
run_episode(env, optimal_weights)
