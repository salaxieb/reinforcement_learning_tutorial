import gym

env = gym.make('CartPole-v0')

print(env.reset())

box = env.observation_space

actions = env.action_space
print('sample', actions.sample())
print('sample', actions.sample())
print('sample', actions.sample())
print('sample', actions.sample())
done = False
steps = 0
while not done:
    observation, reward, done, info = env.step(actions.sample())
    print(observation, reward, done, info)
    steps += 1
print(steps)
