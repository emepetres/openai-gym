import gym
from agents.random_agent import RandomAgent

env = gym.make("CartPole-v0")
agent = RandomAgent(env.action_space)
for i_episode in range(20):
    observation = env.reset()
    reward = 0
    done = False
    for t in range(100):
        env.render()
        print(observation)
        action = agent.act(object, reward, done)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
