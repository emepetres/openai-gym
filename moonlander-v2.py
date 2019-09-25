import gym
import numpy as np
from agents.dqn_agent import DQNAgent

env = gym.make("LunarLander-v2")
env.seed(0)
agent = DQNAgent(env.action_space.n, env.observation_space.shape[0])
episodes = 400
steps = 3000
loss = []
for i_episode in range(episodes):
    obv = np.reshape(env.reset(), (1, 8))
    total_reward = 0
    done = False
    for t in range(steps):
        # env.render()
        # print(observation)
        action = agent.act(obv, total_reward, done)
        next_obv, reward, done, info = env.step(action)
        next_obv = np.reshape(next_obv, (1, 8))
        total_reward += reward
        agent.store_transition(obv, action, reward, next_obv, done)
        obv = next_obv
        agent.replay()
        if done:
            print(
                "{}/{}, reward: {} in {} timesteps".format(
                    i_episode, episodes, total_reward, t + 1
                )
            )
            break
    loss.append(total_reward)

    # Average score of last 100 episode
    if len(loss) >= 100:
        is_solved = np.mean(loss[-100:])
        if is_solved > -200:
            print("\n Task Completed! \n")
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))

for i_episode in range(5):
    obv = np.reshape(env.reset(), (1, 8))
    total_reward = 0
    done = False
    for t in range(steps):
        env.render()
        # print(observation)
        action = agent.act(obv, total_reward, done)
        next_obv, reward, done, info = env.step(action)
        next_obv = np.reshape(next_obv, (1, 8))
        total_reward += reward
        obv = next_obv
        if done:
            print(
                "{}/{}, reward: {} in {} timesteps".format(
                    i_episode, episodes, total_reward, t + 1
                )
            )
            break
env.close()
