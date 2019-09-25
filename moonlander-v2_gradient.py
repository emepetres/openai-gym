import gym
import numpy as np
from agents.gradient_agent import GradientAgent

import time

RENDER_ENV = False
EPISODES = 5000
RENDER_REWARD_MIN = 5000

env = gym.make("LunarLander-v2")
env.seed(0)
agent = GradientAgent(
    n_x=env.observation_space.shape[0],
    n_y=env.action_space.n,
    learning_rate=0.02,
    reward_decay=0.99,
    load_path=None,
    save_path=None,
)
steps = 3000
rewards = []
for i_episode in range(EPISODES):
    obv = env.reset()
    episode_reward = 0
    done = False
    tic = time.clock()
    while True:
        if RENDER_ENV or i_episode % 50 == 0:
            env.render()
        # print(observation)
        action = agent.act(obv, episode_reward, done)
        next_obv, reward, done, info = env.step(action)
        agent.store_transition(obv, action, reward)

        toc = time.clock()
        elapsed_sec = toc - tic
        if elapsed_sec > 120:
            done = True

        episode_rewards_sum = sum(agent.episode_rewards)
        if episode_rewards_sum < -250:
            done = True

        if done:
            episode_rewards_sum = sum(agent.episode_rewards)
            rewards.append(episode_rewards_sum)
            max_reward_so_far = np.amax(rewards)

            print("==========================================")
            print("Episode: ", i_episode)
            print("Seconds: ", elapsed_sec)
            print("Reward: ", episode_rewards_sum)
            print("Max reward so far: ", max_reward_so_far)

            # 5. Train neural network
            discounted_episode_rewards_norm = agent.learn()

            if max_reward_so_far > RENDER_REWARD_MIN:
                RENDER_ENV = True

            break

        obv = next_obv

env.close()
