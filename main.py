import time
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gym
import robosuite as suite
from robosuite.wrappers import GymWrapper

from networks import CriticNetwork, ActorNetwork
from buffer import ReplayBuffer
from td3_torch import Agent

if __name__ == "__main__":

    if not os.path.exists('tmp/td3'):
        os.makedirs('tmp/td3')

    env_name = 'Door'

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    ## Test of the classes
    # Networks Class:
    # critic_network = CriticNetwork([8], 8)
    # actor_network = ActorNetwork([8], 8)
    #
    # replay_buffer = ReplayBuffer(8, [8], 8)

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size=128
    layer1_size=256
    layer2_size=128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,tau=0.005,
                  input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0], layer1_size=layer1_size,
                  layer2_size=layer2_size, batch_size=batch_size)

    writer = SummaryWriter('logs')
    n_games = 10

    for experiment in range(10):
        best_score = 0
        episode_identifier = (f'{experiment} - actor_learning_rate {actor_learning_rate} - critic_learning_rate {critic_learning_rate} - '
                              f'layer_1_size={layer1_size} - layer_2_size={layer2_size} - ')

        #agent.load_models()



        for i in range(n_games):
            observation = env.reset()
            done = False
            score = 0

            while not done:
                action = agent.choose_action(observation)
                next_observation, reward, done, info = env.step(action)
                score += reward
                agent.remember(observation, action, reward, next_observation, done)
                agent.learn()
                observation = next_observation
                time.sleep(0.01)

            writer.add_scalar(f'Score - {episode_identifier}', score, global_step=i)

            if not (i % 10):
                agent.save_models()

            print(f'Episode: {i} score: {score}')

            actor_learning_rate = actor_learning_rate * 0.85



