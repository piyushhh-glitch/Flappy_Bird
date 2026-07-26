import flappy_bird_gymnasium
import gymnasium as gym
import os 
import torch
import torch.nn
from dqn import DQN
from experience_replay import ReplayMemory


if torch.backends.mps.is_available:
    device="mps"
elif torch.cuda.is_available:
    device="cuda"
else:
    device="cpu"



def run(self,is_training=True, render=False):
    env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

    num_states=env.observation_space.shape[0] #input dim

    num_actions=env.action_space.n #output dim


    policy_dqn=DQN(num_states,num_actions).to(device)

    state, _ = env.reset()

    if is_training:
        memory=ReplayMemory(10000)

    while True:
        # Next action:
        # (feed the observation to your agent here)
        action = env.action_space.sample()

        if is_training:
                memory.append((state,action,new_state,reward,terminated))

        # Processing:terminated -> done 
        next_state,reward, terminated, _, info = env.step(action)
        
        # Checking if the player is still alive
        if terminated:
            break

    env.close()