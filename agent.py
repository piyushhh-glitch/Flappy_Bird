from Flappy_Bird import dqn
import flappy_bird_gymnasium
import gymnasium as gym
import os 
import torch
import torch.nn as nn
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch.optim as optim

if torch.backends.mps.is_available:
    device="mps"
elif torch.cuda.is_available:
    device="cuda"
else:
    device="cpu"


class Agent:

    def __init__(self,param_set):
        self.param_set=param_set
        with open("parameters.yaml","r") as f:
            all_parameter_set=yaml.safe_load(f)
            params=all_parameter_set[param_set]

        self.alpha=params["alpha"]
        self.gamma=params["gamma"]

        self.epsilon_init=params["epsilon_init"]
        self.epsilon_min=params["epsilon_min"]
        self.epsilon_decay=params["epsilon_decay"]

        self.replay_memory_size=params["replay_memory_size"]
        self.mini_batch_size=params["mini_batch_size"]

        self.reward_threshold=params["reward_threshold"]
        self.network_sync_rate=params["network_sync_rate"]

        self.loss_fn=nn.MSELoss()
        self.optimizer=None


    
    def run(self,is_training=True, render=False):

        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states=env.observation_space.shape[0] #input dim

        num_actions=env.action_space.n #output dim


        policy_dqn=DQN(num_states,num_actions).to(device)

        if is_training:
            memory=ReplayMemory(self.replay_memory_size)
            epsilon=self.epsilon_init

            target_dqn=DQN(num_states,num_actions).to(device)

            # copy the weight and bias from policy => target 
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps=0

            self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)


        for episode in itertools.count():

            state, _ = env.reset()
            state=torch.tensor(state,dtype=torch.float,device=device)

            episode_rewards=0
            terminated=False

            while not terminated:
                if is_training and random.random()<epsilon:
                    action = env.action_space.sample() #explore
                else:
                    with torch.no_grad():
                        action=policy_dqn(state,unsqueeze(dim=0)).squeeze().argmax() #exploit

                # Processing:terminated -> done 
                next_state,reward, terminated, _, _ = env.step(action.item())

                reward=torch.tensor(reward,dtype=torch.float,device=device)
                next_state=torch.tensor(next_state,dtype=torch.float,device=device)

                if is_training:
                    memory.append((state,action,new_state,reward,terminated))
                    steps+=1


                
                state=next_state
                episode_rewards+=reward
            print(f"For episode -> {episode+1}, total reward is -> {episode_rewards} and epsilon =>{epsilon}")
            if is_training:
                #epsilon decay
                epsilon=max(epsilon*self.epsilon_decay,self.epsilon_min)

            if is_training and len(memory)>self.mini_batch_size:
                #get sample
                mini_batch=memory.sample(self.mini_batch_size)

                optimize(mini_batch,policy_dqn,target_dqn)

                #sync the network
                if steps>self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps=0



        # env.close() - manually stop
    
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        #get experience from mini_batch

        states,actions,next_states,rewards,terminations=zip(*mini_batch)

        states=torch.stack(states)
        actions=torch.stack(actions)
        next_states=torch.stack(next_states)
        rewards=torch.stack(rewards)
        terminations=torch.tensor(terminations).float().to(device)

        # calculate target q vallues -> if terminations => true ->zero
        with torch.no_grad():
            target_q=rewards+(1-terminations)*self.gamma*target_dqn(next_states).max(dim=1)[0]

        # current y_pred
        current_q=policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()

        # compute loss
        loss=self.loss_fn(current_q,target_q)

        #optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        