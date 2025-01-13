from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from fast_env import FastHIVPatient
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import random
import os.path
import matplotlib.pyplot as plt
#from tqdm import tqdm
from evaluate import evaluate_HIV, evaluate_HIV_population
from cpprb import ReplayBuffer, PrioritizedReplayBuffer

import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

env = TimeLimit(
    env=FastHIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
"""
class ReplayBuffer:
        def __init__(self, capacity, device):
            self.capacity = int(capacity) # capacity of the buffer
            self.data = []
            self.index = 0 # index of the next cell to be filled
            self.device = device
        def append(self, s, a, r, s_, d):
            if len(self.data) < self.capacity:
                self.data.append(None)
            self.data[self.index] = (s, a, r, s_, d)
            self.index = (self.index + 1) % self.capacity
        def sample(self, batch_size):
            batch = random.sample(self.data, batch_size)
            return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
        def __len__(self):
            return len(self.data)
"""


class DQN(nn.Module):
    
    def __init__(self, state_dim, n_action, nb_neurons, device):
        """
        Initializes the DQN model.

        Args:
            state_dim (int): Dimensionality of the state space (input features).
            n_action (int): Number of possible actions (output size).
            nb_neurons (int): Number of neurons in the hidden layers.
            device (torch.device): The device to which the model will be moved.
        """
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.SiLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.SiLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.SiLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.SiLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.SiLU(),
            nn.Linear(nb_neurons, n_action)
        )
        self.device = device
        self.to(device)
    def forward(self, x):
        """
        Forward pass for the DQN model.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        return self.model(x)

class DQN2(nn.Module):
    def __init__(self, state_dim, n_action, nb_neurons, device):
        super(DQN2, self).__init__()
        self.part1 = DQN(6, 4, 256, device)
        path = os.path.join(os.path.dirname(__file__), "./best_DQN_agent_solo")
        self.part1.load_state_dict(torch.load(path, weights_only=True,map_location=torch.device(device)))
        for param in self.part1.parameters():
            param.requires_grad = False

        self.part2 = nn.Sequential(
            nn.Linear(n_action, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
        )
        self.device = device
        self.to(device)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        """
        Forward pass for the DQN model.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        x = self.part1(x)
        return self.part2(x)
    
class ProjectAgent:
    
    def __init__(self, train=False, config=None, model=None):
        if not train:
            self.device = "cpu"
            self.model = None
        else:
            self.train_init(config, model)

    def train_init(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        self.buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        
        
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.lr = lr
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.loss_log = []
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.device = "cuda"
        #self.memory = ReplayBuffer(self.buffer_size,device)
        self.discounts = self.gamma ** 10
        
        self.memory = PrioritizedReplayBuffer(self.buffer_size,{'obs': {"shape": (1,6)},
                      'act': {"shape": 1},
                      'rew': {"shape": 1},
                      'next_obs': {"shape": (1,6)},
                      'done': {"shape": 1}},
                  Nstep={"size": 10,
                         "gamma": self.gamma,
                         "rew": "rew",
                         "next": "next_obs"})
    
    def get_loss_log(self):
        return self.loss_log

    def act(self, observation, use_random=False):
        Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
        return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model = DQN(6, 4, 512, self.device)
        path = os.path.join(os.path.dirname(__file__), "./best_DQN_agent")
        self.model.load_state_dict(torch.load(path, weights_only=True,map_location=torch.device('cpu')))
        
    def greedy_action(self,network, state):
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            network.eval()
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def gradient_step(self):

        self.target_model.train()
        self.model.train()

        if True:# len(self.memory) >= self.batch_size:
            
            #state, action, reward, next_state, terminated = self.memory.sample(self.batch_size)
            sample = self.memory.sample(self.batch_size)
            #sample["rew"] + (1-sample["done"]) * discounts *
            #Qt_state_action_max = self.target_model(next_state).max(1)[0].detach()
            
            next_obs = self.target_model(torch.from_numpy(sample["next_obs"]).to(self.device))
            Qt_state_action_max = torch.argmax(next_obs, dim=2).detach()
            
            #IDmax = self.target_model(next_state).max(1)[1].detach()
            #Q_state_action_max = self.model(next_state).gather(1, IDmax.to(torch.long).unsqueeze(1)).squeeze()
            #update = torch.addcmul(reward, 1-terminated, Q_state_action_max, value=self.gamma)
            
            #update = torch.addcmul(reward, 1-terminated, Qt_state_action_max, value=self.gamma)
            rew = torch.from_numpy(sample["rew"]).to(self.device)
            done = torch.from_numpy(sample["done"]).to(self.device)
            update = torch.addcmul(rew, 1-done, Qt_state_action_max, value=self.discounts)
            
            obs = torch.from_numpy(sample["obs"]).to(self.device)
            act = torch.from_numpy(sample["act"]).to(torch.long).to(self.device).unsqueeze(1)
            #Q_state_action = self.model(state).gather(1, action.to(torch.long).unsqueeze(1))
            Q_state_action = self.model(obs).gather(2, act)
            
            #loss = self.criterion(Q_state_action, update.unsqueeze(1))
         
            loss = self.criterion(Q_state_action.squeeze(1), update)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            self.loss_log.append(loss.to("cpu").detach())

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_score = 0
        for _ in range(2000*self.batch_size): 
            action = env.action_space.sample()
            next_state, reward, done, trunc, _ = env.step(action)
            #self.memory.append(state, action, reward, next_state, done)
            self.memory.add(obs=np.array(state), 
                            act=np.array(action), 
                            rew=np.array(reward), 
                            next_obs=np.array(next_state), 
                            done=np.array(done))
            if done or trunc:
                state, _ = env.reset()                
            else:
                state = next_state

        state, _ = env.reset()
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        print("Buffer full !")

        while episode < max_episode:
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)

            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.add(obs=np.array(state), 
                            act=np.array(action), 
                            rew=np.array(reward), 
                            next_obs=np.array(next_state), 
                            done=np.array(done))
            episode_cum_reward += reward

            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()

        
            target_state_dict = self.target_model.state_dict()
            model_state_dict = self.model.state_dict()
            tau = self.update_target_tau
            for key in model_state_dict:
                target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
            self.target_model.load_state_dict(target_state_dict)


            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      #", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      ", last loss ", '{:4.1f}'.format(self.loss_log[-1] if self.loss_log  else 0),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)

                if episode_cum_reward > best_score:
                    best_score = episode_cum_reward
                    self.save("./src/best_DQN_agent")
                if episode_cum_reward > 1e10 and self.optimizer.param_groups[0]['lr'] == 0.001:
                    #self.optimizer.param_groups[0]['lr'] /= 100
                    pass
                episode_cum_reward = 0
                

            else:
                state = next_state

        return episode_return

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


if __name__ == "__main__":

    state_dim = env.observation_space.shape[0]
    n_action = env.action_space.n 
    nb_neurons=512 #50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(state_dim, n_action, nb_neurons, device)
    
    dqn = DQN(state_dim, n_action, nb_neurons, device)
    
    
    config = {'nb_actions': env.action_space.n,
                'learning_rate': 0.001,
                'gamma': 0.98,
                'buffer_size': 1000000,
                'epsilon_min': 0.02,
                'epsilon_max': 1.,
                'epsilon_decay_period': 50000, 
                'epsilon_delay_decay': 800,
                'batch_size': 1024,
                'gradient_steps': 2,
                'update_target_freq': 600,
                'update_target_tau': 0.005,
                'criterion': torch.nn.SmoothL1Loss()}
    

    
    agent = ProjectAgent(train=True, config=config, model=dqn)
    scores = agent.train(env, 10000)
    agent.save("./src/DQN_agent0")

    plt.plot(movingaverage(scores,10))
    plt.savefig('learning.png')
    plt.show()
    plt.clf()
   

    plt.plot(agent.get_loss_log())
    plt.savefig('loss.png')
    plt.show()
    plt.clf()

    plt.plot(movingaverage(scores,20))
    plt.savefig('learning1.png')
    plt.show()
    plt.clf()
