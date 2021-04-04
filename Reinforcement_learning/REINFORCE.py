import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import wandb

# wandb sweep config
sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'avg score',
      'goal': 'maximize'
    },
    'parameters': {
        'epoch': {
            'values': [1000, 3000, 5000]
        },
        'learning_rate': {
            'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
        },
        'fc_layer_size':{
            'values':[128,256,512]
        },
        'gamma': {
            'values': [.8, .9, .98]
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        },
    }
}

#Hyperparameters
config_defaults = {
    'epoch': 2000,
    'learning_rate': 1e-3,
    'optimizer': 'adam',
    'fc_layer_size': 128,
    'gamma': 0.98,
}
wandb.init(config=config_defaults)
config = wandb.config

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, config.fc_layer_size)
        self.fc2 = nn.Linear(config.fc_layer_size, 2)

        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=config.learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + config.gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def train():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20
    
    
    for n_epi in range(config.epoch):
        s = env.reset()
        done = False
        # print("n_epi: ", n_epi)
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
            
        pi.train_net()
        # env.render()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0

        wandb.log({"episode": n_epi, "avg score": score / print_interval})
    env.close()

    torch.save(pi.state_dict(), "REINFORCE-model.pth")

def test_env():
    env = gym.make('CartPole-v1')
    pi = Policy()

    pi.load_state_dict(torch.load("REINFORCE-model.pth"))

    s = env.reset()
    done = False

    while not done:  # CartPole-v1 forced to terminates at 500 step.
        prob = pi(torch.from_numpy(s).float())
        m = Categorical(prob)
        a = m.sample()
        s_prime, r, done, info = env.step(a.item())
        s = s_prime
        env.render()

def main():
    sweep_id = wandb.sweep(sweep_config, project="Pytorch-sweeps-reinforce")
    wandb.agent(sweep_id, train)


if __name__ == '__main__':
    main()
    # test_env()