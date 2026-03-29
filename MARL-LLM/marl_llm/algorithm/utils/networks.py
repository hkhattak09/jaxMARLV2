import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 constrain_out=False, norm_in=False, discrete_action=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__() 

        # self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)    
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        out = self.out_fn(self.fc4(h3))
        return out
    
class MLPNetworkRew(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu, constrain_out=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetworkRew, self).__init__() 

        # self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim) 
        self.hidden_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(1)])
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out:
            # initialize small to prevent saturation
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        x = self.nonlin(self.fc1(X))
        for block in self.hidden_blocks:
            x = block(x)
        out = self.out_fn(self.fc4(x))

        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, nonlin=F.leaky_relu):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlin = nonlin

    def forward(self, x):
        residual = x
        x = self.nonlin(self.fc1(x))
        x = self.fc2(x)
        x += residual
        x = self.nonlin(x)
        return x
    
class Discriminator(nn.Module):

    def __init__(self, state_dim, action_dim, gamma, hidden_dim, hidden_num):
        super(Discriminator, self).__init__()

        self.g = MLPUnit(
           input_dim=state_dim + action_dim,
           out_dim=1,
           hidden_dim=hidden_dim,
           hidden_num=hidden_num
        )
        self.h = MLPUnit(
           input_dim=state_dim,
           out_dim=1,
           hidden_dim=hidden_dim,
           hidden_num=hidden_num
        )

        self.gamma = gamma

    def f(self, states, actions, next_states, dones = 0):
        # rs = self.g(states)
        rs = self.g(torch.cat((states, actions), dim=1))
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, actions, log_pis, next_states, dones):
        # Discriminator's output is sigmoid(f - log_pi).
        # return self.f(states, next_states, dones) - log_pis
        return self.f(states, actions, next_states, dones) - log_pis

    def calculate_reward(self, states, actions, log_pis, next_states, dones):
        with torch.no_grad():
            # logits = self.forward(states, actions, log_pis, next_states, dones)
            logits = self.f(states, actions, next_states, dones)
            # logits = self.g(states)
            # logits = self.g(torch.cat((states, actions), dim=1))
            # return -F.logsigmoid(-logits)
            # return F.logsigmoid(logits) - F.logsigmoid(-logits)
            return logits
        
class MLPUnit(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, hidden_num, hidden_activation=nn.LeakyReLU(), output_activation=None):
        """
        Parameters:
            input_dim (int): 输入层的维度
            hidden_dim (int): 每个隐藏层的神经元数量
            out_dim (int): 输出层的维度
            hidden_activation (torch.nn.Module): 隐藏层的激活函数，默认是ReLU
            output_activation (torch.nn.Module or None): 输出层的激活函数，默认为无激活
        """
        super(MLPUnit, self).__init__()
        
        layers = []
        # input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if hidden_activation:
            layers.append(hidden_activation)
        # hidden layer
        for i in range(1, hidden_num):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if hidden_activation:
                layers.append(hidden_activation)
        # output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        if output_activation:
            layers.append(output_activation)
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)