import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method


class QMixer(nn.Module):
    def __init__(
            self,
            args,
            cent_obs_space,
            num_agents,
            device=torch.device("cpu"),
        ):
        super().__init__()
        args = args.copy()
        self.critic_hidden_x = args["critic_hidden_x"]
        args["hidden_sizes"] = [self.critic_hidden_x*size for size in args["hidden_sizes"]]
        self.hidden_sizes = args["hidden_sizes"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.state_dim = cent_obs_shape[0]
        self.embed_dim = self.hidden_sizes[0]
        self.n_agents = num_agents

        if args["valuedecomp"].get("hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif args["valuedecomp"].get("hypernet_layers", 1) == 2:
            hypernet_embed = self.hidden_sizes[0]
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        elif args["valuedecomp"].get("hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # Initialise the hyper-network of the skip-connections, such that the result is close to VDN
        self.use_skip_connections = args["valuedecomp"].get("skip_connections", False)
        if self.use_skip_connections:
            self.skip_connections = nn.Linear(self.state_dim, self.n_agents, bias=True)
            self.skip_connections.bias.data.fill_(1.0)  # bias produces initial VDN weights

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

        self.to(device)

    def forward(self, agent_qs, states, **kwargs):
        states = check(states).to(**self.tpdv)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Skip connections
        s = 0
        if self.use_skip_connections:
            ws = torch.abs(self.skip_connections(states)).view(-1, self.n_agents, 1)
            s = torch.bmm(agent_qs, ws)    # non-negative linear combination of agent utilities
        # Compute final output
        y = torch.bmm(hidden, w_final) + v + s
        # Reshape and return
        q_tot = y.view(-1, 1)
        return q_tot