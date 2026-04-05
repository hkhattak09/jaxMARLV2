import torch.nn as nn
import torch.nn.functional as F
import torch


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 constrain_out=False, norm_in=False, discrete_action=False):
        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            self.out_fn = F.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, X):
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        return self.out_fn(self.fc4(h3))


class AggregatingCritic(nn.Module):
    """
    Permutation-equivariant centralised critic for homogeneous multi-agent teams.

    Each agent's (obs_i, action_i) is passed through a shared encoder independently,
    the resulting embeddings are mean-aggregated across agents, then a small head MLP
    produces the Q-value.

    Why this works better than a flat MLP over the concatenated joint input:
      - The encoder sees 194-dim inputs (same scale as the old per-agent critic),
        not 4,656-dim inputs — the learning problem is tractable.
      - Mean aggregation is permutation equivariant: Q does not change if agents are
        reordered, which is the correct inductive bias for a homogeneous cooperative team.
      - Parameter count is independent of n_agents (shared encoder weights).

    Interface:
        forward(X) where X = torch.cat([obs_all, act_all], dim=1)
                   i.e. (batch, n_agents*obs_dim + n_agents*act_dim)
        — matches the existing maddpg.py call pattern with no changes needed there.
    """

    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=128, embed_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LeakyReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X):
        """
        Args:
            X: (batch, n_agents*obs_dim + n_agents*act_dim)
               Packed as [obs_all | act_all] — matches torch.cat((obs, acs), dim=1).
        Returns:
            Q: (batch, 1)
        """
        batch = X.shape[0]
        obs_all = X[:, :self.n_agents * self.obs_dim]
        act_all = X[:, self.n_agents * self.obs_dim:]

        obs = obs_all.view(batch, self.n_agents, self.obs_dim)   # (B, N, obs_dim)
        act = act_all.view(batch, self.n_agents, self.act_dim)   # (B, N, act_dim)
        x = torch.cat([obs, act], dim=-1)                        # (B, N, obs_dim+act_dim)

        embeds = self.encoder(x)    # (B, N, embed_dim)
        agg = embeds.mean(dim=1)    # (B, embed_dim)
        return self.head(agg)       # (B, 1)
