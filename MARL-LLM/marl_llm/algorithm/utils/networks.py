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
    Permutation-equivariant centralised critic with optional LSTM for temporal reasoning.

    Architecture:
        encoder: (obs_i, act_i) → hidden_dim → embed_dim   [shared, per-agent, independent]
        mean aggregate → embed_dim                           [permutation equivariant]
        LSTM: embed_dim → lstm_hidden_dim                    [temporal reasoning, optional]
        head: lstm_hidden_dim (or embed_dim) → hidden_dim → 1

    The LSTM processes the aggregated team summary over time, enabling the critic
    to track how team state evolves across episode steps. This follows R-MADDPG's
    finding that a recurrent critic is critical for partial observability.

    Permutation equivariance is preserved because aggregation happens before recurrence.

    Interface:
        forward(X, hidden=None) → (Q, new_hidden)
        where X = torch.cat([obs_all, act_all], dim=1)
              hidden = (h_0, c_0) each (1, batch, lstm_hidden_dim), or None
    """

    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=128, embed_dim=64,
                 lstm_hidden_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LeakyReLU(),
        )

        self.use_lstm = lstm_hidden_dim > 0
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden_dim,
                                num_layers=1, batch_first=True)

        head_input_dim = lstm_hidden_dim if self.use_lstm else embed_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def get_initial_hidden(self, batch_size, device):
        """Return zero-initialized LSTM hidden state, or None if no LSTM.

        Returns:
            (h_0, c_0): each (1, batch_size, lstm_hidden_dim), or None
        """
        if not self.use_lstm:
            return None
        h_0 = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        c_0 = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        return (h_0, c_0)

    def forward(self, X, hidden=None):
        """
        Args:
            X: (batch, n_agents*obs_dim + n_agents*act_dim)
               Packed as [obs_all | act_all] — matches torch.cat((obs, acs), dim=1).
            hidden: (h_0, c_0) each (1, batch, lstm_hidden_dim), or None.
        Returns:
            Q: (batch, 1)
            new_hidden: (h_n, c_n) each (1, batch, lstm_hidden_dim), or None
        """
        batch = X.shape[0]
        obs_all = X[:, :self.n_agents * self.obs_dim]
        act_all = X[:, self.n_agents * self.obs_dim:]

        obs = obs_all.view(batch, self.n_agents, self.obs_dim)   # (B, N, obs_dim)
        act = act_all.view(batch, self.n_agents, self.act_dim)   # (B, N, act_dim)
        x = torch.cat([obs, act], dim=-1)                        # (B, N, obs_dim+act_dim)

        embeds = self.encoder(x)    # (B, N, embed_dim)
        agg = embeds.mean(dim=1)    # (B, embed_dim)

        if self.use_lstm:
            if hidden is None:
                hidden = self.get_initial_hidden(batch, X.device)
            lstm_out, new_hidden = self.lstm(agg.unsqueeze(1), hidden)  # (B, 1, lstm_hidden)
            lstm_out = lstm_out.squeeze(1)                               # (B, lstm_hidden)
            return self.head(lstm_out), new_hidden
        else:
            return self.head(agg), None
