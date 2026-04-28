import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.utils.envs_tools import check, get_shape_from_obs_space, get_dim_from_act_space
from harl.utils.models_tools import init, get_init_method


class ComaQNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, cent_obs_space, action_spaces, num_agents, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super().__init__()
        args = args.copy()
        self.critic_hidden_x = args["critic_hidden_x"]
        args["hidden_sizes"] = [self.critic_hidden_x*size for size in args["hidden_sizes"]]
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(self.initialization_method)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.n_agents = num_agents
        self.action_dim = get_dim_from_act_space(action_spaces[0])
        base = MLPBase
        input_shape = list(cent_obs_shape)
        input_shape[0] += self.action_dim * self.n_agents
        self.state_dim = input_shape[0]

        self.base = base(args, input_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.q_out = init_(nn.Linear(self.hidden_sizes[-1], self.action_dim))

        self.to(device)

    def forward(self, cent_obs, action, rnn_states, masks):
        """Compute action from the given inputs.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        action = check(action).long()
        action = F.one_hot(action, num_classes=self.action_dim).squeeze(dim=2).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # action: (batch, n_agent, action_dim), one-hot masked out by agent
        B, T, A = action.shape
        action = action.view(B, 1, -1).repeat(1, T, 1)    # (B, T, T*A)
        agent_mask = 1 - torch.eye(T).to(**self.tpdv)
        agent_mask = agent_mask.view(-1, 1).repeat(1, A).view(T, -1).unsqueeze(0)   # (1, T, T*A)
        action = action * agent_mask

        cent_obs = cent_obs.view(B, 1, -1).repeat(1, T, 1)  # (B, T, O)
        inputs = torch.cat([cent_obs, action], dim=-1)
        critic_features = self.base(inputs)
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(
                critic_features.view(-1, *critic_features.shape[2:]),
                rnn_states.view(-1, *rnn_states.shape[2:]),
                masks.view(-1, *masks.shape[2:]),
            )
            critic_features = critic_features.view(B, T, *critic_features.shape[1:])
            if rnn_states.size(0) == B*T:
                rnn_states = rnn_states.view(B, T, *rnn_states.shape[1:])
        q_values = self.q_out(critic_features)  # (B, T, A)

        return q_values, rnn_states
