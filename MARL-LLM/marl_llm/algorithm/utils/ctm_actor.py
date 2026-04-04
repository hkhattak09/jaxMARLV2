import torch
import torch.nn as nn

from models.ctm_rl import ContinuousThoughtMachineRL


class CTMActor(nn.Module):
    """
    CTM-based actor for MADDPG. Wraps ContinuousThoughtMachineRL with an action head.

    Hidden states (state_trace, activated_state_trace) are managed externally by the
    training loop and passed in/out of forward(). This allows full temporal memory
    during rollout while keeping the replay buffer unchanged (stateless updates).
    """

    def __init__(self, obs_dim, action_dim, d_model=256, memory_length=16,
                 n_synch_out=16, iterations=4, synapse_depth=1,
                 deep_nlms=False, do_layernorm_nlm=True, memory_hidden_dims=64):
        super().__init__()
        self.d_model = d_model
        self.memory_length = memory_length

        self.ctm = ContinuousThoughtMachineRL(
            iterations=iterations,
            d_model=d_model,
            d_input=obs_dim,
            n_synch_out=n_synch_out,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=[memory_hidden_dims] if isinstance(memory_hidden_dims, int) else memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            backbone_type='classic-control-backbone',
        )

        synch_size = n_synch_out * (n_synch_out + 1) // 2
        self.action_head = nn.Linear(synch_size, action_dim)

    def get_initial_hidden_state(self, batch_size, device):
        """
        Returns a fresh hidden state tuple for batch_size agents.

        state_trace is zeroed. activated_state_trace is expanded from the learned
        start_activated_trace parameter (gradients flow through this during updates).

        Args:
            batch_size: number of agents (n_rollout_threads * n_agents during rollout,
                        or training batch_size during stateless actor update)
            device: torch.device
        Returns:
            (state_trace, activated_state_trace): each (batch_size, d_model, memory_length)
        """
        state_trace = torch.zeros(batch_size, self.d_model, self.memory_length, device=device)
        activated_state_trace = (
            self.ctm.start_activated_trace
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        return (state_trace, activated_state_trace)

    def forward(self, obs, hidden_states):
        """
        Args:
            obs: (batch_size, obs_dim)
            hidden_states: (state_trace, activated_state_trace),
                           each (batch_size, d_model, memory_length)
        Returns:
            actions: (batch_size, action_dim) in [-1, 1]
            new_hidden_states: updated (state_trace, activated_state_trace) tuple
        """
        synch_out, new_hidden_states = self.ctm(obs, hidden_states)
        actions = torch.tanh(self.action_head(synch_out))
        return actions, new_hidden_states
