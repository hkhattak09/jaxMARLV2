import torch
from harl.models.base.act import ACTLayer


class ACTLayerTrans(ACTLayer):

    def forward(self, x, available_actions=None, deterministic=False):
        """Compute actions and action logprobs from given input.
        Args:
            x: (torch.Tensor) input to network.
            available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """

        if self.multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_distribution = action_out(x, available_actions)
                action = (
                    action_distribution.mode()
                    if deterministic
                    else action_distribution.sample()
                )
                action_log_prob = action_distribution.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(
                dim=-1, keepdim=True
            )
        else:
            action_distribution = self.action_out(x, available_actions)
            actions = (
                action_distribution.mode()
                if deterministic
                else action_distribution.sample()
            )
            action_log_probs = action_distribution.log_probs(actions)

        return actions, action_log_probs, action_distribution.probs
