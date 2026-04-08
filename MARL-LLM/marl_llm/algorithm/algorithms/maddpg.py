import torch
from algorithm.utils import soft_update, average_gradients
from algorithm.utils import DDPGAgent

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types, epsilon, noise, gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3,
                 hidden_dim=64, lstm_hidden_dim=64, n_agents=None, device='cpu', discrete_action=False,
                 use_ctm_actor=False, ctm_config=None, prior_mode='none'):
        """
        Initialize the MADDPG object with given parameters.

        Args:
            prior_mode: How to use the Reynolds prior during training.
                'none'       — prior is ignored entirely.
                'regularize' — prior is used as an MSE regularization term on actor output.
                'seed'       — prior seeds the CTM state_trace at the start of each
                               actor-update forward pass (CTM only; error if MLP).
        """
        if prior_mode in ('seed', 'seed+reg') and not use_ctm_actor:
            raise ValueError(f"prior_mode='{prior_mode}' requires use_ctm_actor=True — MLP has no hidden state to seed.")
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.epsilon = epsilon
        self.noise = noise
        self.use_ctm_actor = use_ctm_actor
        self.prior_mode = prior_mode
        # n_agents: number of physical agents (≥ nagents when parameter sharing is used)
        self.n_agents = n_agents if n_agents is not None else self.nagents

        if use_ctm_actor:
            from algorithm.utils.ctm_agent import CTMDDPGAgent
            self.agents = [CTMDDPGAgent(lr_actor=lr_actor,
                                        lr_critic=lr_critic,
                                        hidden_dim=hidden_dim,
                                        lstm_hidden_dim=lstm_hidden_dim,
                                        epsilon=self.epsilon,
                                        noise=self.noise,
                                        ctm_config=ctm_config,
                                        **params) for params in agent_init_params]
        else:
            self.agents = [DDPGAgent(lr_actor=lr_actor,
                                     lr_critic=lr_critic,
                                     discrete_action=discrete_action,
                                     hidden_dim=hidden_dim,
                                     lstm_hidden_dim=lstm_hidden_dim,
                                     epsilon=self.epsilon,
                                     noise=self.noise,
                                     **params) for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'
        self.critic_dev = 'cpu'
        self.trgt_pol_dev = 'cpu'
        self.trgt_critic_dev = 'cpu'

        self.niter = 0

        if device != 'cpu':
            self.prep_training(device=device)


    @property           
    def policies(self):
        """
        Get the policies of all agents.
        """
        return [a.policy for a in self.agents]

    def target_policies(self, agent_i, next_obs_all):
        """
        Compute target actions for ALL agents from joint next observations.

        Args:
            next_obs_all: (batch, n_agents * obs_dim) — joint next observations
        Returns:
            (batch, n_agents * action_dim) — all agents' target actions concatenated
        """
        batch = next_obs_all.shape[0]
        obs_dim = next_obs_all.shape[1] // self.n_agents
        # (batch, n_agents*obs_dim) → (batch*n_agents, obs_dim)
        obs_flat = next_obs_all.reshape(batch * self.n_agents, obs_dim)

        if self.use_ctm_actor:
            hidden = self.agents[agent_i].target_policy.get_initial_hidden_state(
                batch * self.n_agents, next_obs_all.device)
            actions, _ = self.agents[agent_i].target_policy(obs_flat, hidden)
        else:
            actions = self.agents[agent_i].target_policy(obs_flat)

        # TD3 target policy smoothing: add clipped noise to prevent exploiting Q-function peaks
        noise = torch.clamp(torch.randn_like(actions) * 0.2, -0.5, 0.5)
        actions = torch.clamp(actions + noise, -1.0, 1.0)

        # (batch*n_agents, action_dim) → (batch, n_agents*action_dim)
        return actions.reshape(batch, self.n_agents * actions.shape[1])

    def scale_noise(self, scale, new_epsilon):
        """
        Scale noise for each agent.
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)       
            a.epsilon = new_epsilon

    def reset_noise(self):
        """
        Reset noise for each agent.
        """
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, start_stop_num, explore=False, hidden_states=None):
        """
        Take a step forward in environment with all agents.
        Inputs:
            observations: observations tensor, shape (obs_dim, n_total_agents)
            start_stop_num: list of slices for agent index ranges
            explore (boolean): Whether or not to add exploration noise
            hidden_states: (state_trace, activated_state_trace) for CTM actor; None for MLP
        Outputs:
            actions: list of action tensors, each (action_dim, n_agents)
            log_pis: list of log-prob tensors (None entries for CTM)
            new_hidden_states: updated hidden state tuple (CTM) or None (MLP)
        """
        actions = []
        log_pis = []
        new_hidden_states = None
        for i in range(len(start_stop_num)):
            obs_slice = observations[:, start_stop_num[i]].t()
            if self.use_ctm_actor:
                action, log_pi, new_hidden_states = self.agents[i].step(obs_slice, hidden_states, explore=explore)
            else:
                action, log_pi = self.agents[i].step(obs_slice, explore=explore)
            actions.append(action)
            log_pis.append(log_pi)
        return actions, log_pis, new_hidden_states
    

    def update(self, obs, acs, rews, next_obs, dones, agent_i, acs_prior=None, alpha=0.5, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer.
        
        Performs actor-critic update with optional regularization towards prior actions.
        
        Args:
            obs: Current observations
            acs: Actions taken
            rews: Rewards received
            next_obs: Next observations
            dones: Episode termination flags
            agent_i (int): Index of agent to update
            acs_prior: Prior actions for regularization (optional)
            alpha (float): Regularization coefficient
            parallel (bool): Whether to average gradients across threads
            logger: Tensorboard logger for metrics tracking (optional)
        """
        curr_agent = self.agents[agent_i]    

        ######################### Update Critic (TD3 double critic) #########################
        curr_agent.critic_optimizer.zero_grad()
        curr_agent.critic2_optimizer.zero_grad()

        # Compute target actions using target policies
        all_trgt_acs = self.target_policies(agent_i, next_obs)
        trgt_vf_in = torch.cat((next_obs, all_trgt_acs), dim=1)

        # TD3: target value uses min of both target critics — prevents Q-overestimation
        with torch.no_grad():
            target_Q1, _ = curr_agent.target_critic(trgt_vf_in)
            target_Q2, _ = curr_agent.target_critic2(trgt_vf_in)
            target_value = rews + self.gamma * torch.min(
                target_Q1, target_Q2,
            ) * (1 - dones)

        # Compute current Q-values from both critics
        vf_in = torch.cat((obs, acs), dim=1)
        actual_value1, _ = curr_agent.critic(vf_in)
        actual_value2, _ = curr_agent.critic2(vf_in)

        # Critic losses (TD error) — both trained against the same min target
        vf_loss1 = MSELoss(actual_value1, target_value)
        vf_loss2 = MSELoss(actual_value2, target_value)

        # Backward pass for critic 1
        vf_loss1.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        # Backward pass for critic 2
        vf_loss2.backward()
        if parallel:
            average_gradients(curr_agent.critic2)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic2.parameters(), 0.5)
        curr_agent.critic2_optimizer.step()

        vf_loss = vf_loss1  # used for logging below

        ######################### Update Actor (Option A, delayed) #########################
        # TD3 delayed policy updates: only update actor every 2 critic steps.
        # Lets the critic stabilise before the actor chases it.
        pol_loss = torch.tensor(0.0)
        regularization_term = torch.tensor(0.0)

        if self.niter % 2 == 0:
            # Recompute only agent_i's actions through the shared policy; use stored
            # buffer actions for all other slots. This is 24× cheaper than Option B
            # while still being correct: gradient flows only through agent_i's slot,
            # which is sufficient since all 24 agents share the same policy weights.
            curr_agent.policy_optimizer.zero_grad()

            batch_size = obs.shape[0]
            obs_dim = obs.shape[1] // self.n_agents
            action_dim = acs.shape[1] // self.n_agents

            # Extract agent_i's obs slice: (batch, obs_dim)
            agent_obs = obs[:, agent_i * obs_dim : (agent_i + 1) * obs_dim]

            if self.use_ctm_actor:
                if self.prior_mode == 'seed' and acs_prior is not None:
                    # Seed mode: initialise state_trace from (obs, prior) via learned seed_mlp.
                    # acs_prior is joint (batch, n_agents*action_dim); extract agent_i's slice.
                    prior_agent_i = acs_prior[:, agent_i * action_dim : (agent_i + 1) * action_dim]
                    hidden = curr_agent.policy.get_prior_seeded_hidden_state(agent_obs, prior_agent_i)
                else:
                    hidden = curr_agent.policy.get_initial_hidden_state(batch_size, obs.device)
                curr_pol_out, _ = curr_agent.policy(agent_obs, hidden)
            else:
                curr_pol_out = curr_agent.policy(agent_obs)  # (batch, action_dim)

            # Substitute agent_i's slot; use stored actions for all other agents
            all_pol_acs = torch.cat([
                acs[:, :agent_i * action_dim].detach(),
                curr_pol_out,
                acs[:, (agent_i + 1) * action_dim:].detach(),
            ], dim=1)  # (batch, n_agents * action_dim)

            # Actor loss: maximize Q-value (negative for gradient ascent)
            vf_in = torch.cat((obs, all_pol_acs), dim=1)
            Q_for_pol, _ = curr_agent.critic(vf_in)
            pol_loss = -Q_for_pol.mean()

            # Regularization towards prior actions (only for prior_mode='regularize')
            regularization_term = torch.tensor(0.0, requires_grad=True)
            if self.prior_mode == 'regularize' and acs_prior is not None:
                mse_loss = torch.nn.MSELoss()

                # Filter out near-zero prior actions (likely invalid/padding)
                mask = (acs_prior.abs() < 1e-2).all(dim=1)
                valid_mask = ~mask
                filtered_all_pol_acs = all_pol_acs[valid_mask]
                filtered_acs_prior = acs_prior[valid_mask]

                # Compute regularization loss only for valid prior actions
                if filtered_all_pol_acs.numel() > 0 and filtered_acs_prior.numel() > 0:
                    regularization_term = mse_loss(filtered_all_pol_acs, filtered_acs_prior)

                pol_loss = pol_loss + 0.3 * alpha * regularization_term

            # Backward pass for actor
            pol_loss.backward()
            if parallel:
                average_gradients(curr_agent.policy)
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()

        # Log metrics if logger is provided
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i, {
                'vf_loss': vf_loss,
                'pol_loss': pol_loss,
                'regularization_loss': 0.5 * alpha * regularization_term
            }, self.niter)

        return vf_loss.item(), pol_loss.item(), (0.3 * alpha * regularization_term).item()

    def update_sequence(self, obs_seq, acs_seq, rews_seq, next_obs_seq, dones_seq,
                        agent_i, prior_seq=None, alpha=0.5, reg_weight=None,
                        burn_in_length=16, logger=None):
        """
        Sequence-based update with R2D2-style burn-in for recurrent critic.

        Processes contiguous episode chunks: burn-in prefix reconstructs LSTM hidden
        states without gradient, then training suffix computes losses with gradient.

        Args:
            obs_seq:      (seq_len, num_seq, n_agents*obs_dim)
            acs_seq:      (seq_len, num_seq, n_agents*act_dim)
            rews_seq:     (seq_len, num_seq, 1)
            next_obs_seq: (seq_len, num_seq, n_agents*obs_dim)
            dones_seq:    (seq_len, num_seq, 1)
            agent_i:      index of agent to update
            prior_seq:    (seq_len, num_seq, n_agents*act_dim) or None
            alpha:        regularization coefficient for prior
            burn_in_length: number of prefix timesteps for hidden state reconstruction
            logger:       tensorboard logger (optional)

        Returns:
            (critic_loss, actor_loss, reg_loss) — floats
        """
        curr_agent = self.agents[agent_i]
        seq_len, num_seq = obs_seq.shape[:2]
        train_len = seq_len - burn_in_length
        device = obs_seq.device

        obs_dim = obs_seq.shape[2] // self.n_agents
        action_dim = acs_seq.shape[2] // self.n_agents

        # ── Split into burn-in and training portions ──
        obs_burn, obs_train = obs_seq[:burn_in_length], obs_seq[burn_in_length:]
        acs_burn, acs_train = acs_seq[:burn_in_length], acs_seq[burn_in_length:]
        rews_train = rews_seq[burn_in_length:]
        next_obs_burn = next_obs_seq[:burn_in_length]
        next_obs_train = next_obs_seq[burn_in_length:]
        dones_train = dones_seq[burn_in_length:]
        prior_train = prior_seq[burn_in_length:] if prior_seq is not None else None

        # ════════════════════════════════════════════════════════════════
        #  CRITIC UPDATE
        # ════════════════════════════════════════════════════════════════
        curr_agent.critic_optimizer.zero_grad()
        curr_agent.critic2_optimizer.zero_grad()

        # ── Burn-in: reconstruct LSTM hidden states without gradient ──
        critic_h = curr_agent.critic.get_initial_hidden(num_seq, device)
        critic2_h = curr_agent.critic2.get_initial_hidden(num_seq, device)
        tgt_critic_h = curr_agent.target_critic.get_initial_hidden(num_seq, device)
        tgt_critic2_h = curr_agent.target_critic2.get_initial_hidden(num_seq, device)

        with torch.no_grad():
            for t in range(burn_in_length):
                vf_in = torch.cat((obs_burn[t], acs_burn[t]), dim=1)
                _, critic_h = curr_agent.critic(vf_in, critic_h)
                _, critic2_h = curr_agent.critic2(vf_in, critic2_h)

                # Target critics burn-in on (next_obs, target_actions)
                trgt_acs = self.target_policies(agent_i, next_obs_burn[t])
                trgt_vf_in = torch.cat((next_obs_burn[t], trgt_acs), dim=1)
                _, tgt_critic_h = curr_agent.target_critic(trgt_vf_in, tgt_critic_h)
                _, tgt_critic2_h = curr_agent.target_critic2(trgt_vf_in, tgt_critic2_h)

        # Detach hidden states at the burn-in / training boundary (None when no LSTM)
        if critic_h is not None:
            critic_h = tuple(h.detach() for h in critic_h)
        if critic2_h is not None:
            critic2_h = tuple(h.detach() for h in critic2_h)
        if tgt_critic_h is not None:
            tgt_critic_h = tuple(h.detach() for h in tgt_critic_h)
        if tgt_critic2_h is not None:
            tgt_critic2_h = tuple(h.detach() for h in tgt_critic2_h)

        # ── Training: forward with gradient, accumulate TD losses ──
        total_vf_loss1 = torch.tensor(0.0, device=device)
        total_vf_loss2 = torch.tensor(0.0, device=device)

        for t in range(train_len):
            vf_in = torch.cat((obs_train[t], acs_train[t]), dim=1)
            Q1, critic_h = curr_agent.critic(vf_in, critic_h)
            Q2, critic2_h = curr_agent.critic2(vf_in, critic2_h)

            with torch.no_grad():
                trgt_acs = self.target_policies(agent_i, next_obs_train[t])
                trgt_vf_in = torch.cat((next_obs_train[t], trgt_acs), dim=1)
                tgt_Q1, tgt_critic_h = curr_agent.target_critic(trgt_vf_in, tgt_critic_h)
                tgt_Q2, tgt_critic2_h = curr_agent.target_critic2(trgt_vf_in, tgt_critic2_h)
                target_value = rews_train[t] + self.gamma * torch.min(tgt_Q1, tgt_Q2) * (1 - dones_train[t])

            total_vf_loss1 = total_vf_loss1 + MSELoss(Q1, target_value)
            total_vf_loss2 = total_vf_loss2 + MSELoss(Q2, target_value)

        # Average over training timesteps
        total_vf_loss1 = total_vf_loss1 / train_len
        total_vf_loss2 = total_vf_loss2 / train_len

        total_vf_loss1.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        total_vf_loss2.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.critic2.parameters(), 0.5)
        curr_agent.critic2_optimizer.step()

        # ════════════════════════════════════════════════════════════════
        #  ACTOR UPDATE (delayed — every 2 critic steps)
        # ════════════════════════════════════════════════════════════════
        pol_loss_val = 0.0
        reg_loss_val = 0.0

        if self.niter % 2 == 0:
            curr_agent.policy_optimizer.zero_grad()

            # Burn-in: reconstruct critic hidden state for actor's Q evaluation
            actor_critic_h = curr_agent.critic.get_initial_hidden(num_seq, device)
            with torch.no_grad():
                for t in range(burn_in_length):
                    vf_in = torch.cat((obs_burn[t], acs_burn[t]), dim=1)
                    _, actor_critic_h = curr_agent.critic(vf_in, actor_critic_h)
            if actor_critic_h is not None:
                actor_critic_h = tuple(h.detach() for h in actor_critic_h)

            # Burn-in for CTM actor hidden state (if using CTM)
            if self.use_ctm_actor:
                actor_h = curr_agent.policy.get_initial_hidden_state(num_seq, device)
                with torch.no_grad():
                    for t in range(burn_in_length):
                        agent_obs_t = obs_burn[t][:, agent_i * obs_dim : (agent_i + 1) * obs_dim]
                        _, actor_h = curr_agent.policy(agent_obs_t, actor_h)
                if actor_h is not None:
                    actor_h = tuple(h.detach() for h in actor_h)

            # Training: recompute agent_i's actions, evaluate Q
            total_pol_loss = torch.tensor(0.0, device=device)
            total_reg = torch.tensor(0.0, device=device)

            for t in range(train_len):
                agent_obs_t = obs_train[t][:, agent_i * obs_dim : (agent_i + 1) * obs_dim]

                if self.use_ctm_actor:
                    if self.prior_mode in ('seed', 'seed+reg') and prior_train is not None:
                        prior_agent_t = prior_train[t][:, agent_i * action_dim : (agent_i + 1) * action_dim]
                        actor_h = curr_agent.policy.get_prior_seeded_hidden_state(agent_obs_t, prior_agent_t)
                    curr_pol_out, actor_h = curr_agent.policy(agent_obs_t, actor_h)
                else:
                    curr_pol_out = curr_agent.policy(agent_obs_t)

                # Substitute agent_i's slot; stored actions for all others
                all_pol_acs = torch.cat([
                    acs_train[t][:, :agent_i * action_dim].detach(),
                    curr_pol_out,
                    acs_train[t][:, (agent_i + 1) * action_dim:].detach(),
                ], dim=1)

                vf_in = torch.cat((obs_train[t], all_pol_acs), dim=1)
                Q, actor_critic_h = curr_agent.critic(vf_in, actor_critic_h)
                total_pol_loss = total_pol_loss + (-Q.mean())

                # Regularization towards prior:
                #   'regularize' — always on with fixed weight 0.3*alpha
                #   'seed+reg'  — uses decaying reg_weight (cosine schedule from training loop)
                do_reg = (
                    (self.prior_mode == 'regularize') or
                    (self.prior_mode == 'seed+reg' and reg_weight is not None and reg_weight > 1e-8)
                )
                if do_reg and prior_train is not None:
                    prior_t = prior_train[t]
                    mask = (prior_t.abs() < 1e-2).all(dim=1)
                    valid = ~mask
                    if valid.any():
                        total_reg = total_reg + MSELoss(all_pol_acs[valid], prior_t[valid])

            total_pol_loss = total_pol_loss / train_len

            # Apply regularization with appropriate weight
            do_reg = (
                (self.prior_mode == 'regularize') or
                (self.prior_mode == 'seed+reg' and reg_weight is not None and reg_weight > 1e-8)
            )
            if do_reg and prior_train is not None:
                total_reg = total_reg / train_len
                if self.prior_mode == 'seed+reg':
                    effective_weight = reg_weight
                else:
                    effective_weight = 0.3 * alpha
                total_pol_loss = total_pol_loss + effective_weight * total_reg

            total_pol_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()

            pol_loss_val = total_pol_loss.item()
            if do_reg and prior_train is not None:
                effective_weight = reg_weight if self.prior_mode == 'seed+reg' else 0.3 * alpha
                reg_loss_val = (effective_weight * total_reg).item()
            else:
                reg_loss_val = 0.0

        # Log metrics
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i, {
                'vf_loss': total_vf_loss1.item(),
                'pol_loss': pol_loss_val,
                'regularization_loss': reg_loss_val,
            }, self.niter)

        return total_vf_loss1.item(), pol_loss_val, reg_loss_val

    def update_all_targets(self):    
        """
        Update all target networks (called after normal updates have been performed for each agent).
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_critic2, a.critic2, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        """
        Prepare agents for training by setting them to training mode and moving them to the specified device.
        """
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
            a.critic.train()
            a.critic2.train()
            a.target_critic.train()
            a.target_critic2.train()

        # device transform
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
                a.critic2 = fn(a.critic2)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
                a.target_critic2 = fn(a.target_critic2)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        """
        Prepare agents for rollouts by setting them to evaluation mode and moving them to the specified device.
        """
        for a in self.agents:
            a.policy.eval()   
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file.
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG", gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3,
                      hidden_dim=64, lstm_hidden_dim=64, name='flocking', device='cpu', epsilon=0.1, noise=0.1,
                      use_ctm_actor=False, ctm_config=None, prior_mode='none'):
        """
        Instantiate instance of this class from multi-agent environment.
        """
        agent_init_params = []
        n_agents = env.num_agents  # total physical agents (e.g. 24)
        dim_input_policy = env.observation_space.shape[0]    # per-agent obs dim
        dim_output_policy = env.action_space.shape[0]        # per-agent action dim

        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for atype in env.agent_types]
        for algtype in alg_types:
            agent_init_params.append({'dim_input_policy': dim_input_policy,
                                      'dim_output_policy': dim_output_policy,
                                      'n_agents': n_agents})

        if name == 'assembly':
            init_dict = {'gamma': gamma, 'tau': tau, 'lr_actor': lr_actor, 'lr_critic': lr_critic,
                         'epsilon': epsilon, 'noise': noise, 'hidden_dim': hidden_dim,
                         'lstm_hidden_dim': lstm_hidden_dim,
                         'n_agents': n_agents, 'device': device, 'alg_types': alg_types,
                         'agent_init_params': agent_init_params,
                         'use_ctm_actor': use_ctm_actor, 'ctm_config': ctm_config,
                         'prior_mode': prior_mode}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):    
        """
        Instantiate instance of this class from file created by 'save' method.
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance

