import torch
from algorithm.utils import soft_update, average_gradients
from algorithm.utils import DDPGAgent

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types, epsilon, noise, gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3,  
                 hidden_dim=64, device='cpu', discrete_action=False):
        """
        Initialize the MADDPG object with given parameters.
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.epsilon = epsilon
        self.noise = noise
        self.agents = [DDPGAgent(lr_actor=lr_actor, 
                                 lr_critic=lr_critic, 
                                 discrete_action=discrete_action, 
                                 hidden_dim=hidden_dim, 
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

        self.spatial_loss = False
        self.temporal_loss = False 
        self.niter = 0

    @property           
    def policies(self):
        """
        Get the policies of all agents.
        """
        return [a.policy for a in self.agents]

    def target_policies(self, agent_i, obs):
        """
        Get the target policies of a specific agent.
        """
        return self.agents[agent_i].target_policy(obs)

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

    def step(self, observations, start_stop_num, explore=False):
        """
        Take a step forward in environment with all agents.
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """  
        actions = []      
        log_pis = []          
        for i in range(len(start_stop_num)):
            action, log_pi = self.agents[i].step(observations[:, start_stop_num[i]].t(), explore=explore)
            actions.append(action)
            log_pis.append(log_pi)
        return actions, log_pis
    
    def step_rew(self, observations, start_stop_num):
        """
        Take a step forward in environment with all agents to get intrinsic rewards.
        Inputs:
            observations: List of observations for each agent
        Outputs:
            rewards: List of intrinsic rewards for each agent
        """                                                           
        return [self.agents[i].step_rew(observations[:, start_stop_num[i]].t()) for i in range(len(start_stop_num))]

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

        ######################### Update Critic #########################       
        curr_agent.critic_optimizer.zero_grad()
        
        # Compute target actions using target policies
        all_trgt_acs = self.target_policies(agent_i, next_obs)  
        trgt_vf_in = torch.cat((next_obs, all_trgt_acs), dim=1) 

        # Calculate target Q-value using Bellman equation
        target_value = (rews + self.gamma * curr_agent.target_critic(trgt_vf_in) * (1 - dones))
        
        # Compute current Q-value
        vf_in = torch.cat((obs, acs), dim=1)
        actual_value = curr_agent.critic(vf_in)
        
        # Critic loss (TD error)
        vf_loss = MSELoss(actual_value, target_value.detach())

        # Backward pass for critic
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        # torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        ######################### Update Actor #########################
        curr_agent.policy_optimizer.zero_grad()  

        # Get current policy output
        if not self.discrete_action:
            curr_pol_out = curr_agent.policy(obs)
            curr_pol_vf_in = curr_pol_out

        # Actor loss: maximize Q-value (negative for gradient ascent)
        all_pol_acs = curr_pol_vf_in  
        vf_in = torch.cat((obs, all_pol_acs), dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()

        # Add regularization term towards prior actions if provided
        regularization_term = torch.tensor(0.0, requires_grad=True)
        if acs_prior is not None:
            mse_loss = torch.nn.MSELoss()
            
            # Filter out near-zero prior actions (likely invalid/padding)
            mask = (acs_prior.abs() < 1e-2).all(dim=1)
            valid_mask = ~mask
            filtered_all_pol_acs = all_pol_acs[valid_mask]
            filtered_acs_prior = acs_prior[valid_mask]
            
            # Compute regularization loss only for valid prior actions
            if filtered_all_pol_acs.numel() == 0 or filtered_acs_prior.numel() == 0:
                regularization_term = torch.tensor(0.0, requires_grad=True)
            else:
                regularization_term = mse_loss(filtered_all_pol_acs, filtered_acs_prior)

            # Add weighted regularization to policy loss
            pol_loss = pol_loss + 0.3 * alpha * regularization_term

        # Backward pass for actor
        pol_loss.backward()     
                                
        if parallel:
            average_gradients(curr_agent.policy)
        # torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)   
        curr_agent.policy_optimizer.step() 
        
        # Log metrics if logger is provided
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i, {
                'vf_loss': vf_loss, 
                'pol_loss': pol_loss, 
                'regularization_loss': 0.5 * alpha * regularization_term
            }, self.niter)

    def update_all_targets(self):    
        """
        Update all target networks (called after normal updates have been performed for each agent).
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)   
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        """
        Prepare agents for training by setting them to training mode and moving them to the specified device.
        """
        for a in self.agents:
            a.policy.train()  
            a.target_policy.train()
            a.target_critic.train()

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
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
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
                        hidden_dim=64, name='flocking', device='cpu', epsilon=0.1, noise=0.1):
        """
        Instantiate instance of this class from multi-agent environment.
        """
        agent_init_params = []
        dim_input_policy=env.observation_space.shape[0]
        dim_output_policy=env.action_space.shape[0]
        dim_input_critic=env.observation_space.shape[0] + env.action_space.shape[0]

        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for atype in env.agent_types]   
        for algtype in alg_types:  
            agent_init_params.append({'dim_input_policy': dim_input_policy,
                                      'dim_output_policy': dim_output_policy,
                                      'dim_input_critic': dim_input_critic})

        if name == 'assembly':
            init_dict = {'gamma': gamma, 'tau': tau, 'lr_actor': lr_actor, 'lr_critic': lr_critic, 'epsilon': epsilon, 'noise': noise, 
                         'hidden_dim': hidden_dim, 'device': device, 'alg_types': alg_types, 'agent_init_params': agent_init_params}
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

    @classmethod
    def init_from_save_with_id(cls, filename, list_id):    
        """
        Instantiate instance of this class from file created by 'save' method with specific agent IDs.
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for i in range(len(instance.agents)):
            a = instance.agents[i]
            policy_id = list_id[i]
            if policy_id == 2:
                continue
            params = save_dict['agent_params'][policy_id]
            a.load_params(params)
        return instance