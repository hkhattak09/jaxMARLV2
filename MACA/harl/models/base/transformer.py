"""
References:
1. nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
2. labml.ai: https://nn.labml.ai/transformers/models.html
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import numpy as np
from functools import partial
from torch.distributions import Categorical
from harl.models.base.rnn import RNNLayer
from harl.utils.envs_tools import (
    check,
    get_dim_from_act_space,
    get_shape_from_obs_space,
)
from harl.utils.models_tools import (
    init,
    get_active_func,
    get_init_method,
)


def default_init(module, initialization_method="orthogonal_", gain=0.01, active_fn=""):
    weight_init = get_init_method(initialization_method)
    if active_fn:
        if active_fn == "gelu":
            active_fn = "relu"
        gain = nn.init.calculate_gain(active_fn)
    weight_init = partial(weight_init, gain=gain)
    bias_init = partial(nn.init.constant_, val=0)
    if isinstance(module, nn.Linear):
        weight_init(module.weight)
        if module.bias is not None:
            bias_init(module.bias)
    elif isinstance(module, nn.Embedding):
        weight_init(module.weight)

def tfixup_init(module, gain=1.0, padding_idx=None):
    """
    T-Fixup initialization.
    Reference: https://github.com/layer6ai-labs/T-Fixup, https://proceedings.mlr.press/v119/huang20f.html
    """
    # Apply Xavier initialization for all parameters excluding input embeddings.
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    # Use Gaussian initialization N(0,embedding_dim^-0.5) for input embeddings
    elif isinstance(module, nn.Embedding):
        # weight: (num_embeddings, embedding_dim)
        nn.init.normal_(module.weight, mean=0, std=module.weight.size(-1)**-0.5)
        if padding_idx is not None:
            nn.init.constant_(module.weight[padding_idx], 0)

def nanogpt_init(module):
    """
    nanoGPT initialization.
    Reference: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

nn_init_registry = {
    "default": default_init,
    "tfixup": tfixup_init,
    "nanogpt": nanogpt_init,
}

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch <= 2.0 doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.ndim = ndim
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.n_block = cfg["n_block"]
        self.bias = cfg["bias"]
        self.dropout = cfg["dropout"]
        self.is_causal = cfg["is_causal"]

        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads
        self.query = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        # output projection
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash and self.is_causal:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("mask", torch.tril(torch.ones(self.n_block, self.n_block))
                                        .view(1, 1, self.n_block, self.n_block))

    def forward(self, query, key, value, output_attentions=False):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(query).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(key).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(value).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and not output_attentions:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.is_causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            if self.is_causal:
                att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.proj(y))
        return (y, att.detach()) if output_attentions else (y, None)


class EncodeBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        active_func = get_active_func(cfg["active_fn"])
        self.n_embd = cfg["n_embd"]
        self.bias = cfg["bias"]
        self.dropout = cfg["dropout"]

        self.ln1 = LayerNorm(self.n_embd, bias=self.bias)
        self.ln2 = LayerNorm(self.n_embd, bias=self.bias)
        self.attn = SelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_embd, 1 * self.n_embd, bias=self.bias),
            active_func,
            nn.Linear(1 * self.n_embd, self.n_embd, bias=self.bias),
            nn.Dropout(self.dropout),
        )

    # TODO GTrXL: use gating layer in place of residual connection
    def forward(self, x, output_attentions=False):
        z = self.ln1(x)
        y, att = self.attn(z, z, z, output_attentions)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, att


class DecodeBlock(EncodeBlock):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ln3 = LayerNorm(self.n_embd, bias=self.bias)
        self.attn2 = SelfAttention(cfg)

    def forward(self, x, src):
        z = self.ln1(x)
        x = x + self.attn(z, z, z)[0]
        x = x + self.attn2(query=self.ln3(x), key=src, value=src)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class Encoder(nn.Module):

    def __init__(
            self,
            args,
            obs_space,
            action_space,
            device=torch.device("cpu"),
        ):
        super().__init__()
        self.args = args
        self.cfg = args["transformer"]
        self.hidden_sizes = args["hidden_sizes"]
        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        active_func = get_active_func(self.cfg["active_fn"])
        self.action_type = action_space.__class__.__name__
        self.action_dim = get_dim_from_act_space(action_space)
        obs_shape = get_shape_from_obs_space(obs_space)
        self.obs_dim = obs_shape[0]
        self.n_encode_layer = self.cfg["n_encode_layer"]
        self.n_decode_layer = self.cfg["n_decode_layer"]
        self.n_head = self.cfg["n_head"]
        self.n_embd = self.cfg["n_embd"]
        self.zs_dim = self.cfg["zs_dim"]
        self.n_block = self.cfg["n_block"]
        self.bias = self.cfg["bias"]
        self.dropout = self.cfg["dropout"]
        self.head_aggr = self.cfg.get("aggregation", "mean")
        self.att_sigma = min(self.cfg["att_sigma"] / self.n_block, 1.0)
        self.vq_bsln_coef = self.cfg["vq_bsln_coef"]
        self.vq_coma_bsln_coef = self.cfg["vq_coma_bsln_coef"]
        self.att_roll_res = self.cfg.get("att_roll_res", False)

        if self.use_feature_normalization:
            self.feature_norm = LayerNorm(self.obs_dim, bias=self.bias)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.n_embd, bias=self.bias),
            active_func,
            LayerNorm(self.n_embd, bias=self.bias),
        )
        self.drop = nn.Dropout(self.dropout)
        # TODO may add a CLS token embedding like in ViT: https://nn.labml.ai/transformers/vit/index.html
        self.ln1 = LayerNorm(self.n_embd, bias=self.bias)
        self.blocks = nn.Sequential(*[EncodeBlock(self.cfg) for _ in range(self.n_encode_layer)])
        self.s_encoder = nn.Sequential(
            nn.Linear(self.n_block * self.n_embd, self.zs_dim, bias=self.bias),
            active_func,
            LayerNorm(self.zs_dim, bias=self.bias),
        )
        self.v_head = nn.Sequential(
            nn.Linear(self.zs_dim, self.n_embd, bias=self.bias),
            active_func,
            LayerNorm(self.n_embd, bias=self.bias),
            nn.Linear(self.n_embd, 1, bias=self.bias),
        )

        self.ln2 = LayerNorm(self.n_embd, bias=self.bias)
        if self.n_decode_layer:
            self.cross_blocks = nn.Sequential(*[DecodeBlock(self.cfg) for _ in range(self.n_decode_layer)])
            self.act_encoder = nn.Sequential(
                nn.Linear(self.action_dim, self.n_embd, bias=self.bias),
                active_func,
                LayerNorm(self.n_embd, bias=self.bias),
            )
            self.sa_encoder = nn.Sequential(
                nn.Linear(self.n_block * self.n_embd, self.zs_dim, bias=self.bias),
                active_func,
                LayerNorm(self.zs_dim, bias=self.bias),
            )
            self.q_head = nn.Sequential(
                nn.Linear(self.zs_dim, self.n_embd, bias=self.bias),
                active_func,
                LayerNorm(self.n_embd, bias=self.bias),
                nn.Linear(self.n_embd, self.n_embd, bias=self.bias),
                active_func,
                LayerNorm(self.n_embd, bias=self.bias),
                nn.Linear(self.n_embd, 1, bias=self.bias),
            )
        else:
            self.sa_encoder = nn.Linear(
                self.zs_dim + self.n_block * self.action_dim,
                self.zs_dim,
                bias=self.bias,
            )
            self.q_head = nn.Linear(self.zs_dim, 1, bias=self.bias)

        if self.action_type != "Discrete":
            self.log_std = torch.nn.Parameter(torch.ones(self.action_dim, dtype=torch.float32))

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        """
        Initialize the weights of the model.
        """
        init_fn = nn_init_registry[self.cfg["weight_init"]]
        self.apply(init_fn)
        if self.cfg["weight_init"] == "default":
            for name, param in self.named_parameters():
                if (
                    name.endswith(("mlp.0.weight",))
                    or name in [
                        "obs_encoder.0.weight",
                        "s_encoder.0.weight",
                        "v_head.0.weight",
                    ]
                    or (self.n_decode_layer and name in [
                        "act_encoder.0.weight",
                        "sa_encoder.0.weight",
                        "q_head.0.weight",
                        "q_head.2.weight",
                        # "r_head.0.weight",
                    ])
                ):
                    gain = nn.init.calculate_gain("relu")
                    nn.init.orthogonal_(param, gain=gain)
        elif self.cfg["weight_init"] == "nanogpt":
            # apply special scaled init to the residual projections, per GPT-2 paper
            for pn, p in self.named_parameters():
                if pn.endswith((
                    "attn.proj.weight",
                    "mlp.2.weight",
                )):
                    nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_encode_layer))
        elif self.cfg["weight_init"] == "tfixup":
            temp_state_dic = {}
            # Scale ve and we matrices in each encoder attention block and
            # weight matrices in each encoder MLP block by 0.67N^−0.25.
            for name, param in self.blocks.named_parameters():
                if name.endswith((
                    "mlp.0.weight",
                    "mlp.2.weight",
                    "attn.proj.weight",
                )):
                    temp_state_dic[f"blocks.{name}"] = (0.67*self.n_encode_layer**-0.25) * param
                elif name.endswith((
                    "attn.value.weight",
                )):
                    temp_state_dic[f"blocks.{name}"] = (0.67*self.n_encode_layer**-0.25) * param * (2**0.5)

            # Scale vd and wd matrices in each decoder attention block, weight
            # matrices in each decoder MLP block and input embeddings x and y in
            # encoder and decoder by (9N)^−0.25
            if self.n_decode_layer:
                for name, param in self.cross_blocks.named_parameters():
                    if name.endswith((
                        "mlp.0.weight",
                        "mlp.2.weight",
                        "attn.proj.weight",
                    )):
                        temp_state_dic[f"cross_blocks.{name}"] = (9*self.n_decode_layer)**-0.25 * param
                    elif name.endswith((
                        "attn.value.weight",
                    )):
                        temp_state_dic[f"cross_blocks.{name}"] = (9*self.n_decode_layer)**-0.25 * param * (2**0.5)

            if isinstance(self.obs_encoder, nn.Embedding):
                temp_state_dic["obs_encoder.weight"] = (9*self.n_encode_layer)**-0.25 * self.obs_encoder.weight

            if self.n_decode_layer and isinstance(self.act_encoder, nn.Embedding):
                temp_state_dic["act_encoder.weight"] = (9*self.n_decode_layer)**-0.25 * self.act_encoder.weight

            for name in self.state_dict():
                if name not in temp_state_dic:
                    temp_state_dic[name] = self.state_dict()[name]
            self.load_state_dict(temp_state_dic)


    def forward(self, obs, action, policy_prob, rnn_states, masks, output_attentions=False):
        """Compute values from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            action: (np.ndarray / torch.Tensor) action inputs into network.
            policy_prob: (np.ndarray / torch.Tensor) policy probabilities.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            v_loc: (torch.Tensor) V value function predictions.
            q_loc: (torch.Tensor) Q value function predictions.
            eq_loc: (torch.Tensor) Q value function expected wrt pi predictions.
            s_rep: (torch.Tensor) encoded state representations.
            sa_rep: (torch.Tensor) encoded state-action representations.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # n_block: number of agents or entities
        # obs: (batch, n_block, obs_dim)
        # action: (batch, n_block, action_dim), one-hot/logits?
        obs = check(obs).to(**self.tpdv)
        action = check(action).long()
        action = F.one_hot(action, num_classes=self.action_dim).squeeze(dim=2).to(**self.tpdv)
        pi = check(policy_prob).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.use_feature_normalization:
            obs = self.feature_norm(obs)

        s_rep = self.drop(self.obs_encoder(obs))
        B, T, _ = s_rep.shape
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            s_rep, rnn_states = self.rnn(
                s_rep.view(-1, *s_rep.shape[2:]),
                rnn_states.view(-1, *rnn_states.shape[2:]),
                masks.view(-1, *masks.shape[2:]),
            )
            s_rep = s_rep.view(B, T, *s_rep.shape[1:])
            if rnn_states.size(0) == B*T:
                rnn_states = rnn_states.view(B, T, *rnn_states.shape[1:])

        all_self_attns = ()
        for block in self.blocks:
            s_rep, self_attn = block(s_rep, output_attentions)
            all_self_attns += (self_attn,)
        s_rep = self.ln1(s_rep)
        zs = self.s_encoder(s_rep.view(B, -1))
        v_loc = self.v_head(zs)
        if output_attentions:
            # Stack over layers and aggregate over heads: (B, n_layer, nh, T, T) -> (B, n_layer, T, T)
            all_self_attns = torch.stack(all_self_attns, dim=1)
            all_self_attns = getattr(torch, self.head_aggr)(all_self_attns, dim=2)
            # Compute attention rollout: (B, n_layer, T, T)
            joint_attentions = compute_joint_attention(all_self_attns, add_residual=self.att_roll_res)
            # Compute mixed action-pi for each agent: (B, T, T, A)
            mix_a_pi, coma_a_pi = self.get_mixed_action_pi(action, pi, joint_attentions)
            # Compute baseline weights for each agent: (B, T, 3)
            baseline_weights = self.get_baseline_weights(joint_attentions)

        if self.n_decode_layer:
            sa_rep = self.act_encoder(action)
            for crs_block in self.cross_blocks:
                sa_rep = crs_block(x=sa_rep, src=s_rep)
            sa_rep = self.ln2(sa_rep)
            zsa = self.sa_encoder(sa_rep.view(B, -1))
            spi_rep = self.act_encoder(pi)
            for crs_block in self.cross_blocks:
                spi_rep = crs_block(x=spi_rep, src=s_rep)
            spi_rep = self.ln2(spi_rep)
            zspi = self.sa_encoder(spi_rep.view(B, -1))
            if output_attentions:
                s_rep = s_rep.unsqueeze(dim=1).repeat(1, T, 1, 1).view(B*T, T, -1)
                mix_a_pi = mix_a_pi.view(B*T, T, -1)
                sapi_rep = self.act_encoder(mix_a_pi)
                for crs_block in self.cross_blocks:
                    sapi_rep = crs_block(x=sapi_rep, src=s_rep)
                sapi_rep = self.ln2(sapi_rep)
                zsapi = self.sa_encoder(sapi_rep.view(B*T, -1))

                coma_a_pi = coma_a_pi.view(B*T, T, -1)
                sapi_coma_rep = self.act_encoder(coma_a_pi)
                for crs_block in self.cross_blocks:
                    sapi_coma_rep = crs_block(x=sapi_coma_rep, src=s_rep)
                sapi_coma_rep = self.ln2(sapi_coma_rep)
                zsapi_coma = self.sa_encoder(sapi_coma_rep.view(B*T, -1))
        else:
            zsa = self.sa_encoder(torch.cat([zs, action.reshape(B, -1)], dim=-1))
            zspi = self.sa_encoder(torch.cat([zs, pi.reshape(B, -1)], dim=-1))
            if output_attentions:
                zsapi = self.sa_encoder(torch.cat([
                    zs.unsqueeze(dim=1).repeat(1, T, 1).view(B*T, -1),
                    mix_a_pi.reshape(B*T, -1)], dim=-1))
                zsapi_coma = self.sa_encoder(torch.cat([
                    zs.unsqueeze(dim=1).repeat(1, T, 1).view(B*T, -1),
                    coma_a_pi.reshape(B*T, -1)], dim=-1))
        q_loc = self.q_head(zsa)
        eq_loc = self.q_head(zspi)
        if output_attentions:
            vq_loc = self.q_head(zsapi)
            vq_loc = vq_loc.view(B, T, -1)
            vq_loc_coma = self.q_head(zsapi_coma)
            vq_loc_coma = vq_loc_coma.view(B, T, -1)
        else:
            vq_loc = None
            vq_loc_coma = None
            baseline_weights = None
            joint_attentions = None

        return (
            v_loc,
            q_loc,
            eq_loc,
            vq_loc,
            vq_loc_coma,
            baseline_weights,
            joint_attentions[:, -1], # (B, T, T)
            zs,
            zsa,
            rnn_states,
        )

    def get_mixed_action_pi(self, action, pi, joint_attentions, layer=-1):
        """
        Construct a mixed action pi tensor. Based on the specified layer of joint attention matrix,
        for attention value greater than sigma, use the corresponding value from pi,
        otherwise use the corresponding value from action.
        Args:
            action: (torch.Tensor) (batch, n_block, action_dim) action tensor.
            pi: (torch.Tensor) (batch, n_block, action_dim) policy probabilities.
            joint_attentions: (torch.Tensor) (batch, layers, n_block, n_block) joint attention matrix.
            layer: (int) layer of joint attention matrix to use.
        Output:
            mixed_action_pi: (torch.Tensor) (batch, n_block, n_block, action_dim) mixed action pi tensor.
        """
        B, T, A = action.shape
        pi = pi.unsqueeze(dim=1).repeat(1, T, 1, 1)     # (B, T, T, A)
        action = action.unsqueeze(dim=1).repeat(1, T, 1, 1) # (B, T, T, A)
        joint_attentions = joint_attentions[:, layer].unsqueeze(dim=-1) # (B, T, T, 1)
        mixed_action_pi = torch.where(joint_attentions >= self.att_sigma, pi, action)    # (B, T, T, A)
        # Subset must include the ego agent => set diagonal to pi
        diag_att = torch.eye(T).view(1, T, T, 1).to(**self.tpdv)
        mixed_action_pi = torch.where(diag_att >= self.att_sigma, pi, mixed_action_pi)    # (B, T, T, A)
        coma_action_pi = torch.where(diag_att >= self.att_sigma, pi, action)    # (B, T, T, A)
        return mixed_action_pi, coma_action_pi

    def get_baseline_weights(self, joint_attentions, layer=-1):
        B, _, T, _ = joint_attentions.shape
        joint_attentions = joint_attentions[:, layer] # (B, T, T)
        self_weights = joint_attentions.diagonal(dim1=-2, dim2=-1).unsqueeze(dim=-1) # (B, T, 1)
        group_weights = torch.zeros_like(joint_attentions)
        group_weights = torch.where(joint_attentions >= self.att_sigma, joint_attentions, group_weights)
        diag_att = torch.eye(T).view(1, T, T).to(**self.tpdv)
        group_weights = torch.where(diag_att >= self.att_sigma, joint_attentions, group_weights)
        group_weights = group_weights.sum(dim=-1, keepdim=True) - self_weights # (B, T, 1)
        joint_weights = joint_attentions.sum(dim=-1, keepdim=True) # (B, T, 1)
        self_weights = self.vq_bsln_coef * self_weights
        group_weights = self.vq_coma_bsln_coef * group_weights
        joint_weights = (joint_weights - self_weights - group_weights).clamp(min=0.0, max=1.0)  # TODO remove clamp
        all_weights = torch.cat([self_weights, group_weights, joint_weights], dim=-1) # (B, T, 3)
        return all_weights

    def zero_std(self):
        if self.action_type != "Discrete":
            self.log_std.data = torch.zeros(self.action_dim, dtype=torch.float32)

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, learning_rate, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.cfg["wght_decay"]},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=self.cfg["betas"], **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        L, H, Q, T = self.n_encode_layer+self.n_decode_layer, self.n_head, self.n_embd//self.n_head, self.n_block
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

def compute_joint_attention(att_mat, add_residual=True):
    """
    Compute attention rollout.
    Args:
        att_mat: (batch, layers, n_block, n_block) attention matrix.
        add_residual: (bool) whether to add residual attention.
    """
    _, L, T, _ = att_mat.shape
    if add_residual:
        residual_att = torch.eye(
            T, dtype=torch.float32, device=att_mat.device,
        ).view(1, 1, T, T)
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)
    else:
       aug_att_mat =  att_mat

    joint_attentions = torch.zeros_like(aug_att_mat, dtype=torch.float32, device=aug_att_mat.device)
    joint_attentions[:, 0] = aug_att_mat[:, 0]
    for i in range(1, L):
        joint_attentions[:, i] = aug_att_mat[:, i] @ joint_attentions[:, i-1]

    return joint_attentions