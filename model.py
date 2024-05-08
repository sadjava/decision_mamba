from dataclasses import dataclass
from typing import Union
import math
from einops import rearrange, repeat, einsum

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    d_model: int

    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True,
    bias: bool = False,

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4


    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
    

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner, 
            out_channels=args.d_inner, 
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1)
        
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        dt_init_std = args.dt_rank**-0.5 * args.dt_scale
        if args.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif args.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
    
        dt = torch.exp(torch.rand(args.d_inner) * (math.log(args.dt_max) - math.log(args.dt_min)) + math.log(args.dt_min)).clamp(min=args.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :1]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        
        y = torch.stack(ys, dim = 1)
        y = y + u * D
        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class Block(nn.Module):
    def __init__(self, n_embd, n_inner, layer_norm_epsilon, drop_p):
        super().__init__()
        hidden_size = n_embd
        inner_dim = n_inner if n_inner is not None else 4 * hidden_size
        
        self.norm_mamba = RMSNorm(hidden_size)
        self.mamba = MambaBlock(ModelArgs(d_model=hidden_size))

        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.mlp_channels = nn.Sequential(
            nn.Linear(hidden_size, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_size),
            nn.Dropout(drop_p)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mamba(self.norm_mamba(x))
        x = x + self.mlp_channels(self.ln_2(x))
        return x

class GPT2Model(nn.Module):
    def __init__(self,
                 n_embd: int,
                 n_inner: int,
                 embd_drop: float, 
                 drop_p: float,
                 n_layer: int, 
                 layer_norm_epsilon: float,
                 **kwargs):
        super().__init__()
        self.drop = nn.Dropout(embd_drop)
        self.h = nn.ModuleList([Block(n_embd, n_inner, layer_norm_epsilon, drop_p) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
    
    def forward(self, input_embeds=None):
        hidden_states = input_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states)
        
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class DecisionMamba(nn.Module):
    def __init__(self,
                 state_dim,
                 act_dim,
                 hidden_size,
                 max_length=None, 
                 max_ep_len=4096,
                 action_tanh=True,
                 **kwargs
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.mamba = GPT2Model(n_embd=hidden_size, **kwargs)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

    def forward(self, timesteps, states, actions, returns_to_go):
        batch_size, seq_length = states.shape[0], states.shape[1]

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings
        actions_embeddings = self.embed_action(actions) + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, actions_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        x = self.mamba(input_embeds=stacked_inputs)

        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        state_reps = x[:, 1]
        action_preds = self.predict_action(state_reps)
        return action_preds
    
    def get_action(self, states, actions, returns_to_go, timesteps):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:, -self.max_length]
        actions = actions[:, -self.max_length]
        returns_to_go = returns_to_go[:, -self.max_length]
        timesteps = timesteps[:, -self.max_length]

        states = torch.cat(
            [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
            dim=1).to(dtype=torch.float32)
        actions = torch.cat(
            [torch.zeros((actions.shape[0], self.max_length-actions.shape[1], self.act_dim), device=actions.device), actions],
            dim=1).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
            dim=1).to(dtype=torch.float32)
        timesteps = torch.cat(
            [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
            dim=1).to(dtype=torch.float32)
        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[0, -1]
    