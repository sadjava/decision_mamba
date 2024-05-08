import random
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def discounted_returns(rewards, gamma):
    returns = np.zeros_like(rewards)
    returns[-1] = rewards[-1]
    for i in reversed(range(rewards.shape[0] - 1)):
        returns[i] = rewards[i] + gamma * returns[i + 1]
    return returns

def get_d4rl_normalized_score(score, env_name):
    env_key = env_name.split('-')[0].lower()
    assert env_key in REF_MAX_SCORE
    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])

def get_d4rl_dataset_stats(env_name):
    return D4RL_DATASET_STATS[env_name]


def evaluate_env(model, device, context_len, env, rtg_target, rtg_scale,
                 n_episodes=10, max_episode_len=1000, state_mean=None, state_std=None,
                 render=False):
    
    eval_batch_size = 1

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim, )).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim, )).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)
    
    timesteps = torch.arange(max_episode_len)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():
        for _ in range(n_episodes):

            actions = torch.zeros((eval_batch_size, max_episode_len, action_dim), 
                                  dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_episode_len, state_dim),
                                 dtype=torch.float32, device=device)
            rewards_to_go = torch.zeros((eval_batch_size, max_episode_len, 1),
                                        dtype=torch.float32, device=device)
            
            running_state = env.reset()
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            for t in range(max_episode_len):
                total_timesteps += 1

                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std

                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                if t < context_len:
                    action_preds = model(timesteps[:, :context_len], 
                                               states[:, :context_len],
                                               actions[:, :context_len],
                                               rewards_to_go[:, :context_len])
                    action = action_preds[0, t].detach()
                else:
                    action_preds = model(timesteps[:, t - context_len + 1:t + 1], 
                                               states[:, t - context_len + 1:t + 1],
                                               actions[:, t - context_len + 1:t + 1],
                                               rewards_to_go[:, t - context_len + 1:t + 1])
                    action = action_preds[0, -1].detach()

                running_state, running_reward, done, _ = env.step(action.cpu().numpy())
                actions[0, t] = action

                total_reward += running_reward

                if render:
                    env.render()
                if done:
                    break 

    results['eval/avg_reward'] = total_reward / n_episodes
    results['eval/avg_ep_len'] = total_timesteps / n_episodes

    return results


class D4RLDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale):

        self.context_len = context_len

        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)
        
        min_len = 10 ** 6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            traj['returns_to_go'] = discounted_returns(traj['rewards'], 1.0) / rtg_scale

        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0)

        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std
        
    
    def get_state_stats(self):
        return self.state_mean, self.state_std
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]
        
        if traj_len >= self.context_len:
            s1 = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][s1:s1 + self.context_len])
            actions = torch.from_numpy(traj['actions'][s1:s1 + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][s1:s1 + self.context_len])
            timesteps = torch.arange(s1, s1 + self.context_len)

            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros([padding_len] + list(states.shape[1:]),
                                            dtype=states.dtype)],
                                dim=0)
            
            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                 torch.zeros([padding_len] + list(actions.shape[1:]),
                                             dtype=actions.dtype)],
                                 dim=0)
            
            
            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                 torch.zeros([padding_len] + list(returns_to_go.shape[1:]),
                                             dtype=returns_to_go.dtype)],
                                 dim=0)
            
            timesteps = torch.arange(self.context_len)
            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long), 
                                   torch.zeros(padding_len, dtype=torch.long)],
                                   dim=0)
            
        return timesteps, states, actions, returns_to_go, traj_mask