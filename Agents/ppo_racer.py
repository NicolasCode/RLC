'''
PPO (Proximal Policy Optimization) para CarRacing con acciones discretas.

Basado en la receta estándar de PPO para Atari (CleanRL / Schulman et al. 2017):
  * Red actor-crítico con la CNN de "Nature DQN".
  * Ventaja con GAE(lambda).
  * Pérdida recortada (clip) + pérdida de valor + bonus de entropía.
  * Varias épocas de minibatches sobre cada rollout.
'''
import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    '''
    Entrada: (batch, 4, 96, 96) uint8. Salida: logits de las acciones y valor.
    '''
    def __init__(self, n_actions:int, in_channels:int=4):
        super().__init__()
        self.features = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),   # -> 32x23x23
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),            # -> 64x10x10
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),            # -> 64x8x8
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        h = self.features(x.float() / 255.0)
        return self.actor(h), self.critic(h).squeeze(-1)

    def value(self, x):
        return self.forward(x)[1]

    def act(self, x, action=None, greedy=False):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class PPO:
    def __init__(self,
                 net:ActorCritic,
                 lr=2.5e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip=0.1,
                 epochs=4,
                 minibatch_size=256,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 max_grad_norm=0.5):
        self.net = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def compute_gae(self, rewards, values, dones, next_value):
        '''
        rewards, values, dones: (T, N). dones[t] = el episodio terminó
        (de verdad) tras la acción t. next_value: (N,) valor del estado
        posterior al último paso.
        '''
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last = torch.zeros_like(next_value)
        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nv * nonterminal - values[t]
            last = delta + self.gamma * self.gae_lambda * nonterminal * last
            advantages[t] = last
        return advantages, advantages + values

    def update(self, obs, actions, logprobs, advantages, returns):
        b = obs.shape[0]
        stats = {'pg_loss': [], 'v_loss': [], 'entropy': [], 'approx_kl': [], 'clipfrac': []}
        for _ in range(self.epochs):
            idx = torch.randperm(b)
            for start in range(0, b, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                _, newlogp, entropy, newv = self.net.act(obs[mb], actions[mb])
                logratio = newlogp - logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.clip).float().mean()

                adv = advantages[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio,
                                    -adv * ratio.clamp(1 - self.clip, 1 + self.clip)).mean()

                v_loss = 0.5 * ((newv - returns[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss - self.ent_coef * ent + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats['pg_loss'].append(pg_loss.item())
                stats['v_loss'].append(v_loss.item())
                stats['entropy'].append(ent.item())
                stats['approx_kl'].append(approx_kl.item())
                stats['clipfrac'].append(clipfrac.item())
        return {k: float(np.mean(v)) for k, v in stats.items()}
