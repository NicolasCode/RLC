'''
TD3 (Twin Delayed DDPG, Fujimoto et al. 2018): la versión para acciones
continuas de nuestro DQN. DQN elige la acción con argmax_a Q(s, a), lo que
exige una lista finita de acciones. TD3 conserva sus ingredientes (replay
buffer, red objetivo, Double Q-learning con dos críticos) y añade un actor
que produce directamente la acción continua (dirección, acelerador/freno).

Mismo entorno, misma CNN, mismo replay buffer y mismo ritmo de gradiente
que train_dqn.py (un minibatch de 64 cada 8 decisiones), para comparar.

Detalles para aprender desde píxeles (como SAC-AE / DrQ, Yarats et al.):
la CNN es del crítico y solo la entrena la pérdida del crítico; el actor
usa sus características sin propagar gradiente hacia la CNN.

Uso:
    python train_td3.py --total-steps 300000

Guarda en --out-dir (por defecto models/td3_racer/): last.pt, progress.csv
'''
import argparse
import copy
import csv
import os
import time
from collections import deque

import numpy as np
import torch
from torch import nn
import gymnasium as gym

from Agents.ppo_racer import layer_init
from train_dqn import ReplayBuffer
from Utils.car_env import make_env, CONTINUOUS_DIM


class Encoder(nn.Module):
    '''Misma CNN "Nature DQN" que PPO y DQN.'''
    def __init__(self, in_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8 * 8, 512)), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)


class Critic(nn.Module):
    '''Dos cabezas Q(s, a) sobre la CNN compartida (Double Q-learning).'''
    def __init__(self, action_dim, in_channels=4):
        super().__init__()
        self.encoder = Encoder(in_channels)
        def head():
            return nn.Sequential(layer_init(nn.Linear(512 + action_dim, 256)), nn.ReLU(),
                                 layer_init(nn.Linear(256, 1), std=1.0))
        self.q1, self.q2 = head(), head()

    def forward(self, h, a):
        ha = torch.cat([h, a], dim=-1)
        return self.q1(ha).squeeze(-1), self.q2(ha).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.net = nn.Sequential(layer_init(nn.Linear(512, 256)), nn.ReLU(),
                                 layer_init(nn.Linear(256, action_dim), std=0.01), nn.Tanh())

    def forward(self, h):
        return self.net(h)


class TD3Policy(nn.Module):
    '''CNN del crítico + actor: lo necesario para conducir (evaluate_ppo.py).'''
    def __init__(self, encoder, actor):
        super().__init__()
        self.encoder, self.actor = encoder, actor

    def act(self, x, greedy=True):
        # Devuelve la acción en (0, 1) como la política Beta de PPO
        return (self.actor(self.encoder(x)) + 1) / 2, None, None, None


class ContinuousReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, obs_shape, action_dim):
        super().__init__(capacity, obs_shape)
        self.actions = np.zeros((capacity, action_dim), np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--total-steps', type=int, default=300_000, help='decisiones del agente en total')
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--buffer-size', type=int, default=50_000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005, help='promedio suave de las redes objetivo')
    p.add_argument('--learning-starts', type=int, default=5_000)
    p.add_argument('--policy-delay', type=int, default=2)
    p.add_argument('--expl-noise', type=float, default=0.2, help='ruido gaussiano de exploración')
    p.add_argument('--target-noise', type=float, default=0.2)
    p.add_argument('--noise-clip', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--out-dir', default=os.path.join('models', 'td3_racer'))
    return p.parse_args()


def soft_update(target, source, tau):
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(1 - tau).add_(tau * s)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    envs = gym.vector.AsyncVectorEnv(
        [make_env(seed=args.seed + i, actions='continuous') for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    critic = Critic(CONTINUOUS_DIM)
    actor = Actor(CONTINUOUS_DIM)
    critic_target, actor_target = copy.deepcopy(critic), copy.deepcopy(actor)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    rb = ContinuousReplayBuffer(args.buffer_size, (4, 96, 96), CONTINUOUS_DIM)

    csv_file = open(os.path.join(args.out_dir, 'progress.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'frames', 'minutes', 'episodes', 'return', 'tiles_pct',
                     'lap_rate', 'loss', 'q_mean', 'actor_loss'])

    ep_returns, ep_tiles, ep_laps = deque(maxlen=50), deque(maxlen=50), deque(maxlen=50)
    losses, qs, actor_losses = deque(maxlen=200), deque(maxlen=200), deque(maxlen=200)
    episodes, grad_steps, global_step = 0, 0, 0
    obs, _ = envs.reset(seed=args.seed)
    t0 = time.time()
    next_log = 1024
    N = args.num_envs

    while global_step < args.total_steps:
        torch.set_num_threads(1)
        if global_step < args.learning_starts:
            actions = np.random.uniform(-1, 1, (N, CONTINUOUS_DIM)).astype(np.float32)
        else:
            with torch.no_grad():
                actions = actor(critic.encoder(torch.from_numpy(obs))).numpy()
            actions = np.clip(actions + np.random.normal(0, args.expl_noise, actions.shape),
                              -1, 1).astype(np.float32)
        next_obs, reward, term, trunc, info = envs.step(actions)

        real_next = next_obs.copy()
        if 'final_obs' in info:
            for i in np.where(term | trunc)[0]:
                real_next[i] = info['final_obs'][i]
            fi = info['final_info']
            for i in np.where(term | trunc)[0]:
                episodes += 1
                ep_returns.append(fi['raw_return'][i])
                ep_tiles.append(fi['tiles'][i] / fi['track_len'][i])
                ep_laps.append(float(fi['lap'][i]))
        # Cortar por tiempo no es un final real: se sigue haciendo bootstrap
        rb.add(obs, real_next, actions, reward, term.astype(np.float32))
        obs = next_obs
        global_step += N

        if global_step >= args.learning_starts:
            torch.set_num_threads(args.threads)
            o, no, a, r, d = rb.sample(args.batch_size)
            with torch.no_grad():
                # Suavizado de la política objetivo + mínimo de los dos críticos
                noise = (torch.randn_like(a) * args.target_noise).clamp(-args.noise_clip, args.noise_clip)
                h_next = critic_target.encoder(no)
                a_next = (actor_target(h_next) + noise).clamp(-1, 1)
                tq1, tq2 = critic_target(h_next, a_next)
                target = r + args.gamma * (1 - d) * torch.min(tq1, tq2)
            h = critic.encoder(o)
            q1, q2 = critic(h, a)
            loss = nn.functional.smooth_l1_loss(q1, target) + nn.functional.smooth_l1_loss(q2, target)
            critic_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            critic_opt.step()
            grad_steps += 1
            losses.append(loss.item())
            qs.append(q1.mean().item())

            if grad_steps % args.policy_delay == 0:
                # El actor maximiza Q1; no entrena la CNN (características sin gradiente)
                h_det = h.detach()
                actor_loss = -critic(h_det, actor(h_det))[0].mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                actor_losses.append(actor_loss.item())
                soft_update(critic_target, critic, args.tau)
                soft_update(actor_target, actor, args.tau)

        if global_step >= next_log:
            next_log += 1024
            minutes = (time.time() - t0) / 60
            mean_ret = float(np.mean(ep_returns)) if ep_returns else float('nan')
            tiles = float(np.mean(ep_tiles)) * 100 if ep_tiles else float('nan')
            laps = float(np.mean(ep_laps)) if ep_laps else float('nan')
            loss_m = float(np.mean(losses)) if losses else float('nan')
            q_m = float(np.mean(qs)) if qs else float('nan')
            al_m = float(np.mean(actor_losses)) if actor_losses else float('nan')
            writer.writerow([global_step, global_step * 4, round(minutes, 2), episodes,
                             round(mean_ret, 1), round(tiles, 1), round(laps, 3),
                             round(loss_m, 4), round(q_m, 3), round(al_m, 3)])
            csv_file.flush()
            print(f'paso {global_step:>8} | {minutes:6.1f} min | ret {mean_ret:7.1f} | '
                  f'pista {tiles:5.1f}% | vueltas {laps:4.2f} | loss {loss_m:.3f} Q {q_m:.2f}',
                  flush=True)
            torch.save({'encoder': critic.encoder.state_dict(), 'actor': actor.state_dict(),
                        'critic': critic.state_dict(), 'global_step': global_step,
                        'args': vars(args)},
                       os.path.join(args.out_dir, 'last.pt'))

    envs.close()
    csv_file.close()
    print('Entrenamiento terminado.')


if __name__ == '__main__':
    main()
