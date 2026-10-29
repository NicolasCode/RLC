'''
DQN "moderno" (Double DQN + red objetivo + replay buffer grande + pérdida Huber)
sobre EXACTAMENTE el mismo entorno que usa PPO (Utils/car_env.py), para
comparar los dos algoritmos en igualdad de condiciones.

Uso:
    python train_dqn.py --total-steps 300000

Guarda en --out-dir (por defecto models/dqn_racer/): last.pt, progress.csv
'''
import argparse
import csv
import os
import time
from collections import deque

import numpy as np
import torch
from torch import nn
import gymnasium as gym

from Agents.ppo_racer import layer_init
from Utils.car_env import make_env, ACTION_SETS


class QNetwork(nn.Module):
    '''Misma CNN que el actor-crítico de PPO, con una cabeza de valores Q.'''
    def __init__(self, n_actions, in_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8 * 8, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, n_actions), std=1.0),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)


class ReplayBuffer:
    def __init__(self, capacity, obs_shape):
        self.obs = np.zeros((capacity, *obs_shape), np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), np.uint8)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.capacity, self.pos, self.full = capacity, 0, False

    def add(self, o, no, a, r, d):
        n = len(a)
        idx = (self.pos + np.arange(n)) % self.capacity
        self.obs[idx], self.next_obs[idx] = o, no
        self.actions[idx], self.rewards[idx], self.dones[idx] = a, r, d
        self.pos = (self.pos + n) % self.capacity
        self.full = self.full or self.pos < n

    def sample(self, batch):
        high = self.capacity if self.full else self.pos
        idx = np.random.randint(0, high, batch)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.actions[idx]), torch.from_numpy(self.rewards[idx]),
                torch.from_numpy(self.dones[idx]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--total-steps', type=int, default=300_000, help='decisiones del agente en total')
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--buffer-size', type=int, default=50_000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--learning-starts', type=int, default=5_000)
    p.add_argument('--target-update', type=int, default=1_000, help='pasos de gradiente entre copias de la red objetivo')
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--eps-fraction', type=float, default=0.2, help='fracción del entrenamiento en la que baja epsilon')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out-dir', default=os.path.join('models', 'dqn_racer'))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    envs = gym.vector.AsyncVectorEnv(
        [make_env(seed=args.seed + i) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    n_actions = len(ACTION_SETS['custom'][0])
    q = QNetwork(n_actions)
    q_target = QNetwork(n_actions)
    q_target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=args.lr)
    rb = ReplayBuffer(args.buffer_size, (4, 96, 96))

    csv_file = open(os.path.join(args.out_dir, 'progress.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'frames', 'minutes', 'episodes', 'return', 'tiles_pct',
                     'lap_rate', 'loss', 'q_mean', 'epsilon'])

    ep_returns, ep_tiles, ep_laps = deque(maxlen=50), deque(maxlen=50), deque(maxlen=50)
    losses, qs = deque(maxlen=200), deque(maxlen=200)
    episodes, grad_steps, global_step = 0, 0, 0
    obs, _ = envs.reset(seed=args.seed)
    t0 = time.time()
    next_log = 1024
    N = args.num_envs

    while global_step < args.total_steps:
        eps = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end)
                  * global_step / (args.eps_fraction * args.total_steps))
        torch.set_num_threads(1)
        with torch.no_grad():
            greedy = q(torch.from_numpy(obs)).argmax(-1).numpy()
        random_a = np.random.randint(0, n_actions, N)
        actions = np.where(np.random.rand(N) < eps, random_a, greedy)
        next_obs, reward, term, trunc, info = envs.step(actions)

        # Para guardar la transición necesitamos el último estado real (no el del reset)
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
            # Un paso de gradiente (batch 64) por cada 8 transiciones nuevas
            o, no, a, r, d = rb.sample(args.batch_size)
            with torch.no_grad():
                # Double DQN: la red online elige la acción, la objetivo la evalúa
                best = q(no).argmax(-1, keepdim=True)
                target = r + args.gamma * (1 - d) * q_target(no).gather(1, best).squeeze(1)
            pred = q(o).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            optimizer.step()
            grad_steps += 1
            losses.append(loss.item())
            qs.append(pred.mean().item())
            if grad_steps % args.target_update == 0:
                q_target.load_state_dict(q.state_dict())

        if global_step >= next_log:
            next_log += 1024
            minutes = (time.time() - t0) / 60
            mean_ret = float(np.mean(ep_returns)) if ep_returns else float('nan')
            tiles = float(np.mean(ep_tiles)) * 100 if ep_tiles else float('nan')
            laps = float(np.mean(ep_laps)) if ep_laps else float('nan')
            loss_m = float(np.mean(losses)) if losses else float('nan')
            q_m = float(np.mean(qs)) if qs else float('nan')
            writer.writerow([global_step, global_step * 4, round(minutes, 2), episodes,
                             round(mean_ret, 1), round(tiles, 1), round(laps, 3),
                             round(loss_m, 4), round(q_m, 3), round(eps, 3)])
            csv_file.flush()
            print(f'paso {global_step:>8} | {minutes:6.1f} min | ret {mean_ret:7.1f} | '
                  f'pista {tiles:5.1f}% | vueltas {laps:4.2f} | eps {eps:.2f} | '
                  f'loss {loss_m:.3f} Q {q_m:.2f}', flush=True)
            torch.save({'model': q.state_dict(), 'global_step': global_step, 'args': vars(args)},
                       os.path.join(args.out_dir, 'last.pt'))

    envs.close()
    csv_file.close()
    print('Entrenamiento terminado.')


if __name__ == '__main__':
    main()
