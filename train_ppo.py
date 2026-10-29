'''
Entrena un agente PPO para que el carro aprenda a dar la vuelta completa en CarRacing.

Uso:
    python train_ppo.py                       # entrenamiento completo
    python train_ppo.py --total-steps 50000   # prueba corta
    python train_ppo.py --resume              # continúa desde el último checkpoint

Guarda en --out-dir (por defecto models/ppo_racer/):
    last.pt, best.pt, progress.csv
'''
import argparse
import csv
import os
import time
from collections import deque

import numpy as np
import torch
import gymnasium as gym

from Agents.ppo_racer import ActorCritic, PPO
from Utils.car_env import make_env, ACTIONS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--total-steps', type=int, default=1_000_000,
                   help='decisiones del agente en total (cada una = 4 frames)')
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--num-steps', type=int, default=128, help='pasos por entorno en cada rollout')
    p.add_argument('--lr', type=float, default=2.5e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--minibatch-size', type=int, default=256)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out-dir', default=os.path.join('models', 'ppo_racer'))
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    envs = gym.vector.AsyncVectorEnv(
        [make_env(seed=args.seed + i) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    net = ActorCritic(len(ACTIONS))
    ppo = PPO(net, lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
              clip=args.clip, epochs=args.epochs,
              minibatch_size=args.minibatch_size, ent_coef=args.ent_coef)

    global_step, best = 0, -np.inf
    last_path = os.path.join(args.out_dir, 'last.pt')
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, weights_only=False)
        net.load_state_dict(ck['model'])
        ppo.optimizer.load_state_dict(ck['optimizer'])
        global_step, best = ck['global_step'], ck['best']
        print(f'Reanudando desde paso {global_step}')

    csv_path = os.path.join(args.out_dir, 'progress.csv')
    new_csv = not (args.resume and os.path.exists(csv_path))
    csv_file = open(csv_path, 'w' if new_csv else 'a', newline='')
    writer = csv.writer(csv_file)
    if new_csv:
        writer.writerow(['step', 'frames', 'minutes', 'episodes', 'return', 'tiles_pct',
                         'lap_rate', 'pg_loss', 'v_loss', 'entropy', 'approx_kl', 'lr'])

    T, N = args.num_steps, args.num_envs
    obs_buf = torch.zeros((T, N, 4, 96, 96), dtype=torch.uint8)
    act_buf = torch.zeros((T, N), dtype=torch.long)
    logp_buf = torch.zeros((T, N))
    rew_buf = torch.zeros((T, N))
    done_buf = torch.zeros((T, N))
    val_buf = torch.zeros((T, N))

    ep_returns, ep_tiles, ep_laps = deque(maxlen=50), deque(maxlen=50), deque(maxlen=50)
    episodes = 0
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.from_numpy(obs)
    t0 = time.time()
    start_step = global_step
    batch = T * N

    while global_step < args.total_steps:
        frac = 1.0 - global_step / args.total_steps
        ppo.set_lr(args.lr * frac)

        # ---------------- Rollout ----------------
        # Un solo hilo para la inferencia: los núcleos los usan los simuladores
        torch.set_num_threads(1)
        for t in range(T):
            with torch.no_grad():
                action, logp, _, value = net.act(obs)
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, action, logp, value
            next_obs, reward, term, trunc, info = envs.step(action.numpy())
            reward = torch.as_tensor(reward, dtype=torch.float32)

            # Si se cortó por tiempo (no es un final real) usamos el valor del
            # último estado para no castigar al agente por el límite de tiempo.
            if trunc.any():
                for i in np.where(trunc & ~term)[0]:
                    final = torch.from_numpy(info['final_obs'][i]).unsqueeze(0)
                    with torch.no_grad():
                        reward[i] += args.gamma * net.value(final)[0]

            rew_buf[t] = reward
            done_buf[t] = torch.as_tensor(term | trunc, dtype=torch.float32)
            obs = torch.from_numpy(next_obs)
            global_step += N

            if 'final_info' in info:
                fi = info['final_info']
                for i in np.where(term | trunc)[0]:
                    episodes += 1
                    ep_returns.append(fi['raw_return'][i])
                    ep_tiles.append(fi['tiles'][i] / fi['track_len'][i])
                    ep_laps.append(float(fi['lap'][i]))

        # ---------------- Aprendizaje ----------------
        torch.set_num_threads(args.threads)
        with torch.no_grad():
            next_value = net.value(obs)
        adv, ret = ppo.compute_gae(rew_buf, val_buf, done_buf, next_value)
        stats = ppo.update(obs_buf.reshape(batch, 4, 96, 96), act_buf.reshape(batch),
                           logp_buf.reshape(batch), adv.reshape(batch), ret.reshape(batch))

        minutes = (time.time() - t0) / 60
        mean_ret = float(np.mean(ep_returns)) if ep_returns else float('nan')
        tiles = float(np.mean(ep_tiles)) * 100 if ep_tiles else float('nan')
        laps = float(np.mean(ep_laps)) if ep_laps else float('nan')
        writer.writerow([global_step, global_step * 4, round(minutes, 2), episodes,
                         round(mean_ret, 1), round(tiles, 1), round(laps, 3),
                         round(stats['pg_loss'], 4), round(stats['v_loss'], 4),
                         round(stats['entropy'], 3), round(stats['approx_kl'], 4),
                         args.lr * frac])
        csv_file.flush()
        sps = (global_step - start_step) / (time.time() - t0)
        print(f'paso {global_step:>8} | {minutes:6.1f} min | {sps:5.0f} dec/s | '
              f'ret {mean_ret:7.1f} | pista {tiles:5.1f}% | vueltas {laps:4.2f} | '
              f'ent {stats["entropy"]:.2f} kl {stats["approx_kl"]:.4f}', flush=True)

        ck = {'model': net.state_dict(), 'optimizer': ppo.optimizer.state_dict(),
              'global_step': global_step, 'best': best, 'args': vars(args)}
        if len(ep_returns) >= 20 and mean_ret > best:
            best = mean_ret
            ck['best'] = best
            torch.save(ck, os.path.join(args.out_dir, 'best.pt'))
        torch.save(ck, last_path)

    envs.close()
    csv_file.close()
    print('Entrenamiento terminado.')


if __name__ == '__main__':
    main()
