'''
Evalúa un agente PPO entrenado con train_ppo.py.

Uso:
    python evaluate_ppo.py                                # 10 pistas aleatorias + Monza (semilla 18)
    python evaluate_ppo.py --seeds 18 --plot              # dibuja la trayectoria en images/
    python evaluate_ppo.py --seeds 18 --gif images/x.gif  # graba un GIF de la vuelta
    python evaluate_ppo.py --human --seeds 18             # ver la vuelta en una ventana

Se reportan dos criterios de vuelta:
  * oficial: el entorno de Gymnasium la declara terminada (`lap_finished`). En
    Gymnasium 1.x esto solo ocurre al pisar el 100% de las baldosas (ver
    Utils/car_env.py, _on_finish_line_after_95).
  * 95%: la regla documentada por Gymnasium, volver a la baldosa de salida
    habiendo pisado al menos el 95% de la pista.
'''
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Agents.ppo_racer import ActorCritic
from Utils.car_env import RacingEnv, ACTION_SETS


def run_episode(net, env, seed, greedy, frames_out=None, gif_every=2):
    obs, _ = env.reset(seed=seed)
    car = env.env
    path = [tuple(car.car.hull.position)]
    actions = np.zeros(len(env.actions), int)
    done = False
    while not done:
        with torch.no_grad():
            a, _, _, _ = net.act(torch.from_numpy(obs).unsqueeze(0), greedy=greedy)
        a = int(a)
        actions[a] += 1
        obs, _, term, trunc, info = env.step(a)
        path.append(tuple(car.car.hull.position))
        if frames_out is not None and len(path) % gif_every == 0:
            frames_out.append(env.render())
        done = term or trunc
    info['path'] = np.array(path)
    info['track'] = np.array([(x, y) for _, _, x, y in car.track])
    info['actions'] = actions
    return info


def plot_path(info, seed, filename):
    track, path = info['track'], info['path']
    fig, ax = plt.subplots(figsize=(7, 7))
    closed = np.vstack([track, track[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color='#bbbbbb', lw=14, solid_capstyle='round', label='pista')
    ax.plot(path[:, 0], path[:, 1], color='#d62728', lw=1.8, label='carro (PPO)')
    ax.scatter(*track[0], color='black', zorder=5, s=40, label='salida/meta')
    if info['lap']:
        lap = f'VUELTA COMPLETA (100% de baldosas) en {info["frames"] / 50:.1f} s'
    elif info['lap95']:
        lap = f'VUELTA COMPLETA (regla 95%) en {info["lap95_frames"] / 50:.1f} s'
    else:
        lap = 'vuelta incompleta'
    ax.set_title(f'Semilla {seed}: {lap}\n{100 * info["tiles"] / info["track_len"]:.1f}% de la pista, '
                 f'episodio de {info["frames"] / 50:.1f} s, retorno {info["raw_return"]:.0f}')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(filename, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.path.join('models', 'ppo_racer', 'best.pt'))
    p.add_argument('--seeds', type=int, nargs='+', default=[18] + list(range(1000, 1010)))
    p.add_argument('--stochastic', action='store_true', help='muestrea acciones en vez de usar la más probable')
    p.add_argument('--max-frames', type=int, default=3000)
    p.add_argument('--plot', action='store_true', help='guarda la trayectoria en images/ppo_lap_seed_<s>.png')
    p.add_argument('--gif', default=None, help='guarda un GIF del primer episodio')
    p.add_argument('--human', action='store_true', help='muestra la carrera en una ventana')
    p.add_argument('--torch-seed', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.torch_seed)

    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    cfg = ck.get('args', {})
    env_kwargs = dict(actions=cfg.get('actions', 'custom'), frame_skip=cfg.get('frame_skip', 4),
                      stack=cfg.get('stack', 4))
    net = ActorCritic(len(ACTION_SETS[env_kwargs['actions']][0]), in_channels=env_kwargs['stack'])
    net.load_state_dict(ck['model'])
    net.eval()
    print(f'Modelo: {args.model} (entrenado {ck["global_step"]} decisiones = '
          f'{env_kwargs["frame_skip"] * ck["global_step"]} frames)')

    render_mode = 'human' if args.human else ('rgb_array' if args.gif else None)
    # En evaluación no hay fin anticipado: el carro tiene todo el tiempo (max-frames) para terminar.
    env = RacingEnv(render_mode=render_mode, patience=10**9, max_frames=args.max_frames, **env_kwargs)

    results = []
    for k, seed in enumerate(args.seeds):
        frames = [] if (args.gif and k == 0) else None
        info = run_episode(net, env, seed, greedy=not args.stochastic, frames_out=frames)
        pct = 100 * info['tiles'] / info['track_len']
        results.append((seed, info['lap'], pct, info['raw_return'], info['frames'], info['lap95']))
        dist = ' '.join(f'{n}:{c}' for n, c in zip(env.action_names, info['actions']))
        lap95 = f'SI ({info["lap95_frames"] / 50:4.1f} s)' if info['lap95'] else 'no        '
        print(f'semilla {seed:>5} | vuelta {"SI" if info["lap"] else "no"} | vuelta95 {lap95} | '
              f'pista {pct:5.1f}% | retorno {info["raw_return"]:6.1f} | tiempo {info["frames"] / 50:5.1f} s | {dist}')
        if args.plot:
            os.makedirs('images', exist_ok=True)
            plot_path(info, seed, os.path.join('images', f'ppo_lap_seed_{seed}.png'))
        if frames:
            from PIL import Image
            imgs = [Image.fromarray(f).resize((300, 200)) for f in frames]
            imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=80, loop=0, optimize=True)
            print(f'GIF guardado en {args.gif}')

    laps = np.mean([r[1] for r in results])
    laps95 = np.mean([r[5] for r in results])
    print(f'\nVueltas completadas: {int(laps * len(results))}/{len(results)} ({100 * laps:.0f}%) | '
          f'regla 95%: {int(round(laps95 * len(results)))}/{len(results)} ({100 * laps95:.0f}%) | '
          f'pista media {np.mean([r[2] for r in results]):.1f}% | '
          f'retorno medio {np.mean([r[3] for r in results]):.1f}')
    env.close()


if __name__ == '__main__':
    main()
