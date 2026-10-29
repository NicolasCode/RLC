'''
Dibuja la curva de aprendizaje a partir de models/ppo_racer/progress.csv.

Uso:
    python plot_progress.py [--csv models/ppo_racer/progress.csv] [--out images/PPO_racer_learning.png]
'''
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=os.path.join('models', 'ppo_racer', 'progress.csv'))
    p.add_argument('--out', default=os.path.join('images', 'PPO_racer_learning.png'))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    x = df['frames'] / 1e6
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(x, df['return'], color='#1f77b4')
    axes[0].axhline(900, ls='--', color='gray', lw=1)
    axes[0].set_ylabel('Retorno del entorno\n(media 50 episodios)')
    axes[1].plot(x, df['tiles_pct'], color='#2ca02c')
    axes[1].axhline(95, ls='--', color='gray', lw=1)
    axes[1].set_ylabel('% de la pista recorrida')
    axes[2].plot(x, 100 * df['lap_rate'], color='#d62728')
    axes[2].set_ylabel('% de episodios con\nvuelta completa')
    axes[2].set_xlabel('Frames de simulación (millones)')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle('PPO en CarRacing: aprendizaje durante el entrenamiento')
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f'Guardado en {args.out}')


if __name__ == '__main__':
    main()
