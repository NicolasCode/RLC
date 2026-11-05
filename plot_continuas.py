'''
Compara las curvas de aprendizaje con acciones discretas y continuas.

Uso:
    python plot_continuas.py

Genera images/discretas_vs_continuas.png con dos paneles:
    PPO discreto (7 acciones) vs PPO continuo (Beta)
    DQN (7 acciones) vs TD3 (continuo)
'''
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = [
    ('PPO', [('discreto (7 acciones)', 'models/ppo_racer/progress.csv', '#1f77b4'),
             ('continuo (Beta)', 'models/ppo_continuo/progress.csv', '#d62728')]),
    ('DQN / TD3', [('DQN discreto (7 acciones)', 'models/dqn_racer/progress.csv', '#1f77b4'),
                   ('TD3 continuo', 'models/td3_continuo/progress.csv', '#d62728')]),
]


def smooth(s, w=5):
    return s.rolling(w, min_periods=1).mean()


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, (title, runs) in enumerate(RUNS):
        for label, path, color in runs:
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            axes[0, col].plot(df['frames'] / 1e6, smooth(df['tiles_pct']), label=label, color=color)
            axes[1, col].plot(df['frames'] / 1e6, smooth(df['return']), label=label, color=color)
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel('% de la pista recorrida')
        axes[1, col].set_ylabel('Retorno del entorno')
        axes[1, col].set_xlabel('Frames de simulación (millones)')
        axes[0, col].legend()
        for ax in axes[:, col]:
            ax.grid(alpha=0.3)
    fig.suptitle('Acciones discretas vs. continuas (mismo entorno, misma CNN, mismo presupuesto)')
    fig.tight_layout()
    fig.savefig(os.path.join('images', 'discretas_vs_continuas.png'), dpi=110)
    print('Guardado images/discretas_vs_continuas.png')


if __name__ == '__main__':
    main()
