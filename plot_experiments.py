'''
Gráficas de los experimentos de experiments/run_ablations.sh.

Uso:
    python plot_experiments.py

Genera:
    images/PPO_ablaciones.png   receta completa vs. quitar un cambio a la vez
    images/PPO_vs_DQN.png       PPO vs. DQN moderno en el mismo entorno
e imprime una tabla con el resultado de cada experimento.
'''
import os
import glob

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAIN = os.path.join('models', 'ppo_racer', 'progress.csv')
ABL = os.path.join('models', 'ablations')
DQN = os.path.join('models', 'dqn_racer', 'progress.csv')

NAMES = {
    'receta_completa': 'receta completa',
    'acciones_originales': 'acciones originales (5, gas 0.2)',
    'sin_fin_anticipado': 'sin fin anticipado',
    'sin_stack_de_frames': 'sin stack (1 frame)',
    'sin_frame_skip': 'sin frame skip',
    'lr_alto_0.01': 'lr = 0.01',
}


def smooth(s, w=5):
    return s.rolling(w, min_periods=1).mean()


def summary(df, label):
    tail = df.tail(10)
    return {'experimento': label,
            'decisiones': int(df['step'].iloc[-1]),
            'frames': int(df['frames'].iloc[-1]),
            'minutos': round(float(df['minutes'].iloc[-1]), 1),
            'retorno': round(float(tail['return'].mean()), 1),
            'pista_%': round(float(tail['tiles_pct'].mean()), 1),
            'vueltas_%': round(100 * float(tail['lap_rate'].mean()), 1)}


def main():
    main_df = pd.read_csv(MAIN)
    rows = []
    runs = sorted(glob.glob(os.path.join(ABL, '*', 'progress.csv')))
    if runs:
        limit = max(pd.read_csv(r)['step'].iloc[-1] for r in runs)
        base = main_df[main_df['step'] <= limit]
        # (nombre, dataframe) — la receta completa con semilla 1 son las primeras
        # decisiones del run principal (mismo ritmo de learning rate)
        curves = [('receta_completa', base)]
        curves += [(os.path.basename(os.path.dirname(r)), pd.read_csv(r)) for r in runs]
        colors = {'receta_completa': 'black', 'acciones_originales': '#1f77b4',
                  'lr_alto_0.01': '#ff7f0e', 'sin_fin_anticipado': '#2ca02c',
                  'sin_frame_skip': '#d62728', 'sin_stack_de_frames': '#9467bd'}
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for name, df in curves:
            key = name[:-3] if name.endswith('_s2') else name
            label = NAMES.get(key, key) + (' (semilla 2)' if name.endswith('_s2') else '')
            rows.append(summary(df, label))
            style = dict(color=colors.get(key), lw=2.5 if key == 'receta_completa' else 1.5,
                         ls='--' if name.endswith('_s2') else '-')
            axes[0].plot(df['step'] / 1e3, smooth(df['tiles_pct']), label=label, **style)
            axes[1].plot(df['frames'] / 1e6, smooth(df['tiles_pct']), label=label, **style)
            axes[2].plot(df['step'] / 1e3, smooth(df['return']), label=label, **style)
        axes[0].set_xlabel('Decisiones del agente (miles)')
        axes[1].set_xlabel('Frames de simulación (millones)')
        axes[2].set_xlabel('Decisiones del agente (miles)')
        axes[0].set_ylabel('% de la pista recorrida')
        axes[1].set_ylabel('% de la pista recorrida')
        axes[2].set_ylabel('Retorno del entorno')
        for ax in axes:
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle('¿Qué aporta cada cambio? Receta completa vs. quitar un cambio a la vez (100k decisiones)')
        fig.tight_layout()
        fig.savefig(os.path.join('images', 'PPO_ablaciones.png'), dpi=110)
        print('Guardado images/PPO_ablaciones.png')

    if os.path.exists(DQN):
        dqn = pd.read_csv(DQN)
        limit = dqn['step'].iloc[-1]
        ppo = main_df[main_df['step'] <= limit]
        rows.append(summary(ppo, f'PPO (primeras {limit // 1000}k decisiones)'))
        rows.append(summary(dqn, 'DQN moderno'))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for df, label, c in [(ppo, 'PPO', '#1f77b4'), (dqn, 'DQN (Double DQN)', '#ff7f0e')]:
            axes[0].plot(df['step'] / 1e3, smooth(df['tiles_pct']), label=label, color=c)
            axes[1].plot(df['step'] / 1e3, smooth(100 * df['lap_rate']), label=label, color=c)
            axes[2].plot(df['step'] / 1e3, smooth(df['return']), label=label, color=c)
        axes[0].set_ylabel('% de la pista recorrida')
        axes[1].set_ylabel('% de episodios con vuelta completa')
        axes[2].set_ylabel('Retorno del entorno')
        for ax in axes:
            ax.set_xlabel('Decisiones del agente (miles)')
        for ax in axes:
            ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle('PPO vs. DQN en el mismo entorno (mismas acciones, frames y red)')
        fig.tight_layout()
        fig.savefig(os.path.join('images', 'PPO_vs_DQN.png'), dpi=110)
        print('Guardado images/PPO_vs_DQN.png')

    if rows:
        print()
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
