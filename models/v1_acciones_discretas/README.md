# v1 · Modelos con acciones discretas

Primera versión de los agentes que aprendieron a dar la vuelta en CarRacing
desde píxeles. Las 7 acciones son **discretas** (nada, izquierda, derecha,
gas, freno, izquierda+gas, derecha+gas). Cómo se llegó a estos modelos:
[`docs/aprendizaje_ppo.md`](../../docs/aprendizaje_ppo.md). Comparación con
la literatura: [`docs/estado_del_arte.md`](../../docs/estado_del_arte.md).

## Archivos

| Archivo | Algoritmo | Entrenamiento | SHA-256 |
|---|---|---|---|
| `ppo_discreto_final.pt` | PPO | 1 500 160 decisiones (6.0M frames), 2 h 53 min | `c065525a998d256c6b945f3d6500cadf783595f73e10a1fae14d7ee2e06d0e8f` |
| `ppo_discreto_primera_vuelta.pt` | PPO (el mismo run) | 390 144 decisiones (1.56M frames), ~43 min | `5d006ec09132b515d772f6800aa0b55034e79f9d85169f6cda81fb2c0afcb486` |
| `dqn_discreto_300k.pt` | Double DQN | 299 008 decisiones (1.2M frames), ~90 min | `21c5b0fa75a9a0a4f722da2ae4d05e42c5d809c24f9c30f94c26d4e9550efa4b` |

Cada archivo es un diccionario de PyTorch con estas claves:
* `model`: los pesos (`state_dict`).
* `global_step`: decisiones entrenadas.
* `args`: los argumentos del entrenamiento.

Los dos PPO se entrenaron antes de que existieran las opciones `--actions`,
`--frame-skip` y `--stack`. Por eso su `args` no las incluye y
`evaluate_ppo.py` aplica los valores por defecto, que son los usados.

## Resultados

**Protocolo estándar de la literatura:** 100 pistas nunca vistas (semillas
2000–2099), 1000 frames y puntaje bruto del entorno. "Resuelto" significa un
promedio ≥ 900. Registros en `experiments/eval_estandar_*.txt`.

| Modelo | Puntaje | Episodios ≥ 900 | Vuelta (regla 95%) |
|---|---|---|---|
| `ppo_discreto_final.pt`, acción más probable | **864.7 ± 73.5** | 27/100 | 84/100 |
| `ppo_discreto_final.pt`, acciones muestreadas | 833.2 ± 116.9 | 22/100 | 68/100 |
| `dqn_discreto_300k.pt`, acción de mayor Q | 846.3 ± 71.7 | 10/100 | 58/100 |

**Monza y 20 pistas nuevas**, con hasta 60 s por intento. Registros en
`experiments/evaluacion_final.txt` y `experiments/evaluacion_dqn.txt`.

| Modelo | Vuelta oficial (100% de baldosas) | Regla 95% | Tiempo mediano |
|---|---|---|---|
| `ppo_discreto_primera_vuelta.pt` | 4/21 | 14/21 | 21.3 s |
| `ppo_discreto_final.pt` | 14/21 | 19/21 | 16.4 s |
| `dqn_discreto_300k.pt` | 9/21 | 21/21 | 28.5 s |

Los registros de evaluación se generaron cuando los archivos estaban en
`models/ppo_racer/` y `models/dqn_racer/`, así que muestran esas rutas.

## Configuración

| | |
|---|---|
| Entorno | Gymnasium 1.4.0, `CarRacing-v3`, envuelto por `Utils/car_env.py` (`RacingEnv`) |
| Observación | 4 frames en escala de grises de 96×96 (uint8) |
| Acciones | 7 discretas (`ACTIONS` en `Utils/car_env.py`) |
| Frame skip | 4 (solo se renderiza el último frame) |
| Entrenamiento | Fin anticipado tras 200 frames sin baldosa nueva (−20), recompensa × 0.1, límite de 2000 frames |
| Red | CNN "Nature DQN": conv 8×8/4 (32), 4×4/2 (64), 3×3/1 (64), lineal 512; ~2.18M parámetros |
| PPO | 8 entornos, rollout 128, 4 épocas, minibatch 256, lr 2.5e-4 → 0, γ 0.99, λ 0.95, clip 0.1, entropía 0.01, semilla 1 |
| DQN | Double DQN, replay 50k, batch 64 cada 8 decisiones, lr 1e-4, red objetivo cada 1000 pasos, ε 1.0 → 0.05 en el 20% inicial, Huber, semilla 1 |
| Hardware | 4 CPUs, sin GPU (Python 3.13, PyTorch 2.15) |

## Uso

```bash
pip install -r requirements-ppo.txt
```

```bash
python evaluate_ppo.py --model models/v1_acciones_discretas/ppo_discreto_final.pt --seeds 18 --human
```

```bash
python evaluate_ppo.py --model models/v1_acciones_discretas/ppo_discreto_final.pt --seeds $(seq 2000 2099) --max-frames 952
```

El segundo comando reproduce el protocolo estándar. Se usan 952 frames
porque el entorno salta los 48 frames de zoom inicial; hay que restar 4.8
puntos al retorno medio para compararlo con la literatura.

Para cargarlo en código:

```python
import torch
from Agents.ppo_racer import ActorCritic

ck = torch.load('models/v1_acciones_discretas/ppo_discreto_final.pt', weights_only=False)
net = ActorCritic(n_actions=7)
net.load_state_dict(ck['model'])
net.eval()
# obs: tensor uint8 (1, 4, 96, 96) que da Utils.car_env.RacingEnv
# accion, _, _, _ = net.act(obs, greedy=True)
```

## Limitaciones conocidas

* No llega al umbral de "resuelto" (900): falla en algunas curvas cerradas,
  con un mínimo de 408 puntos en las 100 pistas.
* Cada modelo sale de un único entrenamiento con semilla 1.
* La vuelta "oficial" de Gymnasium 1.x exige el 100% de las baldosas, porque
  la regla del 95% nunca se activa. Ver `RacingEnv._on_finish_line_after_95`.
