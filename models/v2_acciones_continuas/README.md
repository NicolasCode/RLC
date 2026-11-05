# v2 · Modelos con acciones continuas

El mismo procedimiento que [`v1_acciones_discretas`](../v1_acciones_discretas/README.md),
con el mismo entorno, la misma CNN y el mismo presupuesto. La diferencia es
que el agente elige una **acción continua de 2 dimensiones** en vez de una de
7 acciones fijas:

| Eje | Rango | Se convierte en |
|---|---|---|
| Dirección | [−1, 1] | Volante |
| Acelerador/freno | [−1, 1] | > 0: acelerador (hasta 1.0); < 0: freno (hasta 0.8) |

Juntar acelerador y freno en un solo eje impide acelerar y frenar a la vez
(`continuous_to_car` en `Utils/car_env.py`). Análisis completo:
[`docs/aprendizaje_ppo.md`, sección 8](../../docs/aprendizaje_ppo.md).

## Archivos

| Archivo | Algoritmo | Entrenamiento | SHA-256 |
|---|---|---|---|
| `ppo_continuo_final.pt` | PPO con política Beta | 1 500 160 decisiones (6.0M frames), 2 h 40 min | `b6a736f9b6be890e1ebb36fa7f489fc4772216e92a8a8f0b9c58d53d55621efa` |
| `td3_continuo_300k.pt` | TD3 (versión continua de DQN) | 299 008 decisiones (1.2M frames), 66 min | `0b3e4a7f1da4d11fced92c07765d99b7a6fcad94b699d0232acbf9769d070408` |

* `ppo_continuo_final.pt` tiene las claves `model` (pesos de
  `BetaActorCritic`, 2.18M parámetros), `global_step` y `args`.
* `td3_continuo_300k.pt` tiene las claves `encoder` (CNN), `actor`,
  `global_step` y `args`. Encoder más actor suman 2.31M parámetros. El crítico
  no se guarda porque no hace falta para conducir.

## Resultados

**Protocolo estándar:** 100 pistas nunca vistas (semillas 2000–2099), 1000
frames y puntaje bruto. "Resuelto" significa ≥ 900. Registros en
`experiments/eval_estandar_*continuo*.txt`.

| Modelo | Acción en evaluación | Puntaje | ≥ 900 | Vuelta (regla 95%) |
|---|---|---|---|---|
| `ppo_continuo_final.pt` | **muestreada de la Beta** | **847.8 ± 80.8** | 15/100 | 68/100 |
| `ppo_continuo_final.pt` | media de la Beta | 833.6 ± 87.3 | 4/100 | 61/100 |
| `ppo_continuo_final.pt` | moda de la Beta | 798.1 ± 141.9 | 5/100 | 56/100 |
| `td3_continuo_300k.pt` | salida del actor (determinista) | 814.4 ± 112.2 | 13/100 | 41/100 |
| *Referencia v1:* PPO discreto | acción más probable | 864.7 ± 73.5 | 27/100 | 84/100 |
| *Referencia v1:* DQN discreto | acción de mayor Q | 846.3 ± 71.7 | 10/100 | 58/100 |

**Monza y 20 pistas nuevas**, con hasta 60 s por intento:

| Modelo | Vuelta oficial (100%) | Regla 95% | Tiempo mediano |
|---|---|---|---|
| `ppo_continuo_final.pt` (media) | 12/21 | 17/21 | 16.8 s |
| `td3_continuo_300k.pt` | **18/21** | 19/21 | 23.4 s |

En resumen, las acciones continuas **aprenden más rápido** durante el
entrenamiento, pero **no** dan mejor puntaje final que las discretas en la
evaluación estándar.

## Configuración

Igual que v1: Gymnasium 1.4.0 `CarRacing-v3`, 4 frames en grises de 96×96,
frame skip 4, fin anticipado tras 200 frames sin progreso, recompensa × 0.1,
CNN "Nature DQN", 4 CPUs sin GPU, semilla 1. Cambios propios de cada
algoritmo:

| | PPO continuo | TD3 |
|---|---|---|
| Política | Beta(α, β) por eje, con α, β = softplus(·) + 1 | Actor determinista con tanh y ruido gaussiano de exploración (σ = 0.2) |
| Resto | Mismos hiperparámetros que el PPO discreto (lr 2.5e-4 → 0, clip 0.1, entropía 0.01, 4 épocas…) | Replay 50k, batch 64 cada 8 decisiones, lr 1e-4, γ 0.99, τ 0.005, actor cada 2 pasos, ruido objetivo 0.2 (recorte 0.5), dos críticos (Double Q), Huber. La CNN solo la entrena el crítico |

## Uso

```bash
python evaluate_ppo.py --model models/v2_acciones_continuas/ppo_continuo_final.pt --seeds 18 --stochastic --human
```

```bash
python evaluate_ppo.py --model models/v2_acciones_continuas/td3_continuo_300k.pt --seeds 18 --human
```

```bash
bash experiments/run_continuas.sh
```

El tercer comando reentrena ambos modelos.
