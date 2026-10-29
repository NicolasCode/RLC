# Cómo aprendió el carrito a dar la vuelta (PPO en CarRacing)

Este documento explica **todo lo que se hizo** para que el agente pasara de no
avanzar (≈5–8% de la pista) a **completar vueltas enteras** en CarRacing, cuánto
aporta cada cambio (medido, no supuesto), y por qué PPO lo logró donde el DQN
anterior no.

<!-- RESULTADOS -->

---

## 1. Cómo reproducirlo

```bash
pip install "gymnasium[box2d]" torch opencv-python-headless matplotlib pandas
```

```bash
python train_ppo.py --total-steps 1500000
```

```bash
python evaluate_ppo.py --model models/ppo_racer/ppo_racer_final.pt --seeds 18 --stochastic --plot --gif images/ppo_vuelta_seed_18.gif
```

```bash
python plot_progress.py
```

```bash
bash experiments/run_ablations.sh && python plot_experiments.py
```

| Archivo | Qué hace |
|---|---|
| `Utils/car_env.py` | Entorno CarRacing envuelto: acciones, frame skip, stack de frames, fin anticipado |
| `Agents/ppo_racer.py` | Red actor-crítico (CNN) y el algoritmo PPO |
| `train_ppo.py` | Entrenamiento con 8 simuladores en paralelo; guarda `models/ppo_racer/` |
| `evaluate_ppo.py` | Evalúa en pistas concretas, dibuja la trayectoria y graba GIFs |
| `plot_progress.py` | Curva de aprendizaje desde `progress.csv` |
| `train_dqn.py` | DQN moderno sobre el mismo entorno (para la comparación) |
| `experiments/run_ablations.sh` | Experimentos "quitar un cambio a la vez" + DQN |
| `plot_experiments.py` | Gráficas y tabla de esos experimentos |

Hardware usado: 4 CPUs, **sin GPU**. Todo corre en CPU.

---

## 2. Diagnóstico: por qué el código anterior no aprendía

Antes de entrenar nada se revisó el pipeline existente (`Examples/car_racing.py`,
`Utils/train.py`, `Agents/agentsCS.py`, `Agents/BaseAgent.py`,
`Utils/interpreters.py`). Los problemas no eran de "falta de entrenamiento":
había errores que impedían aprender por completo.

### 2.1 Problemas del entorno / preprocesamiento (afectaban a todos los agentes)

| Problema | Dónde | Efecto |
|---|---|---|
| `lap_complete_percent=0.07` | `Utils/train.py:52` | La "vuelta" se daba por completada con el 7% de la pista: la métrica de éxito no medía una vuelta real |
| `import Utils.track_creating` (archivo inexistente) | `Utils/train.py:8` | `main.py` y `TrainRun` fallan al importar |
| Acciones discretas originales | Gymnasium | El acelerador discreto es solo **0.2** y no se puede girar y acelerar a la vez |
| Imagen reducida a 32×32 | `Utils/interpreters.py:94` | Se pierde el detalle de los bordes de la pista |
| "Memoria" de movimiento con `0.01 * last_state` | `Utils/interpreters.py:100` | El frame anterior pesa 1%: prácticamente invisible, así que el agente no puede saber su velocidad ni su dirección |
| Sin frame skip | — | Una decisión por frame (50 por segundo). Las recompensas quedan muy lejos de las acciones que las causan |
| Sin fin anticipado | — | Si el carro se sale a la hierba, sigue 1000 frames sumando −0.1 sin aprender nada útil |

### 2.2 Problemas del DQN anterior (`Agents/agentsCS.py`, configuración en `Examples/car_racing.py`)

| Problema | Dónde | Efecto |
|---|---|---|
| Replay buffer de solo **1024** transiciones | `agentsCS.py:19` | ≈20 s de conducción; datos muy correlacionados |
| En **cada paso** recalcula los objetivos de las 1024 transiciones y entrena sobre todas (64 minibatches de 16) | `agentsCS.py:364`, `ExperienceDataset` | Sobreajuste a los últimos segundos y entrenamiento lentísimo (miles de forward por cada frame) |
| La red objetivo se copia cuando `len(actions) % c == 0` | `agentsCS.py:373` | Cuando el buffer se llena, `len` queda fijo en 1024 y 1024 % 32 == 0 → **se copia en cada paso**: en la práctica no hay red objetivo y el aprendizaje se vuelve inestable |
| El objetivo usa `make_decision(next_state)` (ε-greedy con ε=0.5) en vez del máximo | `agentsCS.py:301` | El objetivo de Q-learning se contamina con acciones aleatorias |
| `gamma = 1` | `car_racing.py:121` | Episodios de 1000 frames con −0.1 por frame: valores que no se acotan y mucha varianza |
| `epsilon = 0.5` fijo durante todo el entrenamiento | `car_racing.py:44` | La mitad de las acciones son aleatorias, incluso al final |

### 2.3 Problemas del PPO anterior (`Agents/BaseAgent.py`)

| Problema | Dónde | Efecto |
|---|---|---|
| Solo actualiza al final de cada episodio, con **300 épocas** sobre ese único episodio y minibatches de 8 | `car_racing.py:84-85` | Decenas de miles de pasos de gradiente sobre los mismos datos: la política colapsa |
| `alpha_policy = 0.01` | `car_racing.py:78` | Learning rate ~40× más alto de lo habitual para PPO |
| La pérdida de valor solo entrena la cabeza de valor, no la CNN compartida | `BaseAgent.py:83` | El crítico aprende sobre features pensadas solo para la política |
| `make_decision()` no recibe la observación y no existe `store_experience` | `BaseAgent.py:89` | Incompatible con `TrainRun` (que llama `make_decision(obs)` y `store_experience`) |
| Un solo entorno | — | Poca diversidad de experiencia por actualización |

---

## 3. Qué se cambió

### 3.1 Entorno (`Utils/car_env.py`)

1. **Gymnasium 1.x / `CarRacing-v3`** (Python 3.13). `CarRacing-v2` ya no existe; el entorno es el mismo.
2. **Vuelta real:** `lap_complete_percent = 0.95`, el valor oficial. Para que una vuelta cuente hay que pisar el 95% de las baldosas y volver a la meta.
3. **7 acciones discretas propias** sobre el modo continuo: nada, izquierda, derecha, **gas 1.0**, freno 0.8, izquierda+gas 0.5 y derecha+gas 0.5.
4. **Frame skip = 4:** cada decisión se repite 4 frames (12.5 decisiones por segundo). Además **solo se renderiza el último frame** (`FastCarRacing.skip_render`). Renderizar era ~75% del costo, así que la simulación va 3.7× más rápida.
5. **Stack de 4 frames** en escala de grises de 96×96: la red ve el movimiento (velocidad y dirección de giro).
6. **Se salta el zoom inicial:** los primeros 48 frames son una animación de la cámara.
7. **Fin anticipado:** si en **200 frames** (4 s) no pisa una baldosa nueva (está en la hierba, quieto o de reversa), el episodio termina con −20. El agente deja de perder tiempo en estados inútiles y aprende que salirse es malo.
8. **Límite de 2000 frames** por episodio. Cortar por tiempo no es un final real: se hace *bootstrap* con el valor del último estado.
9. **Recompensa × 0.1** para que los valores queden en un rango cómodo para la red.

### 3.2 Agente PPO (`Agents/ppo_racer.py`, `train_ppo.py`)

Receta estándar de PPO (Schulman et al. 2017; implementación tipo CleanRL):

| Hiperparámetro | Valor |
|---|---|
| Red | CNN "Nature DQN" (3 conv + 512) compartida por el actor (7 logits) y el crítico (1 valor), inicialización ortogonal |
| Entornos en paralelo | 8 (`AsyncVectorEnv`) |
| Rollout | 128 pasos × 8 entornos = 1024 transiciones por actualización |
| Épocas / minibatch | 4 épocas, minibatches de 256 |
| Learning rate | 2.5e-4 con decaimiento lineal hasta 0 |
| γ / λ (GAE) | 0.99 / 0.95 |
| Clip | 0.1 |
| Coef. entropía / valor | 0.01 / 0.5 |
| Grad clip | 0.5 |
| Presupuesto | 1.5M decisiones = 6M frames |

Diferencias clave con el PPO anterior: se actualiza cada 1024 pasos (no al final
de cada episodio), con 4 épocas (no 300), un lr 40× menor, ventajas
normalizadas, la pérdida de valor entrena toda la red y hay bonus de entropía
para que siga explorando.

<!-- ENTRENAMIENTO -->

<!-- ABLACIONES -->

<!-- DQN -->

<!-- LIMITACIONES -->
