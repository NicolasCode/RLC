#!/usr/bin/env bash
# Experimentos de ablación: se quita UN cambio a la vez respecto a la receta final
# y se entrena 100k decisiones con el mismo ritmo de learning rate que el run
# principal (--anneal-steps 1500000), para comparar contra sus primeras 100k decisiones.
# Al final entrena un DQN moderno en el mismo entorno para comparar con PPO.
#
# Uso (desde la raíz del repo):  bash experiments/run_ablations.sh
set -e
STEPS=${STEPS:-100000}
COMMON="--total-steps $STEPS --anneal-steps 1500000"
OUT=models/ablations

python3 train_ppo.py $COMMON --actions original --out-dir $OUT/acciones_originales
python3 train_ppo.py $COMMON --patience 0       --out-dir $OUT/sin_fin_anticipado
python3 train_ppo.py $COMMON --stack 1          --out-dir $OUT/sin_stack_de_frames
python3 train_ppo.py $COMMON --frame-skip 1     --out-dir $OUT/sin_frame_skip
python3 train_ppo.py $COMMON --lr 1e-2          --out-dir $OUT/lr_alto_0.01

python3 train_dqn.py --total-steps ${DQN_STEPS:-300000} --out-dir models/dqn_racer
