#!/usr/bin/env bash
# Experimentos de ablación: se quita UN cambio a la vez respecto a la receta final
# y se entrena 100k decisiones con el mismo ritmo de learning rate que el run
# principal (--anneal-steps 1500000), para comparar contra sus primeras 100k decisiones.
# Al final entrena un DQN moderno en el mismo entorno para comparar con PPO.
#
# Uso (desde la raíz del repo):  bash experiments/run_ablations.sh
# Los experimentos ya terminados (con el archivo .terminado) se saltan.
set -e
STEPS=${STEPS:-100000}
THREADS=${THREADS:-2}   # con 4 hilos la VM usada se volvía más lenta (ver docs/aprendizaje_ppo.md)
COMMON="--total-steps $STEPS --anneal-steps 1500000 --threads $THREADS"
OUT=models/ablations

run() {
    local dir=$1; shift
    if [ -f "$dir/.terminado" ]; then echo "Saltando $dir (ya terminado)"; return; fi
    "$@" --out-dir "$dir"
    touch "$dir/.terminado"
}

run $OUT/acciones_originales python3 train_ppo.py $COMMON --actions original
run $OUT/sin_fin_anticipado  python3 train_ppo.py $COMMON --patience 0
run $OUT/sin_stack_de_frames python3 train_ppo.py $COMMON --stack 1
run $OUT/sin_frame_skip      python3 train_ppo.py $COMMON --frame-skip 1
run $OUT/lr_alto_0.01        python3 train_ppo.py $COMMON --lr 1e-2

run models/dqn_racer python3 train_dqn.py --total-steps ${DQN_STEPS:-300000} --threads $THREADS
