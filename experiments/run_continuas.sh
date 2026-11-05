#!/usr/bin/env bash
# Mismo procedimiento que los modelos discretos, pero con acciones continuas
# (dirección, acelerador/freno):
#   * PPO con política Beta, 1.5M decisiones (igual que models/v1_acciones_discretas PPO)
#   * TD3 (la versión continua de DQN), 300k decisiones (igual que el DQN)
# Uso (desde la raíz del repo):  bash experiments/run_continuas.sh
set -e
THREADS=${THREADS:-2}

if [ ! -f models/ppo_continuo/.terminado ]; then
    python3 train_ppo.py --actions continuous --total-steps ${PPO_STEPS:-1500000} \
        --threads $THREADS --out-dir models/ppo_continuo
    touch models/ppo_continuo/.terminado
fi
if [ ! -f models/td3_continuo/.terminado ]; then
    python3 train_td3.py --total-steps ${TD3_STEPS:-300000} --threads $THREADS --out-dir models/td3_continuo
    touch models/td3_continuo/.terminado
fi
