# RLC - Reinforcement Learning Car Racing

Machine learning methods to drive a car using reinforcement learning with robust seed consistency for reproducible results.

## 🎯 Características Principales

- **Control Manual**: Conduce manualmente usando las teclas de flecha con grabación de video opcional
- **Exploración de Seeds**: Herramienta para encontrar y evaluar diferentes configuraciones de pista
- **Consistencia Robusta**: Configuración de semillas que garantiza reproducibilidad perfecta
- **Grabación de Video**: Captura videos de intentos tanto manuales como de agentes
- **Entrenamiento de Agentes**: Framework completo para entrenar agentes de RL

## 🚀 Inicio Rápido

### Instalación
```bash
pip install gymnasium[box2d] pygame numpy opencv-python matplotlib seaborn pandas torch
```

### Control Manual con Grabación
```bash
python manual_control.py
```
**Controles:**
- ⬅️➡️⬆️⬇️: Dirección y aceleración/freno
- **V**: Iniciar/detener grabación de video
- **R**: Reiniciar pista
- **ESC**: Salir

### Exploración de Seeds
```bash
python seed_explorer.py
```

## 🎥 Grabación de Video

### Control Manual
- Presiona **V** durante el juego para iniciar/detener grabación
- Los videos se guardan automáticamente como `manual_YYYYMMDD_HHMMSS.mp4`
- Incluye overlay con información de recompensa y vueltas

### Agentes de RL
```python
from Utils.train import TrainRun

# Crear instancia de entrenamiento
trainer = TrainRun(env_name, state_interpreter, agent, model_name, num_rounds, num_episodes)

# Ejecutar con grabación de video
trainer.run(visual=True, record_video=True, video_filename="mi_agente.mp4")
```

## 📊 Semillas Recomendadas

### 🏆 Top 10 Circuitos Más Interesantes:
1. **Semilla 19** - Complejidad: 28.17 (la más desafiante)
2. **Semilla 7** - Complejidad: 28.15
3. **Semilla 2** - Complejidad: 24.50
4. **Semilla 40** - Complejidad: 23.18
5. **Semilla 44** - Complejidad: 22.68
6. **✅ Semilla 18** - Complejidad: 22.37 (actualmente en uso - tipo Monza)
7. **Semilla 21** - Complejidad: 22.09
8. **Semilla 22** - Complejidad: 21.98
9. **Semilla 28** - Complejidad: 21.84
10. **Semilla 17** - Complejidad: 21.60

### 🎯 Cambiar Semilla
Para usar una semilla diferente, modifica la línea en `manual_control.py`:
```python
SEED = 18  # Cambia por la semilla deseada
```

## 🔧 Configuración Robusta de Semillas

El proyecto implementa una configuración robusta que garantiza la consistencia:

```python
# Configuración múltiple de randomness
random.seed(seed)
np.random.seed(seed)
env.seed(seed)
env.unwrapped.seed(seed)

# Reset con semilla específica
env.reset(seed=seed)
```

### ✅ Beneficios:
- **Reproducibilidad 100%**: Mismos circuitos en cada ejecución
- **Comparación confiable**: Rankings precisos de complejidad
- **Debugging facilitado**: Errores reproducibles
- **Evaluación consistente**: Métricas comparables entre agentes

## 📁 Estructura del Proyecto

```
RLC/
├── Agents/                 # Implementaciones de agentes RL
│   ├── BaseAgent.py       # Clase base para agentes
│   ├── deepQ.py          # Deep Q-Network
│   ├── agentsCS.py       # PPO y otros algoritmos
│   └── networks.py       # Arquitecturas de redes neuronales
├── Utils/                 # Utilidades y herramientas
│   ├── interpreters.py   # Análisis de resultados
│   ├── train.py         # Scripts de entrenamiento
│   └── utils.py         # Funciones auxiliares
├── data/                 # Modelos entrenados
├── images/               # Visualizaciones y resultados
│   └── seeds/           # Imágenes de circuitos por semilla
├── manual_control.py    # Control manual con semilla robusta
├── seed_explorer.py     # Explorador de semillas
└── main.py             # Script principal
```

## 🎮 Uso Avanzado

### Entrenar un Agente
```bash
python main.py --agent PPO --episodes 1000
```

### Evaluar Rendimiento
```bash
python agente.py --model data/ppo_trained.pt --test
```

### Explorar Más Semillas
```bash
# Modifica el rango en seed_explorer.py
SEED_RANGE = range(51, 101)  # Explora semillas 51-100
```

## 📈 Resultados y Métricas

- **DQN**: 8% de progreso promedio en CarRacing-v2
- **PPO**: 8% de progreso promedio con mejor estabilidad
- **Consistencia**: 100% de reproducibilidad en generación de circuitos
- **Exploración**: 50 semillas analizadas con ranking de complejidad

## 🤝 Contribuciones

El proyecto está diseñado de forma modular para facilitar:
- Nuevos algoritmos de RL
- Diferentes entornos de conducción
- Métricas de evaluación personalizadas
- Configuraciones de circuitos específicas

---

**Nota**: Este proyecto utiliza Gymnasium (sucesor de OpenAI Gym) para el entorno CarRacing-v2 con configuración robusta de semillas para garantizar reproducibilidad completa.
