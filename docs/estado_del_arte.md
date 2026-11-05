# Aprendizaje por refuerzo con visión en CarRacing: revisión de la literatura y comparación con un agente PPO entrenado en CPU

**Proyecto RLC** · Noviembre de 2026

---

## Resumen

CarRacing (OpenAI Gym, hoy Gymnasium) es el entorno de referencia más usado
para controlar un vehículo directamente desde píxeles. Esta revisión reúne
**20 trabajos** (artículos, informes de curso y repositorios públicos) que
abordan el entorno con redes neuronales, en su mayoría convolucionales (CNN),
y los compara con el agente desarrollado en este proyecto: PPO con una CNN
tipo *Nature DQN*, 7 acciones discretas, *frame skip* de 4, pila de 4 frames y
fin anticipado de episodio.

Bajo el protocolo estándar (100 pistas nuevas, 1000 frames), nuestro agente
obtiene **864.7 ± 73.5** puntos, con **6M frames** de entrenamiento y
**2 h 53 min en 4 CPUs sin GPU**. Queda **por debajo del umbral de "resuelto"
(900)**, que superan World Models (906 ± 21), el agente de autoatención
(914 ± 15) y PPO con política Beta (913 ± 26). Queda por encima de los
resultados históricos de DQN (343) y A3C (591–652), de la entrada del
*leaderboard* de Gym citada en World Models (838 ± 11) y de DDPG (848).

Tres hallazgos propios aportan a la literatura:
1. En experimentos controlados, el **fin anticipado** del episodio es la
   modificación con mayor efecto. Casi todos los repositorios exitosos lo
   usan, pero ningún artículo revisado lo aísla.
2. Un **DQN moderno** en el mismo entorno aprende tan bien o mejor que PPO
   con el mismo número de decisiones. Esto contradice la lectura habitual de
   que "PPO supera a DQN" en CarRacing.
3. En Gymnasium 1.x, el criterio de vuelta al 95% **nunca se activa** por un
   detalle de implementación.

---

## 1. Introducción

Conducir a partir de píxeles combina percepción (reconocer la pista en una
imagen de 96×96) y control secuencial con recompensa retardada. CarRacing se
volvió popular porque es barato de simular, genera pistas aleatorias en cada
episodio (lo que obliga a generalizar) y tiene un umbral explícito de éxito.
Hay dos grandes familias de soluciones:

* **Aprendizaje por refuerzo (RL) por gradiente**, extremo a extremo: DQN,
  A3C, DDPG o PPO entrenan una CNN que va de los píxeles a la acción.
* **Modelos del mundo y neuroevolución:** se aprende o fija una
  representación visual compacta y se optimiza un controlador pequeño con
  estrategias evolutivas (CMA-ES o algoritmos genéticos).

Esta revisión resume cada trabajo y luego compara su diseño y sus resultados
con los de nuestro agente (descrito en `docs/aprendizaje_ppo.md`).

## 2. El entorno CarRacing

* **Observación:** imagen RGB de 96×96 vista desde arriba. Incluye una barra
  inferior con indicadores de velocidad, cuatro sensores ABS, posición del
  volante y giroscopio [21].
* **Acciones:** continuas (dirección ∈ [−1, 1], acelerador y freno ∈ [0, 1]).
  Hay una variante discreta de 5 acciones (nada, izquierda, derecha,
  acelerador 0.2 y freno 0.8).
* **Recompensa:** −0.1 por frame y +1000/N por cada baldosa nueva visitada
  (N = número de baldosas). Salirse del área de juego da −100 y termina el
  episodio.
* **Criterio de "resuelto":** retorno medio ≥ 900 en 100 ensayos
  consecutivos [1, 3, 8]. Con el límite de 1000 frames, 900 puntos exigen
  recorrer prácticamente toda la pista en menos de ~1000 frames (20 s).
* **Versiones:** v0 (Gym clásico) → v1/v2 (Gym 0.2x y Gymnasium 0.x) → v3
  (Gymnasium 1.x). Casi toda la literatura usa v0; este proyecto usa v3. La
  dinámica y la recompensa son esencialmente las mismas.
* **Detalle encontrado en este proyecto:** en Gymnasium 1.x el chequeo de
  `lap_complete_percent` (95%) está dentro de `if not tile.road_visited`, y la
  baldosa de salida ya se visita al arrancar. Por eso el entorno solo declara
  la vuelta terminada al visitar el **100%** de las baldosas. No afecta al
  puntaje, pero sí a cualquier métrica de "vueltas completadas".

## 3. Metodología de la revisión

Se buscaron en la web artículos (arXiv, NeurIPS, GECCO, IWAI), informes de
curso (Stanford CS221/AA228) y repositorios públicos sobre CarRacing con redes
neuronales. Por restricciones de red del entorno de trabajo **no fue posible
descargar el texto completo de arXiv**. Los datos de esos artículos provienen
de sus resúmenes, de fragmentos indexados por el buscador y de las tablas
que otros trabajos citan. Los repositorios de GitHub sí se leyeron
directamente, incluido su código.

Cada dato lleva su **nivel de verificación**:

| Marca | Significado |
|---|---|
| **[C]** | Leído en el código o README del autor |
| **[A]** | Tomado del resumen o de fragmentos del propio artículo |
| **[T]** | Citado por un tercero (por ejemplo, la tabla de World Models) |

## 4. Trabajos revisados

### 4.1 Modelos del mundo y neuroevolución

**[1] Ha & Schmidhuber (2018), *World Models*.** Primer trabajo en reportar
CarRacing resuelto. Tiene tres partes: (V) un autoencoder variacional
convolucional que comprime cada frame en un vector latente z ∈ ℝ³² [A];
(M) una MDN-RNN (LSTM de 256 unidades) que predice el siguiente z [A]; y
(C) un controlador lineal de solo **867 parámetros** optimizado con CMA-ES
[A]. V y M se entrenan con 10 000 *rollouts* de una política aleatoria [A].
Obtiene **906 ± 21** en 100 ensayos [A]. Sus ablaciones muestran el aporte
de la memoria: solo V da 632 ± 251 y V con una capa oculta, 788 ± 141 [A].
Su tabla de comparación es la referencia que citan casi todos los trabajos
posteriores: DQN 343 ± 18 [13], A3C continuo 591 ± 45 [10], A3C discreto
652 ± 10 [9] y la entrada del *leaderboard* de Gym 838 ± 11 [A].

**[2] Risi & Stanley (2019), *Deep Neuroevolution of Recurrent and Discrete
World Models*.** Entrena la misma arquitectura de World Models
**extremo a extremo con un algoritmo genético**, sin gradientes y sin cruce
ni especiación [A]. Optimiza unos 4.7 millones de parámetros (VAE de 4.3M y
MDN-RNN de 384k) [T]. Obtiene **903 ± 72**, comparable al original [A].
También prueba un VAE con salidas binarias (discreto) [A].

**[3] Tang, Nguyen & Ha (2020), *Neuroevolution of Self-Interpretable
Agents*.** **No usa CNN.** Divide la imagen en parches de 7×7 con paso 4
(529 parches) y un módulo de **autoatención** vota los 10 parches más
importantes. Sus coordenadas alimentan una LSTM de 16 unidades. El agente
tiene **menos de 4 000 parámetros** y se entrena con CMA-ES [A]. Obtiene
**914 ± 15** en 100 ensayos [A]. La atención se concentra en los bordes de la
pista y se adelanta a las curvas, y el agente generaliza mejor a variaciones
visuales del entorno [A]. Una reproducción pública en CarRacing-v2 obtuvo
903 ± 12 tras **4 días en una máquina de 32 CPUs** (población 256, 16
*rollouts* por individuo) [C, 3b].

**[4] Chang & Futagami (2019–2020), *Convolutional Reservoir Computing*.**
Usa una **CNN con pesos aleatorios fijos** y una red de reservorio (ESN) como
extractores de características. Solo entrena una matriz de salida con una
estrategia evolutiva [A]. Afirma un puntaje de estado del arte en
CarRacing-v0 [A]; no se pudo verificar el número exacto. Su interés es que
una CNN **sin entrenar** puede bastar como representación visual.

**[5] Tang & Ha (2021), *The Sensory Neuron as a Transformer*.** Agentes
invariantes a permutaciones. En CarRacing se barajan los parches de la
imagen y el agente debe seguir conduciendo [A]. Se cita porque prueba la
robustez de la representación, no por su puntaje.

### 4.2 Gradiente de política y actor-crítico con CNN

**[6] Petrazzini & Antonelo (2021), *PPO with Continuous Bounded Action Space
via the Beta Distribution*.** Sustituye la política gaussiana de PPO por una
**distribución Beta**, cuyo soporte acotado coincide con el rango de las
acciones y evita el sesgo de recortar la gaussiana [A]. En CarRacing-v0, con
5 millones de pasos [A], obtiene **913 ± 26** con Beta, 897 ± 41 con la
gaussiana y 740 ± 86 con el PPO base [A]. Es el resultado de RL por
gradiente más alto encontrado.

**[7] xtma, *pytorch_car_caring* (repositorio).** Implementación de PPO con
política Beta muy reutilizada [C]. Escala de grises, pila de 4 frames
(4×96×96), **acción repetida 8 frames** y una CNN de 6 capas convolucionales
(8→256 filtros) con cabezas de actor y crítico. Usa lr 1e-3, clip 0.1, 10
épocas y ventaja TD(0) sin GAE [C]. Trucos de recompensa: **fin anticipado**
si la media móvil de 100 frames de recompensa es ≤ −0.1, **penalización por
hierba** (−0.05 si el canal verde domina) y no castiga la muerte [C].

**[8] Rafael1s (repositorio, basado en [7]).** PPO con Beta en CarRacing-v0.
Reporta un promedio de **901.81** en el episodio 2760, tras **6 h 53 min**
de entrenamiento [C]. Es una media móvil de 100 episodios *durante el
entrenamiento*, no una evaluación aparte con la política fija.

**[9] Khan & Elibol (2017), A3C con acciones discretas** (proyecto de
Stanford CS221). **652 ± 10** según [1]; 652.29 ± 10.17 en 100 episodios
según [22] [T].

**[10] Jang, Min & Lee (2017), *Reinforcement Car Racing with A3C*.** A3C con
acciones continuas y varias arquitecturas CNN [A]. [1] reporta **591 ± 45**;
el repositorio [22] cita 571.68 ± 19.38 para el mismo trabajo [T]. Esta
discrepancia entre fuentes secundarias no se pudo resolver.

**[11] wpiszlogin, *driver_critic* (repositorio).** **DDPG** con un
preprocesamiento manual agresivo: borra los números, agranda la barra de
velocidad, unifica los colores del pasto y la pista y pinta el carro de negro
[C]. CNN de 3 capas (16-32-32 filtros) para actor y crítico [C]. Corta el
episodio tras 200 pasos sin recompensa [C]. Reporta un promedio de
**848** (800–900 según la pista), con 6 h de entrenamiento y resultados
aceptables a los 15–30 min [C].

**[12] Yuwono, Yen & Christopher (2024), *Self-Driving Car Racing:
Application of Deep RL*.** Compara DQN y PPO en Gymnasium CarRacing, más
variantes con *transfer learning* (ResNet) y redes recurrentes (LSTM) [A].
Concluye que integrar ResNet y LSTM mejora al agente, que PPO alcanza mayor
retorno en menos tiempo que DQN y que PPO sufre **colapso de la política**
[A]. No se pudo verificar su tabla de resultados.

### 4.3 Métodos basados en valor (DQN y variantes)

**[13] Prieur (2017), DQN.** **343 ± 18** según [1] [T]. Es la cifra que se
cita como "DQN en CarRacing" en la literatura de modelos del mundo.

**[14] Li (2019), *Challenging On Car Racing Problem from OpenAI Gym*.**
Compara un **perceptrón multicapa evolutivo** (algoritmo genético) con una
**Double DQN** sobre píxeles [A]. El genético converge rápido pero se
estanca; la DDQN tiene mucha varianza y converge lento, pero con suficientes
episodios obtiene mejor puntaje [A]. Recomienda el genético con hardware
limitado [A].

**[15] Aldape & Sowell (2018), Stanford AA228.** DQN con varias arquitecturas,
incluida *transfer learning* con VGG16 [A]. Un tercero le atribuye un puntaje
de **150–200** [T].

**[16] Maniyar & Manoj (2019), Stanford AA228.** **DDQN** sobre RGB sin escala
de grises, con pila de 10 frames y acción repetida 10 frames [C]. Usa 27
acciones discretas (3 direcciones × 3 aceleradores × 3 frenos) y una CNN de 3
capas (64-128-128 filtros) más una capa densa de 1024 [C]. Entrena unas 21 500
decisiones (≈215 000 frames) [C]. El informe menciona puntajes cercanos a
1000 en algunos episodios [A]; el código evalúa solo 3 episodios [C].

**[17] andywu0913, *OpenAI-GYM-CarRacing-DQN* (repositorio).** DQN en Keras
muy difundido: escala de grises, pila de 3 frames, acción repetida 3 frames,
12 acciones discretas y una CNN pequeña (6 y 12 filtros) [C]. Termina el
episodio tras 25 recompensas negativas seguidas y multiplica ×1.5 la
recompensa con acelerador a fondo [C]. No reporta un puntaje numérico; el
autor elige el modelo a ojo [C].

**[18] Dahlin et al. (2020), *Designing Interpretable Approximations to Deep
RL*.** Entrena DQN en **pistas fijas** (elíptica, cuadrada y zigzag) y logra
la vuelta con ≥ 850 en cada una [A]. Luego aproxima la política con árboles
de decisión y máquinas de kernel que igualan su rendimiento [A]. No es
comparable en generalización: entrena y evalúa en la misma pista.

**[19] van Hoeffelen & Lanillos (2021), *Deep Active Inference for
Pixel-Based Discrete Control*.** Agente de **inferencia activa** profunda
(VAE convolucional más modelos de transición, política y energía libre
esperada) con 11 acciones discretas, imágenes de 42×42 y pila de 8 frames
[C]. Logra un desempeño comparable a su DQN de referencia, que obtiene
**515 ± 162** [A], aunque no llega al estado del arte [A].

### 4.4 Aprendizaje por imitación

**[20] Zhang & Sun (2020), *Deep RL with Mixed Convolutional Network*.**
Pese al título, es **aprendizaje supervisado por imitación** a partir de
partidas humanas grabadas y aumentadas ×4 [A]. Combina la imagen con los
sensores de la barra inferior [A]. Su repositorio reporta **594.9** (RGB) y
558.3 (grises) para su red, frente a 471.5 de AlexNet y 438.8 de VGG16; el
mejor modelo llega a 649.1 [C]. Muestra que las redes grandes preentrenadas
no ayudan en esta escala.

### 4.5 Otros recursos

* Modelos PPO en Hugging Face para CarRacing-v2 reportan puntajes
  autodeclarados de −63 a 883 [23]: sin un protocolo común no son
  comparables.
* Hay implementaciones de Dreamer para CarRacing sin evaluación publicada
  [24].

## 5. Nuestro trabajo (resumen)

Detalle completo en `docs/aprendizaje_ppo.md`.

| Aspecto | Nuestro agente |
|---|---|
| Algoritmo | PPO con GAE (λ = 0.95), clip 0.1, 4 épocas, lr 2.5e-4 con decaimiento lineal, 8 entornos en paralelo |
| Red | CNN *Nature DQN* (32-64-64 filtros + 512), compartida entre actor y crítico |
| Entrada | Escala de grises, 96×96, pila de 4 frames, barra de indicadores incluida |
| Acciones | 7 discretas sobre el modo continuo |
| Frame skip | 4, renderizando solo el último frame (simulación 3.7× más rápida) |
| Trucos | Fin anticipado (200 frames sin baldosa nueva, −20), recompensa × 0.1, *bootstrap* al cortar por tiempo |
| Presupuesto | 1.5M decisiones = **6M frames**, 2 h 53 min, 4 CPUs, sin GPU |
| Versión | Gymnasium 1.4, CarRacing-v3 |

**Evaluación estándar** (`experiments/eval_estandar_*.txt`): 100 pistas nunca
vistas (semillas 2000–2099), 1000 frames por episodio y puntaje bruto del
entorno, sin fin anticipado ni escalado de la recompensa. El *wrapper* salta
los 48 frames de zoom inicial, así que se corrió con 952 frames y se restaron
4.8 puntos (48 × 0.1) para igualar el protocolo.

| Política | Puntaje (100 pistas) | Episodios ≥ 900 | Vuelta con regla 95% |
|---|---|---|---|
| **PPO, acción más probable** | **864.7 ± 73.5** | 27/100 | 84/100 |
| PPO, acciones muestreadas | 833.2 ± 116.9 | 22/100 | 68/100 |
| DQN moderno (300k decisiones), acción de mayor Q | 846.3 ± 71.7 | 10/100 | 58/100 |

## 6. Comparativa

### 6.1 Resultados

Puntaje en CarRacing (umbral de resuelto: 900). Los protocolos difieren; ver
la columna "Evaluación".

| # | Trabajo | Método | Representación visual | Versión | Puntaje | Evaluación | Verif. |
|---|---|---|---|---|---|---|---|
| [3] | Tang et al. 2020 | CMA-ES + autoatención + LSTM | Parches + atención (sin CNN) | v0 | **914 ± 15** | 100 ensayos | A |
| [6] | Petrazzini & Antonelo 2021 | PPO, política Beta | CNN | v0 | **913 ± 26** | no verificado | A |
| [1] | Ha & Schmidhuber 2018 | VAE + MDN-RNN + CMA-ES | VAE convolucional | v0 | **906 ± 21** | 100 ensayos | A |
| [2] | Risi & Stanley 2019 | Algoritmo genético extremo a extremo | VAE convolucional evolucionado | v0 | **903 ± 72** | 100 ensayos | A |
| [8] | Rafael1s | PPO Beta | CNN 6 capas | v0 | 901.8 | media móvil en entrenamiento | C |
| [6] | Petrazzini & Antonelo 2021 | PPO, gaussiana | CNN | v0 | 897 ± 41 | no verificado | A |
| — | **Este trabajo** | **PPO, 7 acciones** | **CNN Nature (3 conv)** | **v3** | **864.7 ± 73.5** | **100 pistas nuevas** | — |
| [11] | wpiszlogin | DDPG | CNN 3 capas + preproc. manual | v0 | 848 | no indicado | C |
| — | **Este trabajo** | **DQN moderno (Double DQN)** | **CNN Nature** | **v3** | **846.3 ± 71.7** | **100 pistas nuevas** | — |
| [1] | Gym leaderboard (en [1]) | no indicado | no indicado | v0 | 838 ± 11 | no indicado | T |
| [6] | Petrazzini & Antonelo 2021 | PPO base | CNN | v0 | 740 ± 86 | no verificado | A |
| [9] | Khan & Elibol 2017 | A3C discreto | CNN | v0 | 652 ± 10 | 100 episodios | T |
| [20] | Zhang & Sun 2020 | Imitación (supervisado) | CNN + sensores | v0 | 595 (649 el mejor) | no indicado | C |
| [10] | Jang et al. 2017 | A3C continuo | CNN | v0 | 591 ± 45 (o 572 ± 19) | no indicado | T |
| [19] | van Hoeffelen & Lanillos 2021 | DQN (línea base) | CNN, 42×42 | v0 | 515 ± 162 | 100 episodios | A |
| [13] | Prieur 2017 | DQN | CNN | v0 | 343 ± 18 | no indicado | T |
| [15] | Aldape & Sowell 2018 | DQN | CNN / VGG16 | v0 | 150–200 | no indicado | T |
| — | Código anterior de este repositorio | DQN / PPO | CNN 32×32 | v2 | ~8% de la pista | entrenamiento | — |

### 6.2 Decisiones de diseño

| Trabajo | Entrada | Pila | Acción repetida | Acciones | Fin anticipado / penalización | Presupuesto |
|---|---|---|---|---|---|---|
| **Este trabajo (PPO)** | Grises 96×96 | 4 | 4 | 7 discretas | Sí: 200 frames sin progreso, −20 | 6M frames, 2.9 h, 4 CPUs |
| [7] xtma / [8] Rafael1s (PPO) | Grises 96×96 | 4 | 8 | Continuas (Beta) | Sí: media móvil ≤ −0.1; −0.05 en hierba | ~2760 episodios, 6.9 h [8] |
| [6] PPO Beta | no verificado | no verificado | no verificado | Continuas (Beta) | no verificado | 5M pasos |
| [11] DDPG | RGB con preproc. manual | 1 | 1 | Continuas (gas y freno en un eje) | Sí: 200 pasos sin recompensa | 6 h |
| [17] DQN Keras | Grises 96×96 | 3 | 3 | 12 discretas | Sí: 25 recompensas negativas seguidas; ×1.5 a fondo | ~500–1000 episodios |
| [16] DDQN AA228 | RGB 96×96 | 10 | 10 | 27 discretas | No | ~215k frames |
| [19] dAIF / DQN | Grises 42×42, recortada | 8 | 1 | 11 discretas | No | 1000 episodios |
| [3] Atención | RGB, parches 7×7 | 1 (LSTM) | 1 | Continuas | Sí en entrenamiento: 20 negativas seguidas [3b] | 4 días, 32 CPUs [3b] |
| [1] World Models | VAE, z de 32 | 1 (MDN-RNN) | 1 | Continuas | No | 10k *rollouts* + CMA-ES |
| [20] Imitación | Grises 96×96 + sensores | 1 | — | 7 clases | — | Datos humanos |

## 7. Discusión

**Dónde queda nuestro agente.** Con 864.7 ± 73.5, nuestro PPO es competitivo
con los mejores métodos de RL por gradiente que no llegan a 900 (DDPG 848,
*leaderboard* 838) y muy superior a las cifras históricas de DQN y A3C. Está
~40–50 puntos por debajo de los métodos que resuelven el entorno. El 27% de
nuestros episodios ya supera 900; la diferencia está en la cola de pistas
difíciles, donde el carro se sale en una curva cerrada (mínimo 408).

**Lo que hacen los mejores y nosotros no.**
1. **Acciones continuas con política Beta.** En [6], pasar del PPO base a
   PPO-Beta sube de 740 a 913, y [7]/[8] usan la misma idea. Nuestras 7
   acciones discretas limitan la finura del control: el carro solo puede girar
   a fondo o a medias. Es el candidato más claro para cerrar la brecha.
2. **Más presupuesto o más paralelismo.** La reproducción del agente de
   atención usó 32 CPUs durante 4 días (~3 000 CPU-hora); nosotros usamos unas
   12 CPU-hora. World Models y la neuroevolución evalúan miles de episodios
   completos.
3. **Memoria explícita.** World Models muestra que añadir la MDN-RNN sube de
   788 a 906. Nuestra red solo ve 4 frames.

**Lo que coincide.** La receta de nuestro PPO converge con la práctica de los
repositorios exitosos: escala de grises, pila de 4 frames, repetir la acción
(3–10 frames) y, sobre todo, **cortar el episodio cuando el carro deja de
progresar**. [7], [11], [17] y la reproducción de [3] lo hacen, cada uno con
su propia regla, pero ninguno de los artículos revisados mide su efecto.
Nuestras ablaciones lo hacen: sin fin anticipado el agente pasa ~60k
decisiones estancado en ~8% de la pista, y es el cambio con efecto más claro
(`docs/aprendizaje_ppo.md`, sección 5).

**Sobre PPO vs. DQN.** La literatura suele reportar DQN muy por debajo de PPO:
343 [13], 150–200 [15] y 515 [19] frente a 897–913 [6], y [12] concluye que
PPO supera a DQN. En nuestro experimento controlado (mismo entorno, misma red
y mismas acciones), un DQN moderno (Double DQN, *replay* de 50k, red objetivo
cada 1000 pasos, Huber) logró 846.3 ± 71.7 con solo 300k decisiones. PPO
logró 864.7 con 5× más decisiones. Nuestra hipótesis es que la mala fama de
DQN en CarRacing viene sobre todo de **implementaciones y entornos poco
cuidados**, como le pasó a nuestro propio código anterior (red objetivo
copiada en cada paso, γ = 1, ε = 0.5 fijo), y no del algoritmo. Coincide con
[14]: la DDQN acaba superando al genético si se entrena lo suficiente.

**Sobre la representación visual.** Varios trabajos sugieren que la CNN no
necesita ser grande ni estar muy entrenada:
* [4] usa una CNN aleatoria fija.
* [3] reemplaza la CNN por atención sobre parches con menos de 4k parámetros.
* En [20], VGG16 y AlexNet rinden peor que una CNN pequeña.

En nuestras ablaciones, apilar 4 frames no cambia el resultado al principio,
porque la barra de indicadores ya trae la velocidad. [20] lo aprovecha de
forma explícita al leer los sensores.

**Heterogeneidad de la evaluación.** Las cifras de la tabla 6.1 no son
estrictamente comparables:
* **Lo que se promedia:** unos promedian 100 ensayos con la política fija
  [1, 3]; otros reportan una media móvil durante el entrenamiento [8] o solo
  3 episodios [16].
* **Versión del entorno:** casi todos usan v0, que ya no está disponible; aquí
  se usa v3.
* **Escala de los RL por gradiente:** casi ningún trabajo con RL por gradiente
  reporta varias semillas. Nuestro ruido entre semillas es de ~17 puntos
  porcentuales de pista recorrida a las 100k decisiones.

Por eso se repitió el protocolo de 100 pistas, para que nuestra cifra sí sea
comparable con la de [1], [2] y [3].

## 8. Limitaciones de esta revisión

* **Texto completo de arXiv no consultado:** varias cifras (marcadas [A] y
  [T]) vienen de resúmenes o de tablas de terceros. Los detalles de
  preprocesamiento de [6], [12] y [14] no se pudieron verificar.
* **Fuentes que discrepan:** las dos fuentes de [10] dan cifras distintas
  (591 vs. 572).
* **Búsqueda no exhaustiva:** pueden faltar trabajos, sobre todo informes de
  curso y artículos recientes sobre CarRacing-v2/v3.
* **Una sola semilla:** nuestra cifra de 864.7 corresponde a un único
  entrenamiento.

## 9. Conclusiones y trabajo futuro

Nuestro agente PPO aprende a dar la vuelta desde píxeles con un presupuesto
modesto (6M frames, CPU) y se ubica en la franja alta de los métodos de RL
por gradiente (~850–870). No supera el umbral de 900 que alcanzan los modelos
del mundo, la neuroevolución con atención y PPO-Beta. A partir de la
literatura, los pasos con más probabilidad de cerrar la brecha son:

1. **Acciones continuas con política Beta** [6, 7].
2. **Más entrenamiento** (≥ 5M pasos de agente, como [6]) y **3–5 semillas**.
3. **Memoria** (LSTM o una pila más larga) para anticipar curvas [1].
4. Mantener el **protocolo estándar de 100 pistas** para comparar.

Como aportes propios a la literatura revisada quedan: medir el efecto aislado
del fin anticipado, comparar PPO y DQN en igualdad de condiciones y detectar
que la regla de vuelta del 95% no funciona en Gymnasium 1.x.

---

## Referencias

[1] D. Ha, J. Schmidhuber. *World Models*. arXiv:1803.10122, 2018. Versión NeurIPS 2018: *Recurrent World Models Facilitate Policy Evolution*.
https://arxiv.org/abs/1803.10122 · https://papers.nips.cc/paper_files/paper/2018/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf · https://worldmodels.github.io/

[2] S. Risi, K. O. Stanley. *Deep Neuroevolution of Recurrent and Discrete World Models*. GECCO 2019. arXiv:1906.08857.
https://arxiv.org/abs/1906.08857 · https://dl.acm.org/doi/10.1145/3321707.3321817

[3] Y. Tang, D. Nguyen, D. Ha. *Neuroevolution of Self-Interpretable Agents*. GECCO 2020. arXiv:2003.08165.
https://arxiv.org/abs/2003.08165 · https://attentionagent.github.io/

[3b] O. Kachaiev. *car-racing-attention-agent* (reproducción de [3] en CarRacing-v2).
https://github.com/kachayev/car-racing-attention-agent

[4] H. Chang, K. Futagami. *Convolutional Reservoir Computing for World Models*. arXiv:1907.08040, 2019; *Reinforcement Learning with Convolutional Reservoir Computing*, Applied Intelligence, 2020.
https://arxiv.org/abs/1907.08040 · https://arxiv.org/abs/1912.04161

[5] Y. Tang, D. Ha. *The Sensory Neuron as a Transformer: Permutation-Invariant Neural Networks for Reinforcement Learning*. NeurIPS 2021. arXiv:2109.02869.
https://arxiv.org/abs/2109.02869

[6] I. G. B. Petrazzini, E. A. Antonelo. *Proximal Policy Optimization with Continuous Bounded Action Space via the Beta Distribution*. arXiv:2111.02202, 2021.
https://arxiv.org/abs/2111.02202

[7] xtma. *pytorch_car_caring: Reinforcement Learning for Gym CarRacing-v0 with PyTorch*.
https://github.com/xtma/pytorch_car_caring

[8] Rafael1s. *CarRacing-From-Pixels-PPO*.
https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms/tree/master/CarRacing-From-Pixels-PPO

[9] F. Khan, O. H. Elibol. *Car Racing using Reinforcement Learning*. Stanford CS221, 2017. Citado en [1] y [22].

[10] S. Jang, J. Min, C. Lee. *Reinforcement Car Racing with A3C*. 2017. Citado en [1] y [22].
https://www.scribd.com/document/358019044/Reinforcement-Car-Racing-with-A3C

[11] wpiszlogin. *driver_critic: Solution for CarRacing-v0 using DDPG*.
https://github.com/wpiszlogin/driver_critic

[12] F. Yuwono, G. P. Yen, J. Christopher. *Self-Driving Car Racing: Application of Deep Reinforcement Learning*. arXiv:2410.22766, 2024.
https://arxiv.org/abs/2410.22766

[13] L. Prieur. DQN en CarRacing-v0, 2017. Citado en [1].

[14] C. Li. *Challenging On Car Racing Problem from OpenAI Gym*. arXiv:1911.04868, 2019.
https://arxiv.org/abs/1911.04868

[15] P. Aldape, S. Sowell. *Reinforcement Learning for a Simple Racing Game*. Stanford AA228, 2018. Puntaje citado en: *Applying a Deep Q Network for OpenAI's Car Racing Game* (Towards Data Science).
https://web.stanford.edu/class/aa228/reports/2018/final150.pdf · https://towardsdatascience.com/applying-a-deep-q-network-for-openais-car-racing-game-a642daf58fc9/

[16] Y. Maniyar, N. Manoj. *Racetrack Navigation on OpenAIGym with Deep Reinforcement Learning*. Stanford AA228/CS238, 2019.
https://web.stanford.edu/class/aa228/reports/2019/final9.pdf · https://github.com/nidhimanoj4/CS238CarNavigation

[17] andywu0913. *OpenAI-GYM-CarRacing-DQN*.
https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN

[18] M. Dahlin et al. *Designing Interpretable Approximations to Deep Reinforcement Learning*. arXiv:2010.14785, 2020.
https://arxiv.org/abs/2010.14785

[19] N. van Hoeffelen, P. Lanillos. *Deep Active Inference for Pixel-Based Discrete Control: Evaluation on the Car Racing Problem*. IWAI 2021. arXiv:2109.04155.
https://arxiv.org/abs/2109.04155 · https://github.com/NTAvanHoeffelen/DAIF_CarRacing

[20] Y. Zhang, H. Sun. *Deep Reinforcement Learning with Mixed Convolutional Network*. arXiv:2010.00717, 2020.
https://arxiv.org/abs/2010.00717 · https://github.com/zhangyanyu0722/Car_Racing_DL

[21] Farama Foundation. *Gymnasium: Car Racing*.
https://gymnasium.farama.org/environments/box2d/car_racing/

[22] PDillis. *Experiment-CarRacing* (A3C; cita los puntajes de [9] y [10]).
https://github.com/PDillis/Experiment-CarRacing

[23] Modelos PPO de la comunidad para CarRacing-v2 en Hugging Face (por ejemplo):
https://huggingface.co/satcos/PPO-CarRacing-v2 · https://huggingface.co/Cheekydave/ppo-CarRacing-v2

[24] KohMat. *carracing-dreamer*.
https://github.com/KohMat/carracing-dreamer

Referencias de método usadas en este trabajo: J. Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 (2017); V. Mnih et al., *Human-level control through deep reinforcement learning*, Nature 518 (2015); H. van Hasselt, A. Guez, D. Silver, *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016, arXiv:1509.06461.
