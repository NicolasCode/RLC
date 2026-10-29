'''
Entorno CarRacing preparado para que un agente PPO aprenda rápido.

Cambios respecto al CarRacing "crudo" (ver docs/aprendizaje_ppo.md):
  * Acciones: 7 acciones discretas construidas sobre el modo continuo
    (el modo discreto original solo da 0.2 de acelerador y no permite
    girar y acelerar a la vez).
  * Frame skip: cada decisión del agente se repite 4 frames y solo se
    renderiza el último (renderizar es ~75% del costo de simulación).
  * Observación: 4 últimos frames en escala de grises (4, 96, 96) uint8.
  * Inicio: se saltan los primeros 48 frames (animación de zoom de la cámara).
  * Fin anticipado: si el carro pasa `patience` frames sin pisar
    una baldosa nueva (se salió a la hierba o está quieto), el episodio
    termina con una penalización.
'''
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.car_racing import CarRacing

# (dirección, acelerador, freno)
ACTIONS = np.array([
    [ 0.0, 0.0, 0.0],   # 0: nada
    [-1.0, 0.0, 0.0],   # 1: izquierda
    [ 1.0, 0.0, 0.0],   # 2: derecha
    [ 0.0, 1.0, 0.0],   # 3: acelerar
    [ 0.0, 0.0, 0.8],   # 4: frenar
    [-0.5, 0.5, 0.0],   # 5: izquierda + acelerar suave
    [ 0.5, 0.5, 0.0],   # 6: derecha + acelerar suave
], dtype=np.float32)
ACTION_NAMES = ['nada', 'izq', 'der', 'gas', 'freno', 'izq+gas', 'der+gas']

# Las 5 acciones del modo discreto original de Gymnasium (solo para comparar)
ORIGINAL_ACTIONS = np.array([
    [ 0.0, 0.0, 0.0],   # 0: nada
    [ 0.6, 0.0, 0.0],   # 1: derecha
    [-0.6, 0.0, 0.0],   # 2: izquierda
    [ 0.0, 0.2, 0.0],   # 3: acelerar (solo 0.2)
    [ 0.0, 0.0, 0.8],   # 4: frenar
], dtype=np.float32)
ORIGINAL_ACTION_NAMES = ['nada', 'der', 'izq', 'gas0.2', 'freno']
ACTION_SETS = {'custom': (ACTIONS, ACTION_NAMES),
               'original': (ORIGINAL_ACTIONS, ORIGINAL_ACTION_NAMES)}


class FastCarRacing(CarRacing):
    '''
    CarRacing que puede saltarse el render de la observación en los
    frames intermedios del frame skip.
    '''
    skip_render = False

    def _render(self, mode):
        if mode == 'state_pixels' and self.skip_render:
            return self.state
        return super()._render(mode)


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


class RacingEnv(gym.Env):
    '''
    Envoltura completa: acciones discretas, frame skip, stack de frames,
    fin anticipado y límite de tiempo.
    '''
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}

    def __init__(self,
                 frame_skip=4,
                 stack=4,
                 patience=200,
                 no_progress_penalty=-20.0,
                 max_frames=2000,
                 reward_scale=0.1,
                 render_mode=None,
                 lap_complete_percent=0.95,
                 actions='custom'):
        self.env = FastCarRacing(continuous=True,
                                 render_mode=render_mode,
                                 lap_complete_percent=lap_complete_percent,
                                 domain_randomize=False)
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.stack = stack
        self.patience = patience  # en frames
        self.actions, self.action_names = ACTION_SETS[actions]
        self.no_progress_penalty = no_progress_penalty
        self.max_frames = max_frames
        self.reward_scale = reward_scale
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(0, 255, (stack, 96, 96), np.uint8)
        self.frames = np.zeros((stack, 96, 96), np.uint8)

    def reset(self, seed=None, options=None):
        car = self.env
        car.skip_render = False
        obs, _ = car.reset(seed=seed, options=options)
        # Saltamos la animación de zoom inicial (1 segundo = 50 frames)
        for k in range(48):
            car.skip_render = k < 47
            obs, _, _, _, _ = car.step(ACTIONS[0])
        car.skip_render = False
        self.n_frames = 0
        self.no_progress = 0
        self.raw_return = 0.0
        self.frames[:] = to_gray(obs)
        return self.frames.copy(), {}

    def step(self, action):
        car = self.env
        a = self.actions[int(action)]
        reward = 0.0
        progress = False
        terminated = truncated = False
        info = {}
        for k in range(self.frame_skip):
            car.skip_render = k < self.frame_skip - 1
            obs, r, term, _, inf = car.step(a)
            reward += r
            progress |= r > 0
            self.n_frames += 1
            if term:
                terminated = True
                info.update(inf)
                break
        car.skip_render = False
        if terminated and k < self.frame_skip - 1:
            # Terminó en un frame que no se renderizó
            obs = car._render('state_pixels')
        self.raw_return += reward

        # ¿Pisó baldosas nuevas? (cada baldosa da +1000/N > 0)
        if progress:
            self.no_progress = 0
        else:
            self.no_progress += k + 1
        shaped = reward
        if not terminated and self.no_progress >= self.patience:
            terminated = True
            shaped += self.no_progress_penalty
            info['no_progress'] = True
        if not terminated and self.n_frames >= self.max_frames:
            truncated = True

        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = to_gray(obs)

        info['tiles'] = car.tile_visited_count
        info['track_len'] = len(car.track)
        info['lap'] = bool(info.get('lap_finished', False))
        info['raw_return'] = self.raw_return
        info['frames'] = self.n_frames
        return self.frames.copy(), shaped * self.reward_scale, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_env(seed=None, **kwargs):
    def thunk():
        env = RacingEnv(**kwargs)
        env.action_space.seed(seed)
        return env
    return thunk
