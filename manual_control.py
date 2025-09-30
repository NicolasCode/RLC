"""
Manual control for CarRacing environment with selected track
Allows human player to control the car using keyboard on the legendary Monza circuit
"""
import gymnasium as gym
import pygame
from Utils.interpreters import gym_interpreter_3

def manual_control():
    """
    Run CarRacing environment with manual control on Monza track
    Controls:
    - WASD keys: Steer left/right, accelerate/brake
    - ESC: Quit
    """
    # Create environment with human rendering
    env = gym.make('CarRacing-v2', 
                   render_mode='human',
                   continuous=False,
                   lap_complete_percent=0.07,
                   domain_randomize=False)
    
    # Apply Monza custom track modification (same approach as train.py)
    special_step = env.step
    special_reset = env.reset
    
    # Create interpreter instance
    interpreter = gym_interpreter_3()
    
    def step(action):
        obs, reward, terminated, truncated, info = special_step(action)
        obs = interpreter.interpret(obs)
        return obs, reward, terminated, truncated, info
    
    def reset(seed=None, options=None):
        obs, info = special_reset(seed=seed, options=options)
        obs = interpreter.interpret(obs)
        return obs, info
    
    # Override environment methods
    setattr(env, 'special_step', special_step)
    setattr(env, 'step', step)
    setattr(env, 'special_reset', special_reset)
    setattr(env, 'reset', reset)
    
    # Sobrescribir el método _create_track para usar nuestro layout personalizado
    env._create_track = lambda: tc._create_track(env)
    print("🏁 Usando layout personalizado de Monza")
    
    # Initialize pygame for keyboard input
    pygame.init()
    
    # Action mapping for discrete actions
    # 0: do nothing, 1: steer left, 2: steer right, 3: accelerate, 4: brake
    # 1: a, 2: d, 3: w, 4: s
    action_map = {
        pygame.K_a: 2,    # Steer left
        pygame.K_d: 1,   # Steer right  
        pygame.K_w: 3,      # Accelerate
        pygame.K_s: 4,    # Brake
    }
    
    print("🏁 Control Manual en Monza - El Templo de la Velocidad")
    print("Controles:")
    print("  W: Acelerar")
    print("  A: Girar izquierda") 
    print("  D: Girar derecha")
    print("  S: Frenar")
    print("  R: Reiniciar")
    print("  ESC: Salir")
    print("🏎️ ¡Prepárate para conquistar Monza!")
    print("Starting...")
    
    obs, info = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    
    clock = pygame.time.Clock()
    
    try:
        while not done:
            # Handle pygame events
            action = 0  # Default: do nothing
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
            
            # Check currently pressed keys
            keys = pygame.key.get_pressed()
            for key, mapped_action in action_map.items():
                if keys[key]:
                    action = mapped_action
                    break
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step_count += 1
            
            # Render environment
            env.render()
            
            # Control frame rate
            clock.tick(30)  # 30 FPS
            
            # Print progress occasionally
            if step_count % 100 == 0:
                print(f"Steps: {step_count}, Reward: {total_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print(f"\n🏁 ¡Episodio en Monza completado!")
        print(f"🏎️ Pasos totales: {step_count}")
        print(f"🏆 Recompensa total: {total_reward:.2f}")
        print("🇮🇹 ¡Grazie per aver guidato a Monza!")
        env.close()
        pygame.quit()

def main():
    print("🏁 Configurando Monza con semilla 18...")
    
    # Establecer semilla global de Python
    import random
    import numpy as np
    
    # Configurar TODAS las fuentes de aleatoriedad
    random.seed(18)
    np.random.seed(18)
    
    # Crear entorno con configuración estricta
    env = gym.make('CarRacing-v2', 
                   render_mode='human', 
                   domain_randomize=False,
                   continuous=True)
    
    # Establecer semilla del entorno si es posible
    if hasattr(env, 'seed'):
        env.seed(18)
    if hasattr(env.unwrapped, 'seed'):
        env.unwrapped.seed(18)
    
    # Reset con semilla fresca
    np.random.seed(18)
    env.reset(seed=18)  # Usar seed parameter si está disponible
    
    print("✅ Track generado con semilla 18")
    
    total_reward = 0.0
    steps = 0
    restart = False
    
    def get_keyboard_input():
        # Simple keyboard input function
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            return None
        elif keys[pygame.K_r]:
            return "restart"
        
        # CarRacing expects [steering, gas, brake] format
        steering = 0.0
        gas = 0.0
        brake = 0.0
        
        if keys[pygame.K_a]:  # steer left
            steering = -1.0
        elif keys[pygame.K_d]:  # steer right
            steering = 1.0
            
        if keys[pygame.K_w]:  # accelerate
            gas = 1.0
        elif keys[pygame.K_s]:  # brake
            brake = 0.8
            
        return [steering, gas, brake]
    
    while True:
        env.render()
        
        # Get keyboard input
        action = get_keyboard_input()
        
        if action is None:  # Quit
            break
        elif action == "restart":
            restart = True
        
        if restart:
            # Configuración robusta para mantener consistencia
            import random
            random.seed(18)
            np.random.seed(18)
            
            # Reset con semilla explícita
            try:
                env.reset(seed=18)
            except TypeError:
                # Si no acepta seed parameter, usar reset normal
                env.reset()
            
            total_reward = 0.0
            steps = 0
            restart = False
            print("🔄 Reiniciando con semilla 18...")
            continue
        
        # Only step if we have a valid action array
        if isinstance(action, list):
            s, r, terminated, truncated, info = env.step(action)
            total_reward += r
            
            if steps % 200 == 0 or terminated or truncated:
                print(f"Paso {steps+1}: reward {r:+0.2f} (total: {total_reward:+0.2f})")
            
            steps += 1
            
            if terminated or truncated:
                print(f"🏁 Episodio terminado después de {steps} pasos, reward total: {total_reward:+0.2f}")
                # Auto-restart con configuración robusta de semillas
                import random
                random.seed(18)
                np.random.seed(18)
                
                # Reset con semilla explícita
                try:
                    env.reset(seed=18)
                except TypeError:
                    # Si no acepta seed parameter, usar reset normal
                    env.reset()
                
                total_reward = 0.0
                steps = 0
    
    env.close()

if __name__ == "__main__":
    main()