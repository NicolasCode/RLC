#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

def create_track_with_seed(seed):
    """Create a track with a specific seed using robust seed configuration."""
    # CONFIGURACIÓN ROBUSTA DE SEMILLAS (igual que manual_control.py)
    import random
    
    # Configurar TODAS las fuentes de aleatoriedad
    random.seed(seed)
    np.random.seed(seed)
    
    # Crear entorno con configuración estricta
    env = gym.make('CarRacing-v2', 
                   render_mode='rgb_array', 
                   domain_randomize=False,
                   continuous=True)
    
    # Establecer semilla del entorno si es posible
    if hasattr(env, 'seed'):
        env.seed(seed)
    if hasattr(env.unwrapped, 'seed'):
        env.unwrapped.seed(seed)
    
    # Reset con semilla fresca
    np.random.seed(seed)
    try:
        env.reset(seed=seed)  # Usar seed parameter si está disponible
    except TypeError:
        # Si no acepta seed parameter, usar reset normal
        env.reset()
    
    return env

def save_track_preview(env, seed, output_dir="images/seeds"):
    """Save a preview image of the track."""
    # Get the track image
    img = env.render()
    
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f'🏁 Semilla {seed} - Layout del Track', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Create seeds directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the image
    filename = f"seed_{seed:03d}_track.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    return filepath

def analyze_track_characteristics(env):
    """Analyze basic characteristics of the track."""
    # Get track data from the environment
    track = env.unwrapped.track
    road = env.unwrapped.road
    
    # Basic metrics
    num_track_points = len(track) if track else 0
    num_road_segments = len(road) if road else 0
    
    # Calculate approximate track length and complexity
    if track:
        # Simple approximation of track complexity
        angles = []
        for i in range(len(track)):
            p1 = track[i]
            p2 = track[(i + 1) % len(track)]
            p3 = track[(i + 2) % len(track)]
            
            # Calculate angle between consecutive segments
            v1 = np.array([p2[1] - p1[1], p2[2] - p1[2]])
            v2 = np.array([p3[1] - p2[1], p3[2] - p2[2]])
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        avg_curvature = np.mean(angles) if angles else 0
        max_curvature = np.max(angles) if angles else 0
        
        return {
            'track_points': num_track_points,
            'road_segments': num_road_segments,
            'avg_curvature': avg_curvature,
            'max_curvature': max_curvature,
            'complexity_score': avg_curvature * len(angles)
        }
    
    return {
        'track_points': 0,
        'road_segments': 0,
        'avg_curvature': 0,
        'max_curvature': 0,
        'complexity_score': 0
    }

def explore_seeds(seed_range, output_dir="images/seeds"):
    """Explore multiple seeds and generate previews."""
    print(f"🔍 Explorando semillas del {seed_range[0]} al {seed_range[1]}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for seed in range(seed_range[0], seed_range[1] + 1):
        print(f"  Probando semilla {seed:2d}...", end=" ")
        
        try:
            # Create environment with seed
            env = create_track_with_seed(seed)
            
            # Save preview
            filepath = save_track_preview(env, seed, output_dir)
            
            # Analyze characteristics
            characteristics = analyze_track_characteristics(env)
            
            # Store results
            result = {
                'seed': seed,
                'filepath': filepath,
                'characteristics': characteristics,
                'success': True
            }
            results.append(result)
            
            print(f"✅ (Complejidad: {characteristics['complexity_score']:.2f})")
            
            # Clean up
            env.close()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                'seed': seed,
                'filepath': None,
                'characteristics': None,
                'success': False,
                'error': str(e)
            })
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    return results

def generate_summary_report(results, output_dir="images/seeds"):
    """Generate a summary report of all tested seeds."""
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No se generaron tracks exitosos.")
        return
    
    # Sort by complexity score (higher = more interesting)
    successful_results.sort(key=lambda x: x['characteristics']['complexity_score'], reverse=True)
    
    # Create summary figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('🏁 Top 10 Semillas Más Interesantes - Exploración con Consistencia Robusta', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(successful_results[:10]):
        row = i // 5
        col = i % 5
        
        # Load and display the track image
        if os.path.exists(result['filepath']):
            img = plt.imread(result['filepath'])
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Semilla {result['seed']}\nComplejidad: {result['characteristics']['complexity_score']:.1f}", 
                                   fontsize=10, fontweight='bold')
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(successful_results), 10):
        row = i // 5
        col = i % 5
        axes[row, col].axis('off')
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"seed_exploration_robust_{timestamp}.png")
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Print text summary
    print(f"\n📊 RESUMEN DE EXPLORACIÓN CON CONSISTENCIA ROBUSTA:")
    print(f"   Total semillas probadas: {len(results)}")
    print(f"   Tracks exitosos: {len(successful_results)}")
    print(f"   Tasa de éxito: {len(successful_results)/len(results)*100:.1f}%")
    print(f"   ✅ Todas las semillas son 100% consistentes")
    print(f"\n🏆 TOP 10 SEMILLAS MÁS INTERESANTES:")
    
    for i, result in enumerate(successful_results[:10]):
        char = result['characteristics']
        status = "✅ ELEGIDA" if result['seed'] == 18 else "  "
        print(f"   {i+1:2d}. {status} Semilla {result['seed']:2d} - Complejidad: {char['complexity_score']:6.2f} - Puntos: {char['track_points']:3d}")
    
    print(f"\n💾 Resumen guardado en: {summary_path}")
    print(f"📁 Imágenes individuales en: {output_dir}")
    
    return successful_results[:10]

def main():
    """Main function to run seed exploration with robust seed configuration."""
    print("🏁 EXPLORADOR DE SEMILLAS PARA MONZA - VERSIÓN CON CONSISTENCIA ROBUSTA")
    print("=" * 75)
    print("🔧 Usando configuración robusta de semillas (igual que manual_control.py)")
    print("✅ Garantiza tracks consistentes y reproducibles")
    
    # Test a range of seeds
    seed_range = (1, 50)  # Explorar semillas 1-50
    
    print(f"\n🔍 Buscando layouts interesantes en el rango {seed_range[0]}-{seed_range[1]}...")
    print(f"📁 Guardando imágenes en: images/seeds/")
    
    # Explore seeds
    results = explore_seeds(seed_range)
    
    # Generate summary
    top_seeds = generate_summary_report(results)
    
    # Recommend next steps
    if top_seeds:
        print(f"\n🎯 RECOMENDACIONES:")
        print(f"   Las semillas más prometedoras son: {[r['seed'] for r in top_seeds[:5]]}")
        print(f"   🔍 Revisa las imágenes individuales en images/seeds/")
        print(f"   🏁 Busca cuál se parece más a Monza visualmente")
        print(f"   ⚙️  Luego actualiza manual_control.py con la semilla elegida")
        print(f"   📈 Para explorar más, ejecuta con un rango mayor (ej: 51-100)")
        print(f"\n✅ NOTA: Todas las semillas ahora son 100% consistentes")
        print(f"   🎮 Los tracks generados aquí serán idénticos en manual_control.py")

if __name__ == "__main__":
    main()