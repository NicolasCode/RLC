# 🏁 Guía para Escoger Pistas - Exploración de Semillas

## 📋 Resumen

Esta guía explica cómo se obtuvieron las **top seeds** y de dónde sale el **cálculo de complejidad** para generar pistas similares a Monza en el entorno CarRacing.

---

## 🆕 **RESULTADOS ACTUALIZADOS - EXPLORACIÓN 2024**

### 🏆 Top 10 Semillas Más Interesantes

| Posición | Semilla | Complejidad | Puntos Track | Características |
|----------|---------|-------------|--------------|-----------------|
| 1 | 38 | 27.95 | 354 | **Máxima complejidad** - Muchas curvas pronunciadas |
| 2 | 35 | 27.20 | 304 | **Alta densidad** - Curvas desafiantes |
| 3 | 36 | 22.58 | 350 | **Balance perfecto** - Complejo pero manejable |
| 4 | 9 | 22.43 | 312 | **Curvas moderadas** - Buen equilibrio |
| 5 | 18 | 22.09 | 306 | **✅ ELEGIDA** - Tu Monza favorito |
| 6 | 1 | 21.77 | 308 | **Clásica** - Siempre confiable |
| 7 | 13 | 21.75 | 324 | **Curvas suaves** - Flujo natural |
| 8 | 42 | 21.28 | 303 | **Moderada** - Buen rendimiento |
| 9 | 3 | 19.60 | 294 | **Equilibrada** - Curvas y rectas |
| 10 | 2 | 17.22 | 321 | **Suave** - Menos compleja pero interesante |

### 📊 Estadísticas de la Exploración

- ✅ **Total semillas probadas**: 50 (1-50)
- ✅ **Tracks exitosos**: 50 (100% éxito)
- ✅ **Imágenes generadas**: 50 individuales + 1 resumen
- ✅ **Directorio**: `images/seeds/`

---

## 🔍 **PROCESO DE EXPLORACIÓN DE SEMILLAS**

### 1. **Algoritmo de Búsqueda Sistemática**

El proceso utiliza el script `seed_explorer.py` que:

```python
# Explora semillas del 1 al 50 sistemáticamente
for seed in range(1, 51):
    env = create_track_with_seed(seed)
    characteristics = analyze_track_characteristics(env)
    # Guarda cada resultado con su puntuación de complejidad
```

**Pasos del algoritmo:**
1. Crea un entorno con cada semilla
2. Genera una imagen preview del track
3. Analiza las características geométricas
4. Calcula el score de complejidad
5. Guarda los resultados para ranking

---

## 📊 **CÁLCULO DE COMPLEJIDAD**

### Fórmula de Complejidad

La **complejidad** se calcula analizando la **curvatura del track**:

```python
# En analyze_track_characteristics()
for i in range(len(track)):
    # Toma 3 puntos consecutivos del track
    p1, p2, p3 = track[i], track[i+1], track[i+2]
    
    # Calcula vectores entre puntos
    v1 = [p2 - p1]  # Vector del segmento 1
    v2 = [p3 - p2]  # Vector del segmento 2
    
    # Calcula el ángulo entre vectores (curvatura)
    angle = arccos(dot(v1, v2) / (norm(v1) * norm(v2)))
    angles.append(angle)

# FÓRMULA FINAL DE COMPLEJIDAD:
complexity_score = avg_curvature * len(angles)
```

### Componentes del Score

- **Curvatura promedio**: Qué tan "curvo" es el track en general
- **Número de curvas**: Cuántos cambios de dirección hay
- **Score final**: `curvatura_promedio × número_de_curvas`

---

## 🎯 **CÓMO ELEGIR LA MEJOR SEMILLA PARA MONZA**

### Criterios de Selección

1. **👁️ Similitud Visual**: ¿Se parece al layout real de Monza?
2. **🏎️ Manejabilidad**: ¿Es desafiante pero no imposible?
3. **⚖️ Balance**: ¿Tiene buena mezcla de curvas y rectas?
4. **🔄 Consistencia**: ¿Genera siempre el mismo layout?

### Semillas Recomendadas para Revisar

**🔥 Candidatos Principales:**
- **Semilla 38**: Máxima complejidad (27.95)
- **Semilla 35**: Alta complejidad (27.20)
- **Semilla 36**: Balance perfecto (22.58)
- **Semilla 9**: Equilibrio ideal (22.43)
- **✅ Semilla 18**: ¡ELEGIDA! Complejidad equilibrada (22.09)

**🎯 Para Monza específicamente, busca:**
- Curvas lentas al inicio
- Recta larga (como la de Monza)
- Chicanes rápidas
- Curvas de alta velocidad

---

## 🛠️ **HERRAMIENTAS UTILIZADAS**

### Scripts Principales

1. **`seed_explorer.py`**: Exploración sistemática actualizada
2. **`manual_control.py`**: Para probar semillas manualmente
3. **`track_visualizer.py`**: Generación de imágenes detalladas

### Archivos Generados

- **📁 `images/seeds/`**: 50 imágenes individuales de tracks
- **📊 `seed_exploration_summary_*.png`**: Resumen visual de top 10
- **🏁 `images/Monza/`**: Imágenes oficiales de semilla 18
- **📝 Este archivo**: Documentación actualizada

---

## 🎮 **CÓMO USAR UNA SEMILLA**

### Para Probar una Semilla Específica

```python
import numpy as np
import gymnasium as gym

# ✅ SEMILLA ELEGIDA: usar semilla 18 (tu Monza favorito)
np.random.seed(18)
env = gym.make('CarRacing-v2')
```

### Para Actualizar manual_control.py

```python
# En manual_control.py, línea ~15
def main():
    np.random.seed(18)  # ✅ SEMILLA ELEGIDA - Tu Monza favorito
    env = gym.make('CarRacing-v2', render_mode='human')
    # ... resto del código
```

---

## 📈 **PRÓXIMOS PASOS**

### 1. **Revisión Visual** 🔍
- Abre la carpeta `images/seeds/`
- Revisa las imágenes de las top 10 semillas
- Compara con el layout real de Monza

### 2. **Prueba Manual** 🎮
- Actualiza `manual_control.py` con tu semilla favorita
- Prueba la manejabilidad del track
- Verifica que se sienta como Monza

### 3. **Validación** ✅
- Ejecuta varias veces para confirmar consistencia
- Ajusta si es necesario
- Documenta tu elección final

---

## 🔄 **EXPLORACIÓN FUTURA**

### Para Encontrar Más Pistas Interesantes

1. **Ampliar rango**: Probar semillas 51-100, 101-200, etc.
2. **Refinar algoritmo**: Incluir métricas de similitud con pistas reales
3. **Validación cruzada**: Combinar análisis algorítmico y visual
4. **Pistas específicas**: Buscar semillas para otros circuitos famosos

### Comando para Explorar Más Semillas

```bash
python seed_explorer.py  # Modifica el rango en main()
```

---

## 📝 **CONCLUSIONES ACTUALIZADAS**

- ✅ **50 semillas exploradas** con 100% de éxito
- ✅ **Semilla 18 ELEGIDA** como el Monza oficial
- ✅ **Imágenes consistentes** guardadas en `images/seeds/`
- ✅ **Scripts actualizados** con semilla 18
- ✅ **Metodología probada** y documentada

**🎯 ¡Semilla 18 implementada! Tu Monza favorito está listo para correr! 🇮🇹🏎️**