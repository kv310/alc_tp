# alc_tp
Trabajo Práctico: Calibración y Pose de la Cámara.

Integrantes:
Francisco Fogwill
Augusto
Benjamín
Ezequiel

# Reglas de entrega — TP 2 (ALC, 2°C 2026)

## 🧼 Calidad de código

- **Encabezado de funciones**: cada función definida debe contener un encabezado que explique
  los **parámetros** que recibe (y su forma/tipo) y **qué se espera que retorne**.
- **Comentarios internos**: las secciones de código dentro de la función deben estar
  debidamente comentadas de a bloques.
- **Nombres de variables**: deben ser **explicativos** (nada de `x1`, `tmp`, `arr2`).
- **Unit tests**: cada función implementada en `alc.py` debe tener sus **tests unitarios
  correspondientes** (carpeta `tests/`, un archivo por ejercicio, ej. `tests/test_ej3.py`),
  en tres niveles:
  1. **Estructurales** — shapes, guards (el `assert` de entradas inconsistentes).
  2. **Construcción sintética** — se genera primero la solución conocida (H, K
     sintéticas), se fabrican los datos de entrada desde esa solución y se verifica
     que la función la recupera (tolerancia `1e-9`). El test **no reimplementa** la
     función: valida contra una construcción independiente.
  3. **Regresión con datos reales** — los checks de QA (shapes, error de reproyección
     con tolerancias holgadas) fijados como baseline.

  Los tests son herramienta de desarrollo, **no forman parte del código entregado**,
  así que en ellos sí se permite todo (incluido lo prohibido más abajo). Correr con
  `python -m pytest tests/ -v` (desde la raíz del repo).

## 📦 Uso de librerías — la lista blanca

Del código entregado, está permitido usar **únicamente** las siguientes funciones de
librerías externas, y nada más:

### Arrays (NumPy)
| | | | |
|---|---|---|---|
| `np.array` | `np.asarray` | `np.zeros` | `np.ones` |
| `np.eye` | `np.arange` | `np.hstack` | `np.vstack` |
| `reshape` | `.T` | `np.float32` | `np.int32` |
| `np.set_printoptions` | indexación y slicing | | |

### Operaciones elemento a elemento (NumPy)
| | | | |
|---|---|---|---|
| `+` `-` `*` `/` `**` | `np.sqrt` | `np.abs` | `np.sum` |
| `np.mean` | `np.max` | `np.min` | `np.all` |
| `np.any` | `np.diff` | | |

### Lectura de archivos e imágenes
| Archivos | Imágenes |
|---|---|
| `np.loadtxt` | `cv2.imread` |
| `pd.read_csv` | `matplotlib.image.imread` |
| `os.listdir` / `os.path.join` | |

### Gráficos (Matplotlib)
| | | | |
|---|---|---|---|
| `plt.figure` | `plt.subplots` | `plt.imshow` | `plt.plot` |
| `plt.scatter` | `plt.text` | `plt.Circle` | `plt.title` |
| `plt.legend` | `plt.tight_layout` | `plt.savefig` | `plt.show` |

## 🚫 Lo prohibido — hay que implementarlo uno mismo

Todo lo demás debe ser implementado por el grupo, **reutilizando el módulo ALC construido
durante el laboratorio**. En particular NO se permite usar librerías para:

| Categoría | Funciones prohibidas |
|---|---|
| Producto matricial | `@`, `np.dot`, `np.matmul` y similares |
| Factorizaciones | **QR, SVD, Cholesky** |
| Álgebra matricial | inversa, determinante, pseudoinversa |
| Medidas | normas, número de condición, **rango**, autovalores |
| Geometría vectorial | producto vectorial |
| OpenCV | `calibrateCamera`, `solvePnP`, `projectPoints`, `findHomography`, `Rodrigues`, etc. |

## ✅ Permiso especial para desarrollo

Estas funciones **no están permitidas en el código entregado**, pero **sí pueden usarse
durante el desarrollo** para **verificar** los resultados de las implementaciones propias.

> 💡 Estrategia estándar: prototipás con `np.linalg.svd` (1 línea), después implementás tu
> propio descompoenente SVD en el módulo ALC, y comparás contra el prototipo hasta que den
> lo mismo.

## 🔀 Flujo de trabajo con Git y PRs

- **No pushear directamente a `main`**: la rama principal no recibe push directo. Todo cambio
  llega a `main` exclusivamente vía **Pull Request**.
- **Aprobación de PRs**: cada PR requiere la aprobación de **al menos 2 integrantes**
  del grupo antes del merge.
- Ramas por issue (cono `<n>-<titulo-del-issue>`) o de workspace personal
  (`<usuario>/<tema>`) — ver `CONTEXTO_TP2.md` para el detalle de la convención.
- Los PRs deben referenciar el issue que cierran (`Closes #N`) y pasar los unit tests
  correspondientes antes del merge.

## 📝 Presentación: conclusiones y razonamientos

- Las conclusiones y razonamientos que responden los ejercicios (y toda experimentación
  agregada) deben estar **explicados en bloques de texto de las notebooks** (markdown
  cells), **separados de los bloques de código**.
- Usen **código LaTeX** para las fórmulas — sobre todo para la **demostración del primer
  ejercicio** (la minimización $\min\|Qm\|^2$ con $\|m\|=1$, que ya derivamos con Lagrange).
