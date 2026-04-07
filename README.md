# Extracción y análisis de comentarios de YouTube

Repositorio para **descargar**, **estructurar** y **analizar** comentarios de un video de YouTube con un enfoque **CRISP-DM** (entendimiento del negocio, datos, preparación, modelado y evaluación). El núcleo del trabajo está en el cuaderno Jupyter `analysis.ipynb`, orientado al debate *vibe coding* vs desarrollo tradicional en torno a un video concreto de Riley Brown.

---

## Tabla de contenidos

1. [Resumen](#resumen)
2. [Qué incluye el repositorio](#qué-incluye-el-repositorio)
3. [Requisitos del sistema](#requisitos-del-sistema)
4. [Instalación](#instalación)
5. [Estructura de carpetas](#estructura-de-carpetas)
6. [Datos de entrada (`comments.json`)](#datos-de-entrada-commentsjson)
7. [Flujo del análisis (`analysis.ipynb`)](#flujo-del-análisis-analysisipynb)
8. [Salidas generadas](#salidas-generadas)
9. [Ollama y modelos locales (opcional)](#ollama-y-modelos-locales-opcional)
10. [Exportar el cuaderno a HTML o PDF](#exportar-el-cuaderno-a-html-o-pdf)
11. [Git y `.gitignore`](#git-y-gitignore)
12. [Problemas frecuentes](#problemas-frecuentes)
13. [Licencia y uso de datos](#licencia-y-uso-de-datos)

---

## Resumen

- **Entrada:** comentarios en `data/raw/comments.json` (JSONL línea a línea, JSON con `metadata` + `comments`, o un array JSON).
- **Proceso:** limpieza de texto, sentimiento (VADER), embeddings (`sentence-transformers`), reducción de dimensionalidad (UMAP), clustering (HDBSCAN), visualizaciones (Matplotlib, Seaborn, Plotly), co-ocurrencia de conceptos (NetworkX) y, opcionalmente, etiquetado de clusters y preguntas en lenguaje natural vía **Ollama**.
- **Salida:** CSV limpio, embeddings `.npy`, CSV de clusters, gráficos en `data/processed/` y figuras interactivas en el propio notebook.

---

## Qué incluye el repositorio

| Elemento | Descripción |
|----------|-------------|
| `analysis.ipynb` | Cuaderno principal: pipeline completo CRISP-DM y visualizaciones. |
| `requirements.txt` | Dependencias de Python recomendadas. |
| `data/raw/comments.json` | Comentarios crudos (no ignorado por Git si lo versionás; puede ser grande). |
| `data/processed/` | Artefactos generados al ejecutar el notebook (ignorado en Git). |
| `.gitignore` | Excluye entornos virtuales, cachés y `data/processed/`. |

El paquete **`youtube-comment-downloader`** en `requirements.txt` sirve para **obtener** comentarios desde la línea de comandos; el flujo típico es exportar a JSON/JSONL y colocar o fusionar el resultado en `data/raw/comments.json`.

---

## Requisitos del sistema

- **Python 3.10+** (3.12 probado en Windows).
- **Espacio en disco:** varios cientos de MB para el entorno + modelo `all-MiniLM-L6-v2` (sentence-transformers) la primera vez que corre la celda de embeddings.
- **RAM:** embeddings y UMAP en ~3k comentarios son razonables en 8 GB; más margen ayuda.
- **Ollama** (opcional): solo si usás las celdas que llaman a `ollama.chat` (etiquetas de clusters, RAG/preguntas). Debe estar instalado y con el servidor en marcha (`ollama serve`), y el modelo que indiques (p. ej. `qwen3:1.7b`) descargado.

---

## Instalación

1. **Clonar o copiar** el repositorio y situarte en la raíz del proyecto:

   ```powershell
   cd "C:\Proyectos\Extración de comentarios"
   ```

2. **Crear y activar un entorno virtual** (recomendado):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias:**

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Registrar el kernel de Jupyter** (opcional, para elegir el venv en VS Code/Cursor):

   ```powershell
   python -m ipykernel install --user --name=comments-analysis --display-name="Comments analysis"
   ```

5. Abrir `analysis.ipynb` y seleccionar el intérprete **`.venv`**.

---

## Estructura de carpetas

```
.
├── analysis.ipynb          # Análisis principal
├── analysis.html           # (Opcional) exportación HTML; no está en .gitignore por defecto
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── raw/
    │   └── comments.json   # Entrada: comentarios
    └── processed/          # Generado al ejecutar el notebook (ignorado por Git)
        ├── comments_clean.csv
        ├── embeddings.npy
        ├── clusters.csv
        └── *.png           # Figuras guardadas desde Matplotlib
```

---

## Datos de entrada (`comments.json`)

El cuaderno intenta cargar el archivo de forma **tolerante**:

1. **JSON objeto** con clave `comments` (lista de objetos) y opcionalmente `metadata`.
2. **JSON array** de objetos (cada uno un comentario).
3. Si falla el parseo completo, **JSONL**: una línea = un objeto JSON por comentario.

Campos que el análisis espera encontrar (típicos del downloader de YouTube):

- `cid`, `text`, `time`, `author`, `channel`, `votes`, `replies`, `photo`, `heart`, `reply`, `time_parsed`, etc.

Tras cargar, **`votes`** y **`replies`** se normalizan a enteros numéricos para evitar errores al comparar o ordenar.

---

## Flujo del análisis (`analysis.ipynb`)

El notebook sigue las fases CRISP-DM descritas en las celdas iniciales.

### Fase 1 — Entendimiento del negocio

Contexto del video (competencia vibe coder vs ingeniero iOS, app tipo Granola, debate técnico/cultural). Preguntas e hipótesis sobre perfiles de comentaristas y actitud hacia el *vibe coding*.

### Fase 2 — Entendimiento de los datos

- Carga desde `DATA_RAW`.
- Vista de dimensiones, columnas y muestras.
- Estadísticas descriptivas (p. ej. comentarios con likes, longitud de texto).
- Exploración de frecuencia de palabras y **cobertura de conceptos** (`CONCEPTS`: ide, terminal, vibe coding, cursor, etc.).

### Fase 3 — Preparación de datos

- Limpieza de URLs, menciones, espacios.
- Filtro de comentarios “mínimamente informativos” y deduplicación por `text_clean`.
- Exportación a `data/processed/comments_clean.csv`.

### Fase 4 — Modelado

- **Embeddings:** `SentenceTransformer('all-MiniLM-L6-v2')` sobre `text_clean`; guardado en `embeddings.npy`.
- **Sentimiento:** VADER (`compound`, etiquetas discretas).
- **Clustering:** UMAP (espacio 10D para clusterizar, 2D para visualizar) + **HDBSCAN** (`min_cluster_size`, `min_samples` configurables; el número de clusters **no** es fijo).
- **Visualización:** scatter Plotly con clusters; grafo de co-ocurrencia de conceptos (NetworkX).

### Fase 5 — Evaluación y síntesis (según celdas ejecutadas)

- Silhouette (donde aplique), tablas cruzadas cluster × sentimiento.
- **Ollama:** etiquetado de clusters con contexto del video (celdas que definen `VIDEO_CONTEXT` / `label_cluster_with_llm`).
- Posible celda de **pregunta-respuesta** sobre comentarios usando embeddings + modelo local.

**Orden recomendado:** ejecutar de arriba abajo (`Run All`) después de cambiar datos o hiperparámetros; si solo cambiás clustering, podés repetir desde la sección UMAP/HDBSCAN en adelante si ya tenés `embeddings.npy` cargado en sesión.

---

## Salidas generadas

| Ruta | Contenido |
|------|-----------|
| `data/processed/comments_clean.csv` | Dataset filtrado y limpio para análisis posterior. |
| `data/processed/embeddings.npy` | Matriz de embeddings normalizados. |
| `data/processed/clusters.csv` | Asignación de cluster por fila (según cómo lo guarde el notebook). |
| `data/processed/*.png` | Exportaciones de figuras (distribuciones, conceptos, grafo, heatmaps, etc.). |

Las rutas exactas están definidas en la primera celda de código del notebook (`DATA_CLEAN`, `EMBEDDINGS_PATH`, `CLUSTERS_PATH`).

---

## Ollama y modelos locales (opcional)

1. Instalar [Ollama](https://ollama.com) y asegurarte de que el servicio responda.
2. Descargar el modelo que uses en el notebook, por ejemplo:

   ```bash
   ollama pull qwen3:1.7b
   ```

3. Si falla la conexión, suele ser porque **no está corriendo** `ollama serve` o el firewall bloquea el puerto local.

El notebook puede usar distintos modelos en distintas celdas; revisá las llamadas a `ollama.chat(model=...)`.

---

## Exportar el cuaderno a HTML o PDF

Usá el **mismo Python del entorno** donde instalaste Jupyter/nbconvert (no `python jupyter` como si fuera un script).

```powershell
cd "C:\Proyectos\Extración de comentarios"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to html analysis.ipynb
```

Eso genera `analysis.html` en la carpeta del proyecto. Para PDF hace falta una cadena LaTeX o usar HTML → imprimir a PDF desde el navegador.

---

## Git y `.gitignore`

Se ignoran, entre otros:

- `.venv/`, `venv/`, `env/`
- `.claude/`, `.code-review-graph/`, `.ruff_cache/`
- `data/processed/` (artefactos reproducibles ejecutando el notebook)
- `.ipynb_checkpoints/`

**Importante:** Si un archivo **ya estaba commiteado** antes de añadirlo al `.gitignore`, Git **sigue** rastreándolo hasta que ejecutes `git rm --cached <archivo>`.

Si no querés versionar `analysis.html`, añadilo manualmente al `.gitignore`.

---

## Problemas frecuentes

| Síntoma | Causa probable | Qué hacer |
|--------|----------------|-----------|
| `JSONDecodeError` al cargar comentarios | Formato distinto al esperado o archivo corrupto | Verificar que sea JSON válido, JSON con `comments`, o JSONL línea a línea. |
| `TypeError` al comparar `votes` | Venían como texto | El notebook convierte con `pd.to_numeric`; reejecutar la celda de carga. |
| `jupyter-nbconvert` not found | Usás el Python del sistema sin el paquete | Activar `.venv` o `pip install jupyter nbconvert`. |
| `can't open file 'jupyter'` | Comando `python jupyter nbconvert` | Usar `python -m jupyter nbconvert` o `.\.venv\Scripts\python.exe -m jupyter nbconvert`. |
| Error con Ollama | Servidor apagado o modelo ausente | `ollama serve` y `ollama pull <modelo>`. |
| Muchos/pocos clusters HDBSCAN | Densidad y tamaño mínimo de cluster | Subir `min_cluster_size` / `min_samples` para **menos** clusters; bajar para **más**. |
| Plotly/Mermaid warnings al exportar HTML | Limitaciones de nbconvert con algunos outputs | Normal; el HTML igualmente se genera. |

---

## Licencia y uso de datos

- Los **comentarios de YouTube** son contenido generado por usuarios; su uso debe respetar los [Términos de servicio de YouTube](https://www.youtube.com/t/terms) y la privacidad de las personas.
- Este README no otorga licencia sobre el video ni sobre los comentarios; el propósito del repo es **investigación / análisis** en un entorno controlado.
- Si publicás resultados, considerá anonimizar o agregar citas de ejemplo sin datos personales innecesarios.

---

## Contribución y mantenimiento

- Mantener `requirements.txt` actualizado si añadís imports en el notebook (`pip freeze` solo como referencia; mejor listar paquetes a mano con versiones si necesitás reproducibilidad estricta).
- Documentar en el propio notebook cualquier cambio de video, hipótesis o lista `CONCEPTS` para que el informe y el código sigan alineados.

Si algo de este README no coincide con tu copia local (por ejemplo rutas o nombres de modelos), actualizá las celdas de configuración al inicio de `analysis.ipynb` y reflejá el cambio aquí en una sola línea bajo “Flujo del análisis”.
