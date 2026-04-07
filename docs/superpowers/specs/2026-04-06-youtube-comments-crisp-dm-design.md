# YouTube Comments Analysis — CRISP-DM Design Spec

**Date:** 2026-04-06  
**Project:** Extración de comentarios  
**Video:** "Can I Vibecode a $250M App Better Than a Pro Developer? (With No Code)" — Riley  
**Comments:** 2,838  

---

## 1. Business Understanding

### Objetivo

Analizar los comentarios del video de Riley sobre vibe coding para entender cómo distintos perfiles de audiencia reciben los conceptos técnicos y el fenómeno del vibe coding.

### Preguntas de negocio

1. ¿Cuál es el sentimiento general hacia el video y hacia el vibe coding?
2. ¿Qué perfil tiene la persona que comenta? (coder, vibe-coder, espectador casual, técnico curioso)
3. ¿Las personas técnicas sienten amenaza o escepticismo ante las herramientas de vibe coding?
4. ¿Las personas no técnicas se sorprenden de que Riley no entienda conceptos básicos como IDE o terminal?
5. ¿Existe una brecha de percepción entre "vibe coding es fácil/el futuro" vs "vibe coding no es coding real"?

### Hipótesis iniciales

- H1: Hay un cluster de coders escépticos que reaccionan negativamente a la ignorancia técnica de Riley.
- H2: Hay un cluster de vibe coders/fans entusiastas que celebran que "no necesitás saber código".
- H3: El término `IDE` y `terminal` generan comentarios con tono condescendiente o de incredulidad.
- H4: Las personas técnicas no tienen miedo a las herramientas, sino escepticismo sobre si lo que Riley hace es "real".

---

## 2. Arquitectura General

### Stack tecnológico

| Componente | Herramienta | Razón |
|-----------|-------------|-------|
| Extracción | `youtube-comment-downloader` | Sin API key, Python puro |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Local, 384 dims, rápido |
| Reducción dimensional | `umap-learn` | Mejor que t-SNE para clustering |
| Clustering | `hdbscan` | Clusters naturales, sin K fijo |
| Sentiment | `vaderSentiment` | Optimizado para texto social/informal |
| Grafo | `networkx` | Co-ocurrencia de conceptos |
| Visualización | `matplotlib`, `seaborn`, `plotly` | Estático + interactivo |
| RAG / LLM | `ollama` SDK + `qwen3:1.7b` | Local, sin costo |
| Base | `pandas`, `numpy`, `sklearn` | Manipulación y métricas |

### Modelos locales disponibles

- `qwen3:1.7b` — para el mini RAG (rápido, bajo consumo)
- `gemma4:e4b` — para análisis más profundo si se necesita

### Entorno

- Python global + venv
- Sin GPU requerida
- Sin API keys

### Estructura de archivos

```
Extración de comentarios/
├── data/
│   ├── raw/
│   │   └── comments.json          # Salida de youtube-comment-downloader
│   └── processed/
│       ├── comments_clean.csv     # Después de limpieza
│       ├── embeddings.npy         # Vectores 384-dim
│       └── clusters.csv           # Comentarios + cluster asignado
├── analysis.ipynb                 # Notebook principal (CRISP-DM completo)
├── docs/
│   └── superpowers/specs/
│       └── 2026-04-06-youtube-comments-crisp-dm-design.md
└── requirements.txt
```

---

## 3. Diseño por Fase CRISP-DM

### Fase 1 — Business Understanding (Notebook Section 1)

- Enunciar las 5 preguntas de negocio
- Definir las hipótesis H1–H4
- Listar los conceptos técnicos clave a rastrear: `IDE`, `terminal`, `vibe coding`, `AI`, `real developer`, `no code`, `cursor`, `TTS`, `transcription`

---

### Fase 2 — Data Understanding (Notebook Section 2)

**Extracción:**
- Usar `youtube-comment-downloader` con la URL del video
- Guardar en `data/raw/comments.json`
- Campos relevantes: `text`, `author`, `votes` (likes), `time`, `replies`

**EDA:**
- Distribución de longitud de comentarios (caracteres y tokens)
- Distribución temporal (¿hay picos de actividad?)
- Top 20 palabras más frecuentes (con y sin stopwords)
- Top comentarios por likes
- Detección de idioma predominante (la mayoría debería ser inglés)
- % de comentarios que mencionan conceptos clave

---

### Fase 3 — Data Preparation (Notebook Section 3)

**Limpieza:**
- Eliminar comentarios duplicados
- Eliminar comentarios vacíos o solo emojis
- Normalizar texto: lowercase, strip URLs, strip mentions (@user)
- Conservar emojis (VADER los usa para sentiment)

**Generación de embeddings:**
- Modelo: `all-MiniLM-L6-v2` via `sentence-transformers`
- Input: texto limpio de cada comentario
- Output: matriz `(N, 384)` — guardar en `embeddings.npy`
- Procesar en batches de 64 para eficiencia

---

### Fase 4 — Modeling (Notebook Section 4)

#### 4a. Sentiment Analysis

- **VADER** sobre cada comentario → score `compound` [-1, +1]
- Clasificar: positivo (>0.05), neutro (-0.05 a 0.05), negativo (<-0.05)
- **Aspect-based:** filtrar comentarios que contienen cada concepto clave y calcular sentimiento promedio por concepto
- Visualizar: pie chart general + bar chart por concepto

#### 4b. Reducción Dimensional y Clustering

1. **UMAP (10D)** sobre `embeddings.npy` → input para HDBSCAN
2. **UMAP (2D)** → input para visualización scatter
3. **HDBSCAN** sobre UMAP 10D:
   - `min_cluster_size=30` (ajustable)
   - `min_samples=10`
   - Comentarios con `label=-1` son ruido (no asignados)
4. Scatter plot 2D coloreado por cluster
5. Para cada cluster: mostrar top 10 comentarios más representativos (los más cercanos al centroide)
6. Etiquetar clusters manualmente + con ayuda del LLM local

**Clusters esperados a validar:**

| Label | Perfil esperado | Señal textual clave |
|-------|----------------|---------------------|
| 0 | Coder escéptico | "not real coding", "doesn't know", "IDE", "terminal" |
| 1 | Vibe coder entusiasta | "the future", "I built X", "no code needed", "amazing" |
| 2 | Fan neutral/casual | "great video", "love Riley", "so funny", "❤️" |
| 3 | Técnico curioso | "interesting", "TTS model", "transcription", "how does" |
| -1 | Ruido | Comentarios cortos sin señal clara |

#### 4c. Grafo de Co-ocurrencia de Conceptos

- Definir vocabulario de conceptos: `['IDE', 'terminal', 'vibe coding', 'AI', 'cursor', 'real developer', 'no code', 'TTS', 'transcription', 'coding', 'fear', 'easy', 'impossible', 'future']`
- Por cada comentario: detectar qué conceptos aparecen
- Construir grafo: nodo = concepto, arista = co-ocurrencia, peso = frecuencia
- Visualizar con NetworkX (tamaño de nodo = frecuencia, grosor de arista = peso)
- Análisis de centralidad: ¿qué concepto es el más conectado?

---

### Fase 5 — Evaluation (Notebook Section 5)

- Validar clusters contra hipótesis H1–H4
- Calcular métricas de clustering: Silhouette Score, DBCV (si disponible)
- Cruzar clusters con sentiment: ¿el cluster "coder escéptico" es mayormente negativo?
- Cruzar clusters con likes: ¿qué perfil recibe más upvotes?
- Conclusiones narrativas por hipótesis: CONFIRMADA / RECHAZADA / PARCIAL

---

### Fase 6 — Deployment: Mini RAG (Notebook Section 6)

**Arquitectura:**

```
[Query en lenguaje natural]
        ↓
[Embedding de la query] (all-MiniLM-L6-v2)
        ↓
[Cosine similarity vs embeddings.npy]
        ↓
[Top-K comentarios más relevantes (K=15)]
        ↓
[Prompt a qwen3:1.7b via ollama SDK]
        ↓
[Respuesta generada]
```

**Función RAG:**
```python
def ask(query: str, k: int = 15) -> str:
    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_idx = scores.argsort()[-k:][::-1]
    context = "\n".join(df.iloc[top_idx]['text'].tolist())
    prompt = f"""Based on these YouTube comments about vibe coding:

{context}

Answer this question: {query}

Be specific and cite patterns you see in the comments."""
    response = ollama.chat(model='qwen3:1.7b', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']
```

**Queries de demostración en el notebook:**
1. "What do professional developers think about vibe coding?"
2. "Are people surprised that Riley doesn't know what an IDE or terminal is?"
3. "Do people think AI tools will replace real programming skills?"
4. "What do non-technical people say about the video?"
5. "Is there a debate about whether vibe coding is real coding?"

---

## 4. Requirements

```
youtube-comment-downloader
sentence-transformers
umap-learn
hdbscan
vaderSentiment
networkx
matplotlib
seaborn
plotly
ollama
pandas
numpy
scikit-learn
jupyter
langdetect
```

---

## 5. Criterios de Éxito

- [ ] Se extraen los 2,838 comentarios correctamente
- [ ] Los embeddings se generan sin errores para todos los comentarios
- [ ] HDBSCAN produce entre 3 y 8 clusters con menos de 20% de ruido
- [ ] Los clusters son interpretables y mapeables a personas reales
- [ ] El grafo de conceptos muestra conexiones no triviales
- [ ] El mini RAG responde preguntas coherentemente con base en los comentarios
- [ ] El notebook está completamente documentado y es reproducible de principio a fin

---

## 6. Decisiones Técnicas Clave

| Decisión | Alternativa descartada | Razón |
|----------|----------------------|-------|
| VADER para sentiment | Transformer-based (DistilBERT) | Sin GPU, VADER es suficiente para texto social |
| HDBSCAN + UMAP | K-Means | Clusters naturales, no fuerza K |
| RAG manual con cosine similarity | LangChain / ChromaDB | Overkill para 2838 comentarios, menos transparente |
| qwen3:1.7b para RAG | gemma4 | Más rápido para queries interactivas en notebook |
| youtube-comment-downloader | YouTube Data API v3 | Sin API key requerida |
