# YouTube Comments CRISP-DM Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir un Jupyter Notebook que analiza 2,838 comentarios del video de vibe coding de Riley usando CRISP-DM: sentiment analysis, clustering de personas, grafo de conceptos y un mini RAG con qwen3:1.7b.

**Architecture:** Extracción sin API key → limpieza y embeddings locales → UMAP+HDBSCAN para detectar perfiles de audiencia → NetworkX para co-ocurrencia de conceptos → RAG directo sobre Ollama sin frameworks intermediarios.

**Tech Stack:** `youtube-comment-downloader`, `sentence-transformers`, `umap-learn`, `hdbscan`, `vaderSentiment`, `networkx`, `plotly`, `ollama`, `pandas`, `scikit-learn`

---

## Task 1: Setup del entorno

**Files:**
- Create: `requirements.txt`
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`

- [ ] **Step 1: Crear el venv e instalar dependencias**

Desde el directorio raíz del proyecto:

```bash
python -m venv .venv
.venv\Scripts\activate
```

- [ ] **Step 2: Crear requirements.txt**

```
youtube-comment-downloader==0.0.22
sentence-transformers==3.4.1
umap-learn==0.5.7
hdbscan==0.8.40
vaderSentiment==3.3.2
networkx==3.4.2
matplotlib==3.10.1
seaborn==0.13.2
plotly==6.0.1
ollama==0.4.7
pandas==2.2.3
numpy==2.2.4
scikit-learn==1.6.1
jupyter==1.1.1
langdetect==1.0.9
ipykernel==6.29.5
nbformat==5.10.4
```

- [ ] **Step 3: Instalar**

```bash
pip install -r requirements.txt
```

Verificar que no haya errores de compilación en `hdbscan` (requiere compilador C en Windows; si falla, instalar con `pip install hdbscan --no-build-isolation`).

- [ ] **Step 4: Registrar el kernel en Jupyter**

```bash
python -m ipykernel install --user --name=vibe-analysis --display-name "Vibe Analysis"
```

- [ ] **Step 5: Verificar Ollama**

```bash
ollama list
```

Expected output debe incluir `qwen3:1.7b`. Si no está:
```bash
ollama pull qwen3:1.7b
```

---

## Task 2: Extracción de comentarios

**Files:**
- Create: `data/raw/comments.json`

- [ ] **Step 1: Obtener la URL del video**

Abrir el video de Riley "Can I Vibecode a $250M App Better Than a Pro Developer?" en YouTube y copiar la URL completa. Ejemplo: `https://www.youtube.com/watch?v=XXXXXXXXXXX`

- [ ] **Step 2: Extraer comentarios**

Con el venv activado:

```bash
youtube-comment-downloader --youtubeid "XXXXXXXXXXX" --output data/raw/comments.json
```

Reemplazar `XXXXXXXXXXX` con el ID del video (la parte después de `?v=`).

- [ ] **Step 3: Verificar extracción**

```bash
python -c "
import json
with open('data/raw/comments.json') as f:
    lines = f.readlines()
print(f'Total comentarios extraídos: {len(lines)}')
first = json.loads(lines[0])
print('Keys disponibles:', list(first.keys()))
print('Ejemplo:', first['text'][:100])
"
```

Expected output:
```
Total comentarios extraídos: ~2838
Keys disponibles: ['id', 'text', 'time', 'author', 'channel', 'votes', 'replies', 'photo', 'heart', 'reply', 'time_parsed']
```

Nota: `comments.json` contiene un JSON por línea (JSONL), no un array.

---

## Task 3: Notebook — Setup y Sección 1 (Business Understanding)

**Files:**
- Create: `analysis.ipynb`

- [ ] **Step 1: Crear el notebook con la celda de configuración global**

Ejecutar en terminal:

```bash
jupyter notebook
```

Crear nuevo notebook llamado `analysis.ipynb`. Seleccionar kernel `Vibe Analysis`.

- [ ] **Step 2: Celda 1 — Imports y configuración global**

```python
# ============================================================
# YOUTUBE COMMENTS ANALYSIS — CRISP-DM
# Video: "Can I Vibecode a $250M App Better Than a Pro Developer?"
# Author: Riley | Comments: ~2,838
# ============================================================

import json
import re
import warnings
from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import ollama
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 120)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# Paths
DATA_RAW = Path('data/raw/comments.json')
DATA_CLEAN = Path('data/processed/comments_clean.csv')
EMBEDDINGS_PATH = Path('data/processed/embeddings.npy')
CLUSTERS_PATH = Path('data/processed/clusters.csv')

print("✓ Setup completo")
```

Ejecutar y verificar que imprime `✓ Setup completo` sin errores de import.

- [ ] **Step 3: Celda 2 — Business Understanding (Markdown)**

Agregar celda de tipo **Markdown**:

```markdown
# 📊 Fase 1: Business Understanding

## Contexto
Riley publicó "Can I Vibecode a $250M App Better Than a Pro Developer? (With No Code)".
El video muestra a un no-programador construyendo una app real usando herramientas de IA,
sin entender conceptos como IDE, terminal, o cómo funciona la transcripción del iPhone.

## Preguntas de Negocio
1. ¿Cuál es el sentimiento general hacia el video y hacia el vibe coding?
2. ¿Qué perfil tiene la persona que comenta? (coder, vibe-coder, casual, técnico)
3. ¿Las personas técnicas sienten amenaza/escepticismo ante el vibe coding?
4. ¿Las personas no técnicas se sorprenden de que Riley no entienda cosas básicas?
5. ¿Existe brecha entre "vibe coding es el futuro" vs "no es coding real"?

## Hipótesis
- **H1:** Hay coders escépticos que reaccionan negativamente a la ignorancia técnica de Riley
- **H2:** Hay vibe coders/fans entusiastas que celebran que "no necesitás saber código"
- **H3:** `IDE` y `terminal` generan comentarios de incredulidad o condescendencia
- **H4:** Los técnicos no tienen miedo a las herramientas, sino escepticismo sobre si es "real"
```

- [ ] **Step 4: Celda 3 — Conceptos clave a rastrear**

```python
# Vocabulario de conceptos técnicos a rastrear en el análisis
CONCEPTS = [
    'ide', 'terminal', 'vibe coding', 'vibe code', 'vibecod',
    'ai', 'artificial intelligence',
    'cursor', 'real developer', 'no code', 'nocode',
    'tts', 'transcription', 'coding', 'programmer',
    'fear', 'easy', 'impossible', 'future', 'replace',
    'cheat', 'shortcut', 'real coding', 'fake'
]

print(f"Monitoreando {len(CONCEPTS)} conceptos clave")
print(CONCEPTS)
```

---

## Task 4: Notebook — Sección 2 (Data Understanding / EDA)

**Files:**
- Modify: `analysis.ipynb` (agregar celdas de EDA)

- [ ] **Step 1: Celda — Cargar datos raw**

```python
# ============================================================
# Fase 2: Data Understanding
# ============================================================

# Cargar JSONL (un JSON por línea)
records = []
with open(DATA_RAW, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

df_raw = pd.DataFrame(records)
print(f"Total comentarios cargados: {len(df_raw)}")
print(f"\nColumnas: {list(df_raw.columns)}")
print(f"\nEjemplos:")
df_raw[['author', 'text', 'votes']].head(5)
```

Expected: `Total comentarios cargados: ~2838`

- [ ] **Step 2: Celda — Info general del dataset**

```python
print("=== Info del Dataset ===")
print(f"Rango de fechas: {df_raw['time'].iloc[-1]} → {df_raw['time'].iloc[0]}")
print(f"Comentarios con replies: {df_raw['replies'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()}")
print(f"Comentarios con likes > 0: {(df_raw['votes'] > 0).sum()} ({(df_raw['votes'] > 0).mean():.1%})")
print(f"Comentario más likeado: {df_raw['votes'].max()} likes")
print(f"\nTop 5 comentarios por likes:")
df_raw.nlargest(5, 'votes')[['author', 'text', 'votes']]
```

- [ ] **Step 3: Celda — Distribución de longitud**

```python
df_raw['char_len'] = df_raw['text'].str.len()
df_raw['word_len'] = df_raw['text'].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_raw['char_len'].clip(upper=500), bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Distribución de longitud (caracteres)')
axes[0].set_xlabel('Caracteres')
axes[0].set_ylabel('Frecuencia')
axes[0].axvline(df_raw['char_len'].median(), color='red', linestyle='--', label=f'Mediana: {df_raw["char_len"].median():.0f}')
axes[0].legend()

axes[1].hist(df_raw['word_len'].clip(upper=80), bins=40, color='coral', edgecolor='white')
axes[1].set_title('Distribución de longitud (palabras)')
axes[1].set_xlabel('Palabras')
axes[1].set_ylabel('Frecuencia')
axes[1].axvline(df_raw['word_len'].median(), color='navy', linestyle='--', label=f'Mediana: {df_raw["word_len"].median():.0f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/processed/dist_longitud.png', bbox_inches='tight')
plt.show()

print(f"Mediana: {df_raw['char_len'].median():.0f} chars | {df_raw['word_len'].median():.0f} words")
print(f"Comentarios muy cortos (<10 chars): {(df_raw['char_len'] < 10).sum()}")
print(f"Comentarios largos (>300 chars): {(df_raw['char_len'] > 300).sum()}")
```

- [ ] **Step 4: Celda — Top palabras frecuentes**

```python
from collections import Counter
import re

STOPWORDS = {
    'the', 'a', 'an', 'is', 'it', 'in', 'to', 'of', 'and', 'or', 'i',
    'this', 'that', 'for', 'you', 'he', 'she', 'they', 'we', 'with',
    'as', 'at', 'be', 'was', 'are', 'have', 'has', 'had', 'but', 'not',
    'do', 'did', 'so', 'if', 'by', 'on', 'your', 'my', 'his', 'her',
    'its', 'just', 'can', 'will', 'would', 'could', 'like', 'what',
    'when', 'how', 'who', 'there', 'here', 'from', 'all', 'one', 'no',
    'up', 'out', 'about', 'get', 'got', 'into', 'more', 'even', 'than',
    'me', 'him', 'them', 'us', 'been', 'were', 'their', 'our', 'im',
    'ive', 'dont', 'doesnt', 'cant', 'its', 'hes', 'shes', 'thats'
}

all_words = []
for text in df_raw['text']:
    words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
    all_words.extend([w for w in words if w not in STOPWORDS])

word_freq = Counter(all_words).most_common(30)
words_list, counts = zip(*word_freq)

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(list(reversed(words_list[:20])), list(reversed(counts[:20])), color='steelblue')
ax.set_title('Top 20 palabras más frecuentes (sin stopwords)', fontsize=14)
ax.set_xlabel('Frecuencia')
for bar, count in zip(bars, reversed(counts[:20])):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(count), va='center')
plt.tight_layout()
plt.savefig('data/processed/top_palabras.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 5: Celda — Cobertura de conceptos clave**

```python
concept_coverage = {}
for concept in CONCEPTS:
    mask = df_raw['text'].str.lower().str.contains(concept, na=False)
    concept_coverage[concept] = mask.sum()

coverage_df = pd.DataFrame.from_dict(concept_coverage, orient='index', columns=['menciones'])
coverage_df = coverage_df[coverage_df['menciones'] > 0].sort_values('menciones', ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(coverage_df.index, coverage_df['menciones'], color='darkorange', edgecolor='white')
ax.set_title('Menciones de conceptos técnicos clave en comentarios', fontsize=14)
ax.set_ylabel('Número de comentarios')
plt.xticks(rotation=45, ha='right')
for bar, val in zip(bars, coverage_df['menciones']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('data/processed/concept_coverage.png', bbox_inches='tight')
plt.show()

print(coverage_df)
```

---

## Task 5: Notebook — Sección 3a (Data Preparation: Limpieza)

**Files:**
- Modify: `analysis.ipynb`
- Create: `data/processed/comments_clean.csv`

- [ ] **Step 1: Celda — Función de limpieza**

```python
# ============================================================
# Fase 3: Data Preparation
# ============================================================

def clean_text(text: str) -> str:
    """
    Limpia el texto del comentario preservando emojis (VADER los usa).
    - Elimina URLs
    - Elimina menciones @usuario
    - Normaliza espacios múltiples
    - NO elimina emojis
    - NO convierte a lowercase (VADER es sensible a mayúsculas)
    """
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)           # URLs
    text = re.sub(r'@\w+', '', text)                      # menciones
    text = re.sub(r'\s+', ' ', text).strip()              # espacios múltiples
    return text

def is_meaningful(text: str) -> bool:
    """Retorna True si el comentario tiene contenido sustancial."""
    stripped = re.sub(r'[^\w]', '', text)  # quitar todo menos alfanuméricos
    return len(stripped) >= 10             # al menos 10 caracteres "reales"

# Test de la función
sample = "  Check this out @user https://t.co/abc123   so cool!!  🔥  "
print(f"Original: {repr(sample)}")
print(f"Limpio:   {repr(clean_text(sample))}")
print(f"Meaningful: {is_meaningful(clean_text(sample))}")
```

Expected:
```
Original: '  Check this out @user https://t.co/abc123   so cool!!  🔥  '
Limpio:   'so cool!!  🔥'
Meaningful: True
```

- [ ] **Step 2: Celda — Aplicar limpieza y filtrar**

```python
df = df_raw.copy()
df['text_clean'] = df['text'].apply(clean_text)
df['meaningful'] = df['text_clean'].apply(is_meaningful)

print(f"Comentarios originales: {len(df)}")
print(f"Comentarios sin sentido (muy cortos): {(~df['meaningful']).sum()}")

# Filtrar
df = df[df['meaningful']].copy()
df = df.drop_duplicates(subset='text_clean').reset_index(drop=True)

print(f"Comentarios después de limpieza: {len(df)}")
print(f"\nEjemplo limpieza:")
print(df[['text', 'text_clean']].head(3).to_string())
```

- [ ] **Step 3: Celda — Guardar CSV limpio**

```python
df[['author', 'text', 'text_clean', 'votes', 'time', 'replies']].to_csv(DATA_CLEAN, index=False)
print(f"✓ Guardado en {DATA_CLEAN}")
print(f"Shape: {df.shape}")
```

---

## Task 6: Notebook — Sección 3b (Data Preparation: Embeddings)

**Files:**
- Modify: `analysis.ipynb`
- Create: `data/processed/embeddings.npy`

- [ ] **Step 1: Celda — Cargar modelo y generar embeddings**

```python
print("Cargando modelo sentence-transformers...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"✓ Modelo cargado. Dimensión de embedding: {embed_model.get_sentence_embedding_dimension()}")

print(f"\nGenerando embeddings para {len(df)} comentarios...")
print("(Esto puede tardar 1-3 minutos en CPU)")

embeddings = embed_model.encode(
    df['text_clean'].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # normalizar para cosine similarity eficiente
)

print(f"\n✓ Embeddings generados")
print(f"Shape: {embeddings.shape}")  # Expected: (N, 384)
print(f"Dtype: {embeddings.dtype}")
```

Expected shape: `(~2800, 384)`

- [ ] **Step 2: Celda — Guardar embeddings**

```python
np.save(EMBEDDINGS_PATH, embeddings)
print(f"✓ Embeddings guardados en {EMBEDDINGS_PATH}")
print(f"Tamaño archivo: {EMBEDDINGS_PATH.stat().st_size / 1024:.1f} KB")
```

---

## Task 7: Notebook — Sección 4a (Sentiment Analysis)

**Files:**
- Modify: `analysis.ipynb`

- [ ] **Step 1: Celda — Calcular sentiment con VADER**

```python
# ============================================================
# Fase 4: Modeling — 4a. Sentiment Analysis
# ============================================================

sia = SentimentIntensityAnalyzer()

def get_sentiment(text: str) -> dict:
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = 'positive'
    elif compound <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return {'compound': compound, 'sentiment': label,
            'pos': scores['pos'], 'neg': scores['neg'], 'neu': scores['neu']}

# Aplicar a todo el dataset
sentiment_data = df['text_clean'].apply(get_sentiment).apply(pd.Series)
df = pd.concat([df, sentiment_data], axis=1)

print("Distribución de sentimiento:")
print(df['sentiment'].value_counts())
print(f"\nSentimiento promedio (compound): {df['compound'].mean():.3f}")
```

- [ ] **Step 2: Celda — Visualizar sentimiento general**

```python
sent_counts = df['sentiment'].value_counts()
colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
axes[0].pie(
    sent_counts.values,
    labels=sent_counts.index,
    colors=[colors[s] for s in sent_counts.index],
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 12}
)
axes[0].set_title('Distribución de Sentimiento General', fontsize=14)

# Histogram del compound score
axes[1].hist(df['compound'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(0.05, color='green', linestyle='--', alpha=0.7, label='Umbral positivo')
axes[1].axvline(-0.05, color='red', linestyle='--', alpha=0.7, label='Umbral negativo')
axes[1].axvline(df['compound'].mean(), color='orange', linewidth=2, label=f'Media: {df["compound"].mean():.3f}')
axes[1].set_title('Distribución del Compound Score (VADER)', fontsize=14)
axes[1].set_xlabel('Compound Score [-1, +1]')
axes[1].set_ylabel('Frecuencia')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/processed/sentiment_general.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 3: Celda — Aspect-based sentiment por concepto**

```python
# Para cada concepto: filtrar comentarios que lo mencionan y calcular sentiment promedio
aspect_results = []
for concept in CONCEPTS:
    mask = df['text_clean'].str.lower().str.contains(concept, na=False)
    subset = df[mask]
    if len(subset) >= 5:  # al menos 5 menciones para ser estadísticamente relevante
        aspect_results.append({
            'concept': concept,
            'count': len(subset),
            'mean_compound': subset['compound'].mean(),
            'pct_positive': (subset['sentiment'] == 'positive').mean(),
            'pct_negative': (subset['sentiment'] == 'negative').mean(),
        })

aspect_df = pd.DataFrame(aspect_results).sort_values('mean_compound', ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
bar_colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in aspect_df['mean_compound']]
bars = ax.barh(aspect_df['concept'], aspect_df['mean_compound'], color=bar_colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
ax.set_title('Sentimiento Promedio por Concepto Técnico (Aspect-Based)', fontsize=14)
ax.set_xlabel('Compound Score promedio')

# Agregar count
for bar, count in zip(bars, aspect_df['count']):
    x = bar.get_width()
    offset = 0.005 if x >= 0 else -0.005
    ax.text(x + offset, bar.get_y() + bar.get_height()/2,
            f'n={count}', va='center', fontsize=9,
            ha='left' if x >= 0 else 'right')

plt.tight_layout()
plt.savefig('data/processed/aspect_sentiment.png', bbox_inches='tight')
plt.show()
print(aspect_df.to_string(index=False))
```

---

## Task 8: Notebook — Sección 4b (Clustering: UMAP + HDBSCAN)

**Files:**
- Modify: `analysis.ipynb`
- Create: `data/processed/clusters.csv`

- [ ] **Step 1: Celda — UMAP 10D para clustering**

```python
# ============================================================
# Fase 4b: Clustering — UMAP + HDBSCAN
# ============================================================

print("Calculando UMAP 10D para clustering...")
reducer_10d = umap.UMAP(
    n_components=10,
    n_neighbors=20,
    min_dist=0.0,        # min_dist=0 mejora clustering (puntos más compactos)
    metric='cosine',
    random_state=42,
    verbose=True
)
embedding_10d = reducer_10d.fit_transform(embeddings)
print(f"✓ UMAP 10D shape: {embedding_10d.shape}")
```

- [ ] **Step 2: Celda — UMAP 2D para visualización**

```python
print("Calculando UMAP 2D para visualización...")
reducer_2d = umap.UMAP(
    n_components=2,
    n_neighbors=20,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
    verbose=True
)
embedding_2d = reducer_2d.fit_transform(embeddings)
print(f"✓ UMAP 2D shape: {embedding_2d.shape}")
```

- [ ] **Step 3: Celda — HDBSCAN clustering**

```python
print("Ejecutando HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='eom'
)
cluster_labels = clusterer.fit_predict(embedding_10d)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = (cluster_labels == -1).sum()

print(f"✓ Clusters encontrados: {n_clusters}")
print(f"  Comentarios por cluster:")
for label in sorted(set(cluster_labels)):
    name = "RUIDO" if label == -1 else f"Cluster {label}"
    count = (cluster_labels == label).sum()
    pct = count / len(cluster_labels) * 100
    print(f"  {name}: {count} comentarios ({pct:.1f}%)")

# Si n_clusters < 2 o noise > 50%, ajustar parámetros
if n_clusters < 2:
    print("\n⚠️  Muy pocos clusters. Reducir min_cluster_size a 15 y re-ejecutar.")
if n_noise / len(cluster_labels) > 0.5:
    print("\n⚠️  Demasiado ruido. Reducir min_samples a 5 y re-ejecutar.")
```

- [ ] **Step 4: Celda — Agregar clusters al dataframe y guardar**

```python
df['cluster'] = cluster_labels
df['umap_x'] = embedding_2d[:, 0]
df['umap_y'] = embedding_2d[:, 1]

df.to_csv(CLUSTERS_PATH, index=False)
print(f"✓ Dataset con clusters guardado en {CLUSTERS_PATH}")
```

- [ ] **Step 5: Celda — Scatter plot 2D interactivo (Plotly)**

```python
# Colores para clusters (-1 = gris, resto = paleta)
palette = px.colors.qualitative.Set2
color_map = {-1: '#cccccc'}
for i, label in enumerate(sorted([l for l in set(cluster_labels) if l != -1])):
    color_map[label] = palette[i % len(palette)]

df['cluster_color'] = df['cluster'].map(color_map)
df['cluster_str'] = df['cluster'].apply(lambda x: 'Ruido' if x == -1 else f'Cluster {x}')

fig = px.scatter(
    df,
    x='umap_x', y='umap_y',
    color='cluster_str',
    hover_data={'text_clean': True, 'sentiment': True, 'votes': True, 'umap_x': False, 'umap_y': False},
    title='Clusters de Comentarios (UMAP 2D + HDBSCAN)',
    opacity=0.6,
    width=900, height=600
)
fig.update_traces(marker=dict(size=4))
fig.show()
```

- [ ] **Step 6: Celda — Comentarios representativos por cluster**

```python
def get_representative_comments(cluster_id: int, n: int = 10) -> pd.DataFrame:
    """Retorna los N comentarios más cercanos al centroide del cluster."""
    mask = df['cluster'] == cluster_id
    cluster_embs = embeddings[mask]
    centroid = cluster_embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(centroid, cluster_embs)[0]
    top_idx = sims.argsort()[-n:][::-1]
    return df[mask].iloc[top_idx][['text_clean', 'votes', 'sentiment', 'compound']]

for label in sorted([l for l in set(cluster_labels) if l != -1]):
    print(f"\n{'='*60}")
    print(f"CLUSTER {label} — Top 10 comentarios más representativos")
    print('='*60)
    reps = get_representative_comments(label)
    for _, row in reps.iterrows():
        print(f"[{row['sentiment']:8s} | {row['compound']:+.2f} | 👍{row['votes']}] {row['text_clean'][:120]}")
```

- [ ] **Step 7: Celda — Etiquetar clusters con ayuda del LLM**

```python
def label_cluster_with_llm(cluster_id: int) -> str:
    """Usa qwen3:1.7b para sugerir una etiqueta/persona para el cluster."""
    reps = get_representative_comments(cluster_id, n=15)
    sample_texts = "\n".join([f"- {t[:150]}" for t in reps['text_clean'].tolist()])

    prompt = f"""You are analyzing YouTube comments about a video where a non-programmer builds an app with AI tools (vibe coding).

These are {len(df[df['cluster'] == cluster_id])} comments from the same cluster (similar comments grouped together):

{sample_texts}

Based on these comments, answer:
1. What type of person is writing these comments? (programmer, vibe coder, casual fan, etc.)
2. What is their attitude toward vibe coding and AI tools?
3. Suggest a short label for this persona (e.g., "Skeptical Developer", "Enthusiastic Vibe Coder")

Be concise. Max 100 words."""

    response = ollama.chat(
        model='qwen3:1.7b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

# Ejecutar para cada cluster
cluster_labels_dict = {}
for label in sorted([l for l in set(cluster_labels) if l != -1]):
    print(f"\n--- Cluster {label} ---")
    description = label_cluster_with_llm(label)
    cluster_labels_dict[label] = description
    print(description)
```

- [ ] **Step 8: Celda Markdown — Tabla resumen de clusters**

Después de ver las respuestas del LLM, agregar celda Markdown con la tabla de personas identificadas:

```markdown
## Resumen de Clusters Identificados

| Cluster | Tamaño | Persona | Sentimiento Dominante |
|---------|--------|---------|----------------------|
| 0 | _N_ | _Label del LLM_ | _positive/negative/neutral_ |
| 1 | _N_ | _Label del LLM_ | _positive/negative/neutral_ |
| ... | ... | ... | ... |
| -1 | _N_ | Ruido (sin asignar) | — |

> Completar con los resultados reales del análisis
```

---

## Task 9: Notebook — Sección 4c (Grafo de Co-ocurrencia)

**Files:**
- Modify: `analysis.ipynb`

- [ ] **Step 1: Celda — Construir grafo de co-ocurrencia**

```python
# ============================================================
# Fase 4c: Grafo de Co-ocurrencia de Conceptos
# ============================================================

def extract_concepts_from_text(text: str) -> list[str]:
    """Detecta qué conceptos del vocabulario aparecen en el texto."""
    text_lower = text.lower()
    found = []
    for concept in CONCEPTS:
        if concept in text_lower:
            found.append(concept)
    return found

# Construir lista de conceptos por comentario
df['concepts_found'] = df['text_clean'].apply(extract_concepts_from_text)

# Calcular frecuencia de cada concepto
concept_freq = Counter()
for concepts in df['concepts_found']:
    concept_freq.update(concepts)

# Calcular co-ocurrencias (pares de conceptos en el mismo comentario)
co_occurrence = Counter()
for concepts in df['concepts_found']:
    if len(concepts) >= 2:
        for pair in combinations(sorted(concepts), 2):
            co_occurrence[pair] += 1

print(f"Nodos (conceptos con al menos 1 aparición): {len([c for c, n in concept_freq.items() if n > 0])}")
print(f"Aristas (pares que co-ocurren): {len(co_occurrence)}")
print(f"\nTop 10 conceptos más frecuentes:")
for concept, count in concept_freq.most_common(10):
    print(f"  {concept}: {count}")
print(f"\nTop 10 co-ocurrencias más frecuentes:")
for (c1, c2), count in co_occurrence.most_common(10):
    print(f"  {c1} ↔ {c2}: {count}")
```

- [ ] **Step 2: Celda — Construir y visualizar el grafo**

```python
# Construir grafo con NetworkX
G = nx.Graph()

# Agregar nodos con atributo de frecuencia
for concept, freq in concept_freq.items():
    if freq >= 3:  # solo conceptos con al menos 3 menciones
        G.add_node(concept, frequency=freq)

# Agregar aristas con peso de co-ocurrencia
for (c1, c2), weight in co_occurrence.items():
    if weight >= 2 and G.has_node(c1) and G.has_node(c2):
        G.add_edge(c1, c2, weight=weight)

print(f"Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

# Layout
pos = nx.spring_layout(G, k=2.5, seed=42, weight='weight')

# Escalar nodos por frecuencia
node_sizes = [concept_freq[n] * 8 for n in G.nodes()]
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
max_weight = max(edge_weights) if edge_weights else 1
edge_widths = [w / max_weight * 6 for w in edge_weights]

# Colorear nodos por centralidad de grado
centrality = nx.degree_centrality(G)
node_colors = [centrality[n] for n in G.nodes()]

fig, ax = plt.subplots(figsize=(14, 10))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                       cmap=plt.cm.YlOrRd, alpha=0.9, ax=ax)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color='gray', ax=ax)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
ax.set_title('Grafo de Co-ocurrencia de Conceptos Técnicos en Comentarios', fontsize=14, pad=20)
ax.axis('off')

# Leyenda
sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=1))
plt.colorbar(sm, ax=ax, label='Centralidad de Grado', shrink=0.5)

plt.tight_layout()
plt.savefig('data/processed/concept_graph.png', bbox_inches='tight', dpi=150)
plt.show()
```

- [ ] **Step 3: Celda — Análisis de centralidad**

```python
print("=== Análisis de Centralidad ===\n")

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=500)

centrality_df = pd.DataFrame({
    'concept': list(G.nodes()),
    'frequency': [concept_freq[n] for n in G.nodes()],
    'degree_centrality': [degree_centrality[n] for n in G.nodes()],
    'betweenness': [betweenness_centrality[n] for n in G.nodes()],
    'eigenvector': [eigenvector_centrality[n] for n in G.nodes()],
}).sort_values('degree_centrality', ascending=False)

print("Top 10 conceptos por centralidad de grado (más conectados):")
print(centrality_df.head(10).to_string(index=False))
```

---

## Task 10: Notebook — Sección 5 (Evaluation)

**Files:**
- Modify: `analysis.ipynb`

- [ ] **Step 1: Celda — Métricas de clustering**

```python
# ============================================================
# Fase 5: Evaluation
# ============================================================

# Silhouette score (excluir ruido)
non_noise_mask = df['cluster'] != -1
if non_noise_mask.sum() > 1:
    sil_score = silhouette_score(
        embeddings[non_noise_mask],
        df.loc[non_noise_mask, 'cluster'],
        metric='cosine',
        sample_size=1000,  # sample para velocidad
        random_state=42
    )
    print(f"Silhouette Score: {sil_score:.4f}")
    print("  (rango -1 a 1; >0.3 es aceptable para texto)")
else:
    print("No hay suficientes puntos no-ruido para calcular Silhouette.")

# Resumen estadístico por cluster
cluster_summary = df[df['cluster'] != -1].groupby('cluster').agg(
    tamaño=('text_clean', 'count'),
    sentiment_promedio=('compound', 'mean'),
    pct_positivo=('sentiment', lambda x: (x == 'positive').mean()),
    pct_negativo=('sentiment', lambda x: (x == 'negative').mean()),
    likes_promedio=('votes', 'mean'),
    likes_max=('votes', 'max'),
).round(3)

print("\nResumen por Cluster:")
print(cluster_summary.to_string())
```

- [ ] **Step 2: Celda — Cruce cluster × sentiment (heatmap)**

```python
cross = pd.crosstab(
    df[df['cluster'] != -1]['cluster'],
    df[df['cluster'] != -1]['sentiment'],
    normalize='index'
) * 100

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(cross, annot=True, fmt='.1f', cmap='RdYlGn',
            linewidths=0.5, ax=ax, cbar_kws={'label': '% comentarios'})
ax.set_title('Distribución de Sentimiento por Cluster (%)', fontsize=13)
ax.set_xlabel('Sentimiento')
ax.set_ylabel('Cluster')
plt.tight_layout()
plt.savefig('data/processed/cluster_sentiment_heatmap.png', bbox_inches='tight')
plt.show()
```

- [ ] **Step 3: Celda — Validación de hipótesis**

```python
print("=" * 60)
print("VALIDACIÓN DE HIPÓTESIS")
print("=" * 60)

# H1: Cluster de coders escépticos con sentimiento negativo
# H2: Cluster de vibe coders entusiastas con sentimiento positivo
# H3: Conceptos IDE y terminal con sentimiento negativo/incredulidad
# H4: Técnicos escépticos, no con miedo sino cuestionamiento de autenticidad

# Extraer el compound promedio para IDE y terminal
for concept in ['ide', 'terminal', 'vibe coding', 'real developer', 'replace']:
    mask = df['text_clean'].str.lower().str.contains(concept, na=False)
    if mask.sum() >= 5:
        avg = df[mask]['compound'].mean()
        pct_neg = (df[mask]['sentiment'] == 'negative').mean() * 100
        print(f"\n'{concept}' ({mask.sum()} comentarios):")
        print(f"  Sentiment promedio: {avg:+.3f}")
        print(f"  % negativos: {pct_neg:.1f}%")

print("\n" + "=" * 60)
print("CONCLUSIONES NARRATIVAS")
print("=" * 60)
print("""
H1: [PENDIENTE — completar con resultados]
H2: [PENDIENTE — completar con resultados]
H3: [PENDIENTE — completar con resultados]
H4: [PENDIENTE — completar con resultados]
""")
```

Nota: Reemplazar los `[PENDIENTE]` con las conclusiones reales basadas en los números obtenidos.

---

## Task 11: Notebook — Sección 6 (Mini RAG)

**Files:**
- Modify: `analysis.ipynb`

- [ ] **Step 1: Celda — Función RAG**

```python
# ============================================================
# Fase 6: Deployment — Mini RAG con qwen3:1.7b
# ============================================================

# Los embeddings y el modelo ya están cargados en memoria.
# Si ejecutás esta sección por separado, descomentar:
# embeddings = np.load(EMBEDDINGS_PATH)
# df = pd.read_csv(CLUSTERS_PATH)
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def ask(query: str, k: int = 15, model: str = 'qwen3:1.7b') -> str:
    """
    Mini RAG: recupera los K comentarios más relevantes para la query
    y los envía como contexto a qwen3:1.7b vía Ollama.

    Args:
        query: Pregunta en lenguaje natural
        k: Número de comentarios a recuperar (default 15)
        model: Modelo Ollama a usar

    Returns:
        Respuesta generada por el modelo
    """
    # 1. Embed la query
    query_emb = embed_model.encode([query], normalize_embeddings=True)

    # 2. Cosine similarity (embeddings ya normalizados → producto punto = cosine)
    scores = (embeddings @ query_emb.T).flatten()

    # 3. Top-K índices
    top_idx = scores.argsort()[-k:][::-1]

    # 4. Construir contexto
    retrieved = df.iloc[top_idx][['text_clean', 'sentiment', 'cluster']].copy()
    context_lines = []
    for _, row in retrieved.iterrows():
        cluster_info = f"cluster={row['cluster']}" if row['cluster'] != -1 else "unassigned"
        context_lines.append(f"[{row['sentiment']} | {cluster_info}] {row['text_clean']}")
    context = "\n".join(context_lines)

    # 5. Prompt
    prompt = f"""You are analyzing YouTube comments from a video about "vibe coding" — where a non-programmer builds an app using AI tools without understanding concepts like IDE, terminal, or transcription APIs.

Here are {k} relevant comments retrieved for this query:

{context}

---

Question: {query}

Answer based specifically on these comments. Identify patterns, quote or paraphrase 2-3 specific comments to support your answer. Be analytical and specific."""

    # 6. Llamar a Ollama
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.3}  # temperatura baja para respuestas más factuales
    )
    return response['message']['content']

print("✓ Función ask() lista")
print("Uso: ask('your question here')")
```

- [ ] **Step 2: Celda — Verificar conexión con Ollama**

```python
# Test rápido de conectividad
try:
    test = ollama.chat(
        model='qwen3:1.7b',
        messages=[{'role': 'user', 'content': 'Say "OK" and nothing else.'}],
        options={'temperature': 0}
    )
    print(f"✓ Ollama conectado. Respuesta de prueba: {test['message']['content']}")
except Exception as e:
    print(f"✗ Error conectando a Ollama: {e}")
    print("Asegurate de que Ollama esté corriendo: ejecutá 'ollama serve' en otra terminal")
```

- [ ] **Step 3: Celda — Query 1**

```python
print("Q1: What do professional developers think about vibe coding?")
print("-" * 60)
print(ask("What do professional developers think about vibe coding?"))
```

- [ ] **Step 4: Celda — Query 2**

```python
print("Q2: Are people surprised that Riley doesn't know what an IDE or terminal is?")
print("-" * 60)
print(ask("Are people surprised that Riley doesn't know what an IDE or terminal is?"))
```

- [ ] **Step 5: Celda — Query 3**

```python
print("Q3: Do people think AI tools will replace real programming skills?")
print("-" * 60)
print(ask("Do people think AI tools will replace real programming skills?"))
```

- [ ] **Step 6: Celda — Query 4**

```python
print("Q4: What do non-technical people say about the video?")
print("-" * 60)
print(ask("What do non-technical people say about the video?"))
```

- [ ] **Step 7: Celda — Query 5**

```python
print("Q5: Is there a debate about whether vibe coding is real coding?")
print("-" * 60)
print(ask("Is there a debate about whether vibe coding is real coding?"))
```

- [ ] **Step 8: Celda — Modo interactivo**

```python
# Modo interactivo: escribí tus propias preguntas
while True:
    query = input("\n🔍 Tu pregunta (o 'exit' para salir): ").strip()
    if query.lower() in ('exit', 'quit', 'q', ''):
        break
    print("\n" + ask(query))
    print()
```

---

## Self-Review del Plan

**Spec coverage:**
- ✅ Fase 1 Business Understanding → Task 3
- ✅ Extracción youtube-comment-downloader → Task 2
- ✅ EDA (longitud, frecuencias, conceptos) → Task 4
- ✅ Limpieza y CSV → Task 5
- ✅ Embeddings all-MiniLM-L6-v2 → Task 6
- ✅ VADER sentiment + aspect-based → Task 7
- ✅ UMAP 10D + UMAP 2D + HDBSCAN → Task 8
- ✅ Representantes por cluster + etiquetado LLM → Task 8
- ✅ Grafo NetworkX + centralidad → Task 9
- ✅ Silhouette + cruce cluster×sentiment → Task 10
- ✅ Validación H1-H4 → Task 10
- ✅ Mini RAG con cosine similarity + Ollama → Task 11
- ✅ 5 queries de demostración → Task 11

**Placeholder scan:** Las conclusiones H1-H4 en Task 10 son intencionalmente marcadas como `[PENDIENTE]` porque dependen de los datos reales. No son placeholders de código, son notas para el analista.

**Type consistency:** `embed_model`, `embeddings`, `df`, `CONCEPTS`, `cluster_labels` se definen en Tasks 3/6/8 y se referencian consistentemente en Tasks 7/9/11.
