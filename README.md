# Semantic Project Matching

## Emparejamiento Semántico en Dos Etapas  
**Bi-Encoder Retrieval + Cross-Encoder Re-Ranking**

Este proyecto implementa un sistema de matching semántico entre:

- Indicadores de producto de los PATR
- Catálogo estructurado de indicadores de los PDT (MGA / SisPT)

El objetivo es identificar, para cada indicador de producto de PATR, los indicadores de producto MGA más similares en significado utilizando modelos basados en transformers.

---

## Arquitectura del Modelo

El sistema sigue una arquitectura de recuperación en dos etapas.

### 1. Recuperación con Bi-Encoder

Cada texto se convierte en un embedding usando modelos como:

- `BAAI/bge-m3`
- `intfloat/multilingual-e5-large`

Los embeddings se normalizan y se calcula la similitud coseno entre estructura programática PATR e indicadores de producto MGA.

Luego se recuperan los Top-N candidatos más similares mediante k-Nearest Neighbors.

Esta etapa es:

- Rápida  
- Escalable  
- Adecuada para catálogos grandes  

---

### 2. Re-Ranking con Cross-Encoder (Opcional)

En esta etapa, un cross-encoder evalúa cada par indicadores producto PATR-indicador producto MGA de forma conjunta.

Modelo recomendado:

- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

Esto permite:

- Mayor precisión  
- Mejor ordenamiento final  
- Evaluación semántica profunda  

---

## ¿Por qué dos etapas?

| Bi-Encoder | Cross-Encoder |
|------------|--------------|
| Recuperación eficiente | Mayor precisión |
| Embeddings independientes | Evaluación conjunta |
| Escalable | Computacionalmente más costoso |

La combinación permite balancear eficiencia y calidad.

---

## Formato de Salida

El modelo genera una tabla en formato largo con las siguientes columnas:

- `matched_politica_text`
- `bi_similarity_score`
- `rerank_score`
- `rank`

Cada proyecto puede tener múltiples coincidencias ordenadas por relevancia.

---

## Parámetros Principales

- `top_k` — Número final de coincidencias por proyecto  
- `top_n_candidates` — Candidatos recuperados antes del re-ranking  
- `min_bi_score` — Umbral mínimo de similitud coseno  
- `use_rerank` — Activa o desactiva la segunda etapa  
- `model_name` — Modelo bi-encoder  
- `cross_encoder_name` — Modelo de re-ranking  

---

## Instalación

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar entorno:

Windows:
```bash
.venv\Scripts\activate
```

Mac / Linux:
```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Ejecución

Colocar archivos de entrada en:

```
data/raw/politicas.xlsx
data/raw/proyectos.xlsx
```

Ejecutar:

Windows:
```bash
$env:PYTHONPATH="src"
python scripts\run_matching.py
```

Mac / Linux:
```bash
PYTHONPATH=src python scripts/run_matching.py
```

Salida generada en:

```
data/outputs/matching_topk.xlsx
```

---

## Stack Tecnológico

- Python  
- Sentence Transformers  
- PyTorch  
- scikit-learn  
- Pandas  

---

## Licencia

MIT License
