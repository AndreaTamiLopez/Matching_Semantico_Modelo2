from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.neighbors import NearestNeighbors

from .utils import clean_text, format_for_model


def semantic_project_merge_advanced(
    df_politicas: pd.DataFrame,
    df_proyectos: pd.DataFrame,
    col_text_politica: str,
    col_text_proyecto: str,
    col_id_proyecto: str,
    top_k: int = 3,
    min_bi_score: float = 0.45,
    top_n_candidates: int = 50,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 256,
    device: str = "cpu",
    use_rerank: bool = True,
    cross_encoder_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    rerank_keep_top_k: Optional[int] = None,
) -> pd.DataFrame:
    """
    Matching semántico PROYECTO -> POLÍTICAS en dos etapas:

    1) Bi-encoder retrieval:
       - Genera embeddings para políticas y proyectos (normalize_embeddings=True)
       - Recupera top_n_candidates por similitud coseno

    2) Cross-encoder re-ranking (opcional):
       - Recalcula relevancia sobre los candidatos
       - Mejora precisión en la selección final

    Output:
      DataFrame "long" con hasta top_k (o rerank_keep_top_k) matches por proyecto.

    Columnas de salida:
      - columnas del proyecto (se copian completas)
      - matched_politica_text
      - bi_similarity_score (coseno, 0-1)
      - rerank_score (score del cross-encoder; escala del modelo)
      - rank (1..K)
    """

    # -----------------------------
    # Validaciones
    # -----------------------------
    if col_text_politica not in df_politicas.columns:
        raise ValueError(f"df_politicas no tiene la columna: {col_text_politica}")
    for c in [col_text_proyecto, col_id_proyecto]:
        if c not in df_proyectos.columns:
            raise ValueError(f"df_proyectos no tiene la columna: {c}")

    if len(df_politicas) == 0 or len(df_proyectos) == 0:
        return pd.DataFrame()

    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k debe ser > 0")

    top_n_candidates = max(top_k, int(top_n_candidates))
    n_neighbors = min(top_n_candidates, len(df_politicas))

    if rerank_keep_top_k is None:
        rerank_keep_top_k = top_k
    rerank_keep_top_k = int(rerank_keep_top_k)

    # -----------------------------
    # 1) Textos limpios
    # -----------------------------
    pol_texts_raw = (
        df_politicas[col_text_politica].fillna("").astype(str).map(clean_text).tolist()
    )
    proy_texts_raw = (
        df_proyectos[col_text_proyecto].fillna("").astype(str).map(clean_text).tolist()
    )

    # -----------------------------
    # 2) Bi-encoder embeddings
    # -----------------------------
    bi = SentenceTransformer(model_name, device=device)

    pol_texts = format_for_model(pol_texts_raw, mode="query", model_name=model_name)
    proy_texts = format_for_model(proy_texts_raw, mode="passage", model_name=model_name)

    E_pol = bi.encode(
        pol_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    E_proy = bi.encode(
        proy_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # -----------------------------
    # 3) Retrieval Top-N por coseno
    # -----------------------------
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(E_pol)

    distances, indices = nn.kneighbors(E_proy, return_distance=True)
    bi_scores = 1.0 - distances  # coseno (0..1 aprox)

    # -----------------------------
    # 4) Cross-encoder re-ranking (opcional)
    # -----------------------------
    ce = CrossEncoder(cross_encoder_name, device=device) if use_rerank else None

    out_rows: List[Dict[str, Any]] = []

    for i in range(len(df_proyectos)):
        base = df_proyectos.iloc[i].to_dict()

        # candidatos filtrados por bi-score
        cand_pairs = []
        cand_meta = []  # (index_politica, bi_score)
        for r in range(n_neighbors):
            j = int(indices[i, r])
            sc_bi = float(bi_scores[i, r])
            if sc_bi < float(min_bi_score):
                continue
            cand_pairs.append([pol_texts_raw[j], proy_texts_raw[i]])  # (política, proyecto)
            cand_meta.append((j, sc_bi))

        if not cand_meta:
            continue

        # ordenar candidatos
        if use_rerank:
            ce_scores = np.asarray(ce.predict(cand_pairs))
            order = np.argsort(-ce_scores)  # descendente
        else:
            ce_scores = np.full(len(cand_meta), np.nan, dtype=float)
            order = np.argsort([-m[1] for m in cand_meta])  # descendente por bi-score

        kept = 0
        for idx_ord in order:
            j, sc_bi = cand_meta[int(idx_ord)]
            sc_ce = float(ce_scores[int(idx_ord)]) if use_rerank else np.nan

            out_rows.append(
                {
                    **base,
                    "matched_politica_text": df_politicas.iloc[j][col_text_politica],
                    "bi_similarity_score": sc_bi,
                    "rerank_score": sc_ce,
                    "rank": kept + 1,
                }
            )

            kept += 1
            if kept >= rerank_keep_top_k:
                break

    df_out = pd.DataFrame(out_rows)
    if not df_out.empty:
        df_out = df_out.sort_values([col_id_proyecto, "rank"]).reset_index(drop=True)
    return df_out
