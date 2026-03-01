import os
import pandas as pd

from semantic_matching import semantic_project_merge_advanced


def main():
    # Inputs (locales)
    politicas_path = "data/raw/politicas.xlsx"
    proyectos_path = "data/raw/proyectos.xlsx"

    df_politicas = pd.read_excel(politicas_path)
    df_proyectos = pd.read_excel(proyectos_path)

    df_match = semantic_project_merge_advanced(
        df_politicas=df_politicas,
        df_proyectos=df_proyectos,
        col_text_politica="Indicador de Producto(MGA)",
        col_text_proyecto="Indicadores de producto PATR",
        col_id_proyecto="codigo_proyecto",
        top_k=4,
        top_n_candidates=80,
        min_bi_score=0.40,
        model_name="BAAI/bge-m3",  # alternativa: "intfloat/multilingual-e5-large"
        batch_size=128,
        device="cpu",
        use_rerank=True,
        cross_encoder_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )

    os.makedirs("data/outputs", exist_ok=True)
    out_path = "data/outputs/matching_topk.xlsx"
    df_match.to_excel(out_path, index=False)
    print("Guardado:", out_path)


if __name__ == "__main__":
    main()
