

import pickle
import sys
import pandas as pd
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure
import matplotlib.pyplot as plt

NUMERIC_THRESHOLD = 0.1
TOP_K = 8

def _ordinal_encode(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.copy()
    for col in df_num.select_dtypes(include="object"):
        df_num[col] = df_num[col].astype("category").cat.codes
    return df_num

def main():
    try:
        with open("../data/processed/shap_values_raw.pkl", "rb") as f:
            saved = pickle.load(f)
        X_raw = saved["X_test_raw"]
        shap_raw = saved["shap_raw"]

        top_feats = shap_raw.abs().mean().nlargest(TOP_K).index.tolist()
        df_sel = X_raw[top_feats]
        df_num = _ordinal_encode(df_sel)

        sm = from_pandas(df_num, w_threshold=0.3)
        for u, v in list(sm.edges()):
            if abs(df_num[u].corr(shap_raw[v])) < NUMERIC_THRESHOLD:
                sm.remove_edge(u, v)

        pickle.dump({
            "structure_model": sm,
            "top_features": top_feats,
            "encoding_info": {
                col: df_sel[col].astype("category").cat.categories.tolist()
                for col in df_sel.select_dtypes(include="object")
            }
        }, open("../data/processed/global_dag.pkl", "wb"))

        viz = plot_structure(sm)

        html_output = "../data/processed/global_dag.html"
        html_str = viz.generate_html()

        with open(html_output, "w", encoding="utf-8") as fout:
            fout.write(html_str)

    except Exception as exc:
        print(f"Error  {exc}",)
        sys.exit(1)

if __name__ == "__main__":
    main()
