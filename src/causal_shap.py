
import os
import pickle
import sys
import pandas as pd
import networkx as nx

ALPHA = 0.5
INSTANCE_DIR = "../data/processed/instance_dags"
OUTPUT_FILE = "../data/processed/causal_shap_adjusted.pkl"

def main():
    try:
        with open('../data/processed/shap_values_raw.pkl', 'rb') as f:
            shap_data = pickle.load(f)
        shap_raw: pd.DataFrame = shap_data['shap_raw']

        adjusted = {}

        for inst, shap_row in shap_raw.iterrows():
            path = os.path.join(INSTANCE_DIR, f"instance_{inst}.graphml")
            if not os.path.exists(path):
                continue
            G = nx.read_graphml(path)

            indeg = {node: G.in_degree(node) for node in G.nodes()}
            adj = {}
            for feat, val in shap_row.items():
                degree = indeg.get(feat, 0)
                adj[feat] = val * (1 + ALPHA * degree)
            raw_sum = float(shap_row.sum())
            adj_sum = float(pd.Series(adj).sum())
            if adj_sum != 0:
                factor = raw_sum / adj_sum
                adj = {k: v * factor for k, v in adj.items()}

            adjusted[inst] = adj

        pd.DataFrame(adjusted).T.to_pickle(OUTPUT_FILE)
    except Exception as e:
        print(f"Error in Phase 4: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
