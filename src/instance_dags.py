
import pickle
import sys
import os
import pandas as pd
import networkx as nx
from causalnex.structure.structuremodel import StructureModel

TOP_M = 5
OUTPUT_DIR = "../data/processed/instance_dags"

def main():
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open("../data/processed/global_dag.pkl", "rb") as f:
            global_data = pickle.load(f)
        sm: StructureModel = global_data["structure_model"]
        dag_nodes = set(sm.nodes)

        with open("../data/processed/shap_values_raw.pkl", "rb") as f:
            shap_data = pickle.load(f)
        shap_raw: pd.DataFrame = shap_data["shap_raw"]

        summary = []
        for idx, row in shap_raw.iterrows():
            top_feats = list(row.abs().nlargest(TOP_M).index)
            valid_feats = [f for f in top_feats if f in dag_nodes]
            if not valid_feats:
                print( "no valid topraw features in global DAG. Skipping.")
                continue

            sub_sm: StructureModel = sm.subgraph(valid_feats).copy()
            sub_sm.threshold_till_dag()

            filepath = os.path.join(OUTPUT_DIR, f"instance_{idx}.graphml")
            nx.write_graphml(sub_sm, filepath)

            summary.append({
                "instance": idx,
                "original_top": top_feats,
                "used_nodes": list(sub_sm.nodes),
                "used_edges": list(sub_sm.edges)
            })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
 

    except Exception as e:
        print(f" Error in Phase 3: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
