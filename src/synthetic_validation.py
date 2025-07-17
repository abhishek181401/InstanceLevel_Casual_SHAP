
import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import networkx as nx
from causalnex.structure.notears import from_pandas

parser = argparse.ArgumentParser()
parser.add_argument("--nodes",     type=int,   default=8,       help="Number of variables in the DAG")
parser.add_argument("--edge_prob", type=float, default=0.25,    help="Probability of each possible edge")
parser.add_argument("--n_samples", type=int,   default=20000,   help="Rows to sample from SEM")
parser.add_argument("--noise_std", type=float, default=0.3,     help="Std dev of Gaussian noise")
parser.add_argument("--pct_keep",  type=float, default=30.0,    help="Keep top X%% of edge weights")
parser.add_argument("--no_scale",  action="store_true",        help="Disable z‑score scaling")
args = parser.parse_args()


OUT_DIR = Path("../data/processed/synthetic_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def random_dag(num_nodes, p, seed=42):
    rng = np.random.default_rng(seed)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if rng.random() < p:
                dag.add_edge(i, j)
    return dag

def sample_linear_sem(dag, n, noise_std, seed=42):
    rng = np.random.default_rng(seed)
    W = {edge: rng.uniform(1.0, 2.0) * rng.choice([-1,1]) for edge in dag.edges()}
    order = list(nx.topological_sort(dag))
    X = np.zeros((n, len(order)))
    for node in order:
        parents = list(dag.predecessors(node))
        eps = rng.normal(0, noise_std, n)
        if parents:
            coefs = np.array([W[(p, node)] for p in parents])
            X[:, node] = X[:, parents] @ coefs + eps
        else:
            X[:, node] = eps
    cols = [f"V{i}" for i in order]
    return pd.DataFrame(X, columns=cols)

def prf1(true_edges, pred_edges, directed=True):
    if not directed:
        true_edges = {frozenset(e) for e in true_edges}
        pred_edges = {frozenset(e) for e in pred_edges}
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    return prec, rec, f1

def main():
    try:
        true_dag = random_dag(args.nodes, args.edge_prob)
        true_dag_str = nx.relabel_nodes(true_dag, {i: f"V{i}" for i in true_dag.nodes()})
        nx.write_gml(true_dag_str, OUT_DIR/"synthetic_dag_true.gml")

        df = sample_linear_sem(true_dag, args.n_samples, args.noise_std)
        if not args.no_scale:
            df = (df - df.mean()) / df.std(ddof=0)

        sm_raw = from_pandas(df, w_threshold=0.0)
        raw_edges = list(sm_raw.edges())
        print(f"🔍 Raw learned edges: {len(raw_edges)}")
        nx.write_gml(sm_raw, OUT_DIR/"synthetic_dag_learned_raw.gml")

        weights = [abs(sm_raw.get_edge_data(u, v)["weight"]) for u, v in raw_edges]
        if weights:
            print("⚖️ Weight stats:",
                  f"min={np.min(weights):.4f},",
                  f"median={np.median(weights):.4f},",
                  f"max={np.max(weights):.4f}")
            thr = 0.0
        else:
            print("⚠️ No edges learned; using thr=0")
            thr = 0.0

        sm_pruned = sm_raw.copy()
        sm_pruned.remove_edges_below_threshold(thr)
        pruned_edges = list(sm_pruned.edges())
        nx.write_gml(sm_pruned, OUT_DIR/"synthetic_dag_learned_pruned.gml")

        true_edges = set(true_dag_str.edges())
        pred_edges = set(pruned_edges)
        p_d, r_d, f_d = prf1(true_edges, pred_edges, directed=True)
        p_s, r_s, f_s = prf1(true_edges, pred_edges, directed=False)

        pd.DataFrame([{
            "nodes": args.nodes,
            "edge_prob": args.edge_prob,
            "n_samples": args.n_samples,
            "noise_std": args.noise_std,
            "pct_keep": args.pct_keep,
            "raw_edges": len(raw_edges),
            "pruned_edges": len(pruned_edges),
            "directed_precision": p_d,
            "directed_recall": r_d,
            "directed_f1": f_d,
            "skeleton_precision": p_s,
            "skeleton_recall": r_s,
            "skeleton_f1": f_s
        }]).to_csv(OUT_DIR/"metrics.csv", index=False)

        print(f"✅ Directed F1: {f_d:.3f} | Skeleton F1: {f_s:.3f}")

    except Exception as e:
        print(f"Error in synthetic validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
