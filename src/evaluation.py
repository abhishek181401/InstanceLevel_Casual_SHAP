
import sys, time, pickle, numpy as np, pandas as pd, joblib, shap
from pathlib import Path
from utils import FeatureMapper

NUM_INSTANCES_CF = 200
NUM_PERTURB = 5
NOISE_SCALE = 0.05
ALPHA = 0.5

def load_resources():
    base = Path(__file__).parent.parent / "data/processed"
    files = {
        "mapper": base / "cleaned.pkl",
        "shap": base / "shap_values_raw.pkl",
        "causal_shap": base / "causal_shap_adjusted.pkl",
        "model": base / "random_forest_model.joblib",
    }
    for name, path in files.items():
        if not path.exists():
            print(f"❌ Missing file: {path}", file=sys.stderr)
            sys.exit(1)
    data = pickle.load(open(files["mapper"], "rb"))
    mapper = data.get("mapper")
    shap_data = pickle.load(open(files["shap"], "rb"))
    X_raw = shap_data["X_test_raw"]
    X_enc = shap_data["X_test_enc"]
    shap_causal = pd.read_pickle(files["causal_shap"])
    model = joblib.load(files["model"])

    return mapper, X_raw, X_enc, shap_causal, model

def counterfactual_test(mapper, X_raw, X_enc, shap_causal, model):
    n = min(NUM_INSTANCES_CF, len(X_raw))
    idxs = np.random.choice(X_raw.index, n, replace=False)
    flips = 0
    distances = []
    for idx in idxs:
        orig_enc = X_enc.loc[[idx]]
        orig_pred = model.predict(orig_enc)[0]
        ranked = shap_causal.loc[idx].abs().sort_values(ascending=False).index
        cf_raw = X_raw.loc[[idx]].copy()
        changed = 0
        for feat in ranked:
            changed += 1
            if np.issubdtype(cf_raw[feat].dtype, np.number):
                cf_raw.at[idx, feat] += np.random.choice([-1, 1]) * np.random.uniform(1, 5)
            else:
                options = list(X_raw[feat].dropna().unique())
                options = [o for o in options if o != cf_raw.at[idx, feat]]
                if options:
                    cf_raw.at[idx, feat] = np.random.choice(options)
            cf_enc = mapper.transform(cf_raw)
            cf_pred = model.predict(cf_enc)[0]
            if cf_pred != orig_pred:
                flips += 1
                distances.append(changed)
                break
        else:
            distances.append(changed)
    return flips / n, float(np.mean(distances))

def stability_test(mapper, X_raw, model):
    idxs = np.random.choice(X_raw.index, min(50, len(X_raw)), replace=False)
    variances = []
    explainer = shap.TreeExplainer(model)
    for idx in idxs:
        base = X_raw.loc[[idx]]
        shap_vals = []
        for _ in range(NUM_PERTURB):
            pert = base.copy()
            for c in pert.select_dtypes(include=np.number):
                pert.at[idx, c] += base.at[idx, c] * NOISE_SCALE * np.random.randn()
            sv = explainer.shap_values(mapper.transform(pert))
            sv = sv[1] if isinstance(sv, list) else sv
            shap_vals.append(sv.flatten())
        variances.append(np.var(shap_vals, axis=0).mean())
    return float(np.mean(variances))

def runtime_profile(mapper, X_enc, model):
    sample = X_enc.iloc[:100]
    explainer = shap.TreeExplainer(model)
    t0 = time.perf_counter()
    explainer.shap_values(sample)
    t1 = time.perf_counter()
    for _ in sample.index:
        _ = X_enc.iloc[0] * (1 + ALPHA)
    return {"shap_100": t1 - t0, "causal_adj_100": time.perf_counter() - t1}

def main():
    mapper, X_raw, X_enc, shap_causal, model = load_resources()
    fr, ed = counterfactual_test(mapper, X_raw, X_enc, shap_causal, model)
    sv = stability_test(mapper, X_raw, model)
    rt = runtime_profile(mapper, X_enc, model)


    pd.DataFrame([{"flip_rate":fr, "avg_edit_distance":ed,
                   "stability_variance":sv, **rt}]).to_csv(
        Path(__file__).parent.parent / "data/processed/evaluation_metrics.csv", index=False
    )

if __name__ == "__main__":
    main()
