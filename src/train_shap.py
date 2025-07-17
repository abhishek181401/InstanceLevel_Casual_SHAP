
import pickle
import sys
import numpy as np
import pandas as pd
import shap
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    try:
        with open('../data/processed/cleaned.pkl', 'rb') as f:
            data = pickle.load(f)

        X_enc = data['X']
        X_raw = data['X_raw']
        y = data['y']
        mapper = data['mapper']

        X_train_enc, X_test_enc, y_train, y_test = train_test_split(
            X_enc, y, stratify=y, test_size=0.2, random_state=42
        )
        # Match raw splits by index
        X_train_raw = X_raw.loc[X_train_enc.index]
        X_test_raw = X_raw.loc[X_test_enc.index]

        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_enc, y_train)

        joblib.dump(model, '../data/processed/random_forest_model.joblib')

        explainer = shap.TreeExplainer(model)
        shap_values_all = explainer.shap_values(X_test_enc)

        if isinstance(shap_values_all, list):
            shap_values = shap_values_all[1]  
        elif isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3:
            shap_values = shap_values_all[:, :, 1]  # Using class-1 SHAP values
        else:
            shap_values = shap_values_all

        if shap_values.shape != X_test_enc.shape:
            raise ValueError(f"Shape mismatch: SHAP={shap_values.shape}")

        shap_df_enc = pd.DataFrame(
            shap_values, index=X_test_enc.index, columns=X_test_enc.columns
        )

        shap_df_raw = mapper.aggregate_shap(shap_df_enc)

        with open('../data/processed/shap_values_raw.pkl', 'wb') as f:
            pickle.dump({
                'X_test_enc': X_test_enc,
                'X_test_raw': X_test_raw,
                'shap_encoded': shap_df_enc,
                'shap_raw': shap_df_raw
            }, f)

    except Exception as e:
        print(f" Error  {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
