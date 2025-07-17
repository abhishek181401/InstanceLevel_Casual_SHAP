
import pickle
import sys
import pandas as pd
import numpy as np
import joblib
from utils import FeatureMapper

def load_resources():
    try:
        with open('../data/processed/cleaned.pkl', 'rb') as f:
            clean_data = pickle.load(f)
        mapper: FeatureMapper = clean_data['mapper']
        with open('../data/processed/shap_values_raw.pkl', 'rb') as f:
            shap_data = pickle.load(f)
        X_raw: pd.DataFrame = shap_data['X_test_raw']
        model = joblib.load('../data/processed/random_forest_model.joblib')
        return mapper, X_raw, model
    except Exception as e:
        print(f" Error loading resources: {e}")
        sys.exit(1)

def generate_counterfactual(mapper, X_raw, model):
    idx = np.random.choice(X_raw.index, size=1)[0]
    instance_raw = X_raw.loc[[idx]]
    instance_enc = mapper.transform(instance_raw)

    orig_class = model.predict(instance_enc)[0]
    print(f"🎯 Instance index: {idx}")
    print("Original raw instance:", instance_raw.to_dict(orient='records')[0])
    print(f"Original encoded instance head:\n{instance_enc.iloc[0, :5].to_dict()}")
    print(f"Original predicted class: {orig_class}")

    cf_raw = instance_raw.copy()
    if 'education-num' in cf_raw.columns:
        cf_raw.loc[idx, 'education-num'] = cf_raw.loc[idx, 'education-num'] + 1
        print(" Increased 'education-num' by 1 ")
    else:
        print(" 'education-num' not found in features")

    # Encode and predict counterfactual
    cf_enc = mapper.transform(cf_raw)
    cf_class = model.predict(cf_enc)[0]
    print("Counterfactual raw instance:", cf_raw.to_dict(orient='records')[0])

    return instance_raw, cf_raw, orig_class, cf_class

def main():
    mapper, X_raw, model = load_resources()
    generate_counterfactual(mapper, X_raw, model)

if __name__ == "__main__":
    main()
