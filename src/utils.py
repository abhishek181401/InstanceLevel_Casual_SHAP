
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class FeatureMapper:
    

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.raw_cols = None
        self.feature_index_map = None
 
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.raw_cols = list(X.columns)
        X_enc = pd.DataFrame(
            self.encoder.fit_transform(X),
            columns=self.encoder.get_feature_names_out(self.raw_cols),
            index=X.index,
        )
        self.feature_index_map = {
            raw: [c for c in X_enc.columns if c.startswith(raw + "_")]
            for raw in self.raw_cols
        }
        return X_enc

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_enc = pd.DataFrame(
            self.encoder.transform(X),
            columns=self.encoder.get_feature_names_out(self.raw_cols),
            index=X.index,
        )
        return X_enc

    def aggregate_shap(self, shap_df: pd.DataFrame) -> pd.Series:
      
        agg = {}
        for raw, cols in self.feature_index_map.items():
            agg[raw] = shap_df[cols].sum(axis=1)
        return pd.DataFrame(agg)
