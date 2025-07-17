
import pandas as pd
import pickle
import sys
from utils import FeatureMapper  

def main():
    try:
        cols = [
            'age','workclass','fnlwgt','education','education-num',
            'marital-status','occupation','relationship','race',
            'sex','capital-gain','capital-loss','hours-per-week',
            'native-country','income'
        ]
        df = pd.read_csv('../data/raw/adult.csv', header=None, names=cols, na_values=' ?')

        before = len(df)
        df = df.dropna()

        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

        # creating a map for raw --> encoded features
        mapper = FeatureMapper()
        X = mapper.fit_transform(df.drop(columns=['income']))
        y = df['income']
        X_raw = df.drop(columns=['income'])        

        with open('../data/processed/cleaned.pkl', 'wb') as f:
            pickle.dump({'X': X, 'X_raw': X_raw, 'y': y, 'mapper': mapper}, f)

    except Exception as e:
        print(f" error:  {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
