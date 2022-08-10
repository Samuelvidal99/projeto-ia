from __future__ import annotations
import pandas as pd
from pandas import DataFrame


def preprocess(path: str, sample_size: int = None, random_state: int = 42) -> tuple[DataFrame, DataFrame]:
    # Read dataset and select columns
    df = pd.read_excel(path, engine='openpyxl')
    df = df[['essay_set', 'essay', 'domain1_score']]
    df = df.loc[df['essay_set'] == 1]
    if sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
    df.rename(columns={'domain1_score': 'score'}, inplace=True)
    print('Dataset length:', len(df))
    
    # Split data into features and labels
    features = DataFrame(df['essay'], columns=['essay'])
    labels = DataFrame(df['score'], columns=['score'])
    
    return features, labels


def main():
    result_output = 'results/dataset.txt'
    input_path = 'data/essay.xlsx'
    features_output_path = 'data/essay_in.csv'
    labels_output_path = 'data/essay_out.csv'
    features, labels = preprocess(input_path, sample_size=300)
    features.to_csv(features_output_path, index=0)
    labels.to_csv(labels_output_path, index=0)

    with open(result_output, 'w') as f:
        f.write(f'Features dataset length: {len(features)} \n')
        f.write(f'Labels dataset length: {len(labels)} \n')

if __name__ == '__main__':
    main()
