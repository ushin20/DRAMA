import os
import pandas as pd
import numpy as np
import argparse
import ast

def preprocess(dataset_name):
    train_path = f'./data/{dataset_name}/train.csv'
    test_path = f'./data/{dataset_name}/test.csv'

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()
    print("Train:", train.shape)
    print("Test:", test.shape)

    if dataset_name == 'swat':
        assert train.columns.equals(test.columns), "Train/Test columns do not match!"
        test['Normal/Attack'] = (
            test['Normal/Attack']
            .astype(str)
            .str.strip()
            .str.replace(' ', '', regex=False)
            .str.capitalize()
        )
        test['Label'] = test['Normal/Attack'].map({'Normal': 0, 'Attack': 1})

        test_label = test['Label'].values
        train = train.drop(columns=["Timestamp", "Normal/Attack"])
        test = test.drop(columns=["Timestamp", "Normal/Attack", "Label"])
    elif dataset_name == 'wadi':
        
        train.columns = [col.split('\\')[-1] if isinstance(col, str) else col for col in train.columns]
    
        test_label = test["Attack LABLE (1:No Attack, -1:Attack)"].values
        test_label = (test_label == -1).astype(int)

        train = train.drop(columns=["Row", "Date", "Time"])
        test = test.drop(columns=["Row", "Date", "Time", "Attack LABLE (1:No Attack, -1:Attack)"])
        assert train.columns.equals(test.columns), "Train/Test columns do not match!"

        empty_cols_train = set(train.columns[train.isna().all()])
        empty_cols_test = set(test.columns[test.isna().all()])
        common_empty_cols = list(empty_cols_train & empty_cols_test)
        train.drop(columns=common_empty_cols, inplace=True)
        test.drop(columns=common_empty_cols, inplace=True)

        train = train.ffill().bfill()
        test = test.ffill().bfill()

    print("Columns:", train.columns)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Window labels shape:", test_label.shape)
    print("Anomalies:", np.sum(test_label), "/", len(test_label),
          f"({np.sum(test_label)/len(test_label)*100:.2f}%)")

    save_dir = f'./data/{dataset_name}/preprocessed/'
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'train.npy'), train)
    np.save(os.path.join(save_dir, 'test.npy'), test)
    np.save(os.path.join(save_dir, 'label.npy'), test_label)

    with open(os.path.join(save_dir, 'columns.txt'), 'w') as f:
        for col in train.columns:
            f.write(col + '\n')

def preprocess_NASA():
    labels = pd.read_csv('./data/NASA/labeled_anomalies.csv')

    smap_path = './data/smap/'
    msl_path = './data/msl/'
    os.makedirs(smap_path, exist_ok=True)
    os.makedirs(msl_path, exist_ok=True)

    smap_anomaly, smap_label = 0,0
    msl_anomaly, msl_label = 0,0

    for row in labels.itertuples(index=True):
        train = os.path.join('./data/NASA/train', row.chan_id+'.npy')
        test = os.path.join('./data/NASA/test', row.chan_id+'.npy')
        
        train = np.load(train)
        test = np.load(test)
        label_idx = ast.literal_eval(row.anomaly_sequences) # on test
        label = np.zeros(test.shape[0])
        
        for s,e in label_idx:
            label[s:e+1] = 1
        
        if row.spacecraft == 'SMAP':
            np.save(os.path.join(smap_path, f'{row.chan_id}_train.npy'), train)
            np.save(os.path.join(smap_path, f'{row.chan_id}_test.npy'), test)
            np.save(os.path.join(smap_path, f'{row.chan_id}_label.npy'), label)
            smap_anomaly += np.sum(label)
            smap_label += len(label)
        elif row.spacecraft == 'MSL':
            np.save(os.path.join(msl_path, f'{row.chan_id}_train.npy'), train)
            np.save(os.path.join(msl_path, f'{row.chan_id}_test.npy'), test)
            np.save(os.path.join(msl_path, f'{row.chan_id}_label.npy'), label)
            msl_anomaly += np.sum(label)
            msl_label += len(label)

        print(f'{row.chan_id}_train.npy shape:', train.shape)
        print(f'{row.chan_id}_test.npy shape:', test.shape)
        print(f'{row.chan_id}_label.npy shape:', label.shape)
        print("Anomalies:", np.sum(label), "/", len(label), f"({np.sum(label)/len(label)*100:.2f}%)\n")
    
    print(f"SMAP: {smap_anomaly}/{smap_label} {smap_anomaly/smap_label*100:.2f}")
    print(f"MSL: {msl_anomaly}/{msl_label} {msl_anomaly/msl_label*100:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess ICS dataset")
    parser.add_argument('--dataset', type=str, required=True, choices=['swat', 'wadi', 'smap', 'msl'],
                        help="Dataset name (e.g., 'swat', 'wadi')")
    args = parser.parse_args()

    if args.dataset in ['swat', 'wadi']:
        preprocess(args.dataset)
    else:
        preprocess_NASA()
