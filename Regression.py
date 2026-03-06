import os
import numpy as np
import pandas as pd
import os

import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.feature_extraction.text import CountVectorizer

from matplotlib import pyplot as plt
import seaborn as sns

import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing

from load_BERT_embeddings import load_arr_from_npz
from wordfreq import word_frequency

import warnings
warnings.filterwarnings('ignore')


RANDOM_SEED = 68

# if __name__ == '__main__':
#     data_dir = 'data_readinglevel'
#     x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
#     y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

#     N, n_cols = x_train_df.shape
#     print("Shape of x_train_df: (%d, %d)" % (N, n_cols))
#     print("Shape of y_train_df: %s" % str(y_train_df.shape))

#     # Print out 8 random entries
#     tr_text_list = x_train_df['text'].values.tolist()
#     prng = np.random.RandomState(101)
#     rows = prng.permutation(np.arange(y_train_df.shape[0]))
#     for row_id in rows[:8]:
#         text = tr_text_list[row_id]
#         print("row %5d | %s BY %s | y = %s" % (
#             row_id,
#             y_train_df['title'].values[row_id],
#             y_train_df['author'].values[row_id],
#             y_train_df['Coarse Label'].values[row_id],
#             ))

#         line_list = textwrap.wrap(tr_text_list[row_id],
#             width=70,
#             initial_indent='  ',
#             subsequent_indent='  ')
#         print('\n'.join(line_list))
#         print("")

def make_regression_pipeline(C):
    pipeline = sklearn.pipeline.Pipeline(
        steps= [
            ('rescaler', sklearn.preprocessing.MinMaxScaler()),
            ('logit', sklearn.linear_model.LogisticRegression(solver="liblinear", penalty="l2", C=C, max_iter=1000))
        ]
    )
    return pipeline
        
def load_data(metrics=[]):
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    text = x_train_df['text'].values.tolist()
    words = [s.lower() for s in text]
    #avg_freq = np.mean([word_frequency(w, 'en') for w in words])
    #x_train = np.append(x_train, avg_freq, axis=1)
    #x_test = np.append(x_test, additional_test_features, axis=1)
    additional_train_features = np.asarray([x_train_df[metric] for metric in metrics]).T
    additional_test_features = np.asarray([x_test_df[metric] for metric in metrics]).T
    # stdscaler = sklearn.preprocessing.StandardScaler(copy=False, with_mean=True, with_std=True)

    x_train = load_arr_from_npz(os.path.join(data_dir, 'x_train_BERT_embeddings.npz'))

    x_train = np.append(x_train, additional_train_features, axis=1)

    x_test = load_arr_from_npz(os.path.join( data_dir, 'x_test_BERT_embeddings.npz'))

    x_test = np.append(x_test, additional_test_features, axis=1) 

    return x_train, x_train_df, y_train_df, x_test


def hyperparameter_selection(x_train_df, y_train_df):
    import torch
    labels = np.array([0 if l == "Key Stage 2-3" else 1 for l in y_train_df["Coarse Label"].tolist()])
    texts = x_train_df["text"].tolist()
    authors = x_train_df["author"].values

    kf = sklearn.model_selection.GroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    max_auc, best_epochs = 0, 0

    for epochs in [1, 2, 3]:
        auc_sum = 0
        for train_ind, val_ind in kf.split(texts, labels, authors):
            auc = fine_tune_bert(
                x_train_df.iloc[train_ind], 
                y_train_df.iloc[train_ind],
                x_train_df.iloc[val_ind],
                epochs
            )
            auc_sum += auc
        avg_auc = auc_sum / 10
        print(f"Epochs {epochs} AUC {avg_auc:.6f}")
        if avg_auc > max_auc:
            max_auc, best_epochs = avg_auc, epochs

    print("Best AUC:", max_auc)
    print("Best epochs:", best_epochs)
    return best_epochs

def test_prediction(x_dev, x_train_df, y_train_df, x_test_df, c):
    y_labels = y_train_df['Coarse Label'].tolist()

    pipe = make_mlp_pipeline(layers=[4, 4], activation='relu', solver='adam', alpha=c, batch_size=64, learning_rate='invscaling')
    pipe.fit(x_dev[train_ind], y_dev[train_ind])
    y_hat = pipe.predict_proba(x_dev[val_ind])[:, 1]
    np.savetxt('yproba1_test.txt', y_hat)
    print("y_hat[:5]:", y_hat[:5])


def main():
    def main():
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
    best_epochs = hyperparameter_selection(x_train_df, y_train_df)
    fine_tune_bert(x_train_df, y_train_df, x_test_df, best_epochs)

if __name__ == "__main__":
    main()