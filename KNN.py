import os
import numpy as np
import pandas as pd
import os
import itertools

import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import seaborn as sns

from load_BERT_embeddings import load_arr_from_npz

import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 68

def load_data(metrics=[]):
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    additional_train_features = np.asarray([x_train_df[metric] for metric in metrics]).T
    additional_test_features = np.asarray([x_test_df[metric] for metric in metrics]).T

    x_train = load_arr_from_npz(os.path.join(data_dir, 'x_train_BERT_embeddings.npz'))
    x_train = np.append(x_train, additional_train_features, axis=1)
    x_test = load_arr_from_npz(os.path.join( data_dir, 'x_test_BERT_embeddings.npz'))
    x_test = np.append(x_test, additional_test_features, axis=1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # x_train = scaler.fit_transform(additional_train_features)
    # x_test = scaler.transform(additional_test_features)

    return x_train, x_train_df, y_train_df, x_test


def make_knn_pipeline(k, p):
    pipeline = sklearn.pipeline.Pipeline(
        steps= [
            ('logit', sklearn.neighbors.KNeighborsClassifier(n_neighbors=k,
                                                             p=p
            ))
        ]
    )
    return pipeline


def hyperparameter_selection(x_dev, x_train_df, y_train_df):
    # Get text and target
    y_labels = y_train_df['Coarse Label'].tolist()
    y_dev = np.array([0 if label == 'Key Stage 2-3' else 1 for label in y_labels])

    max_auc, best_p, best_k = 0, 0, 0
    best_per_p = {}
    # best_per_k = {}
    cat = x_train_df["author"].values

    kf = sklearn.model_selection.GroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    for p in [1, 2]:
        for k in np.linspace(175, 275, 50):
            k = int(k)
            auc_sum = 0
            pipe = make_knn_pipeline(k, p)
            for train_ind, val_ind in kf.split(x_dev, y_dev, cat):
                pipe.fit(x_dev[train_ind], y_dev[train_ind])
                y_hat = pipe.predict_proba(x_dev[val_ind])[:, 1]
                auc = sklearn.metrics.roc_auc_score(y_dev[val_ind], y_hat)
                auc_sum += auc
            avg_auc = auc_sum / 10

            print(f"AUC {avg_auc:.6f} with Lp metric {p} and k {k}")

            if avg_auc > max_auc:
                max_auc, best_p, best_k = avg_auc, p, k

            if str(p) in best_per_p.keys():
                if avg_auc > best_per_p[str(p)][0]:
                    best_per_p[str(p)] = [avg_auc, k]
            else:
                best_per_p[str(p)] = [avg_auc, k]

            # if str(k) in best_per_k.keys():
            #     if avg_auc > best_per_k[str(k)][0]:
            #         best_per_k[str(k)] = [avg_auc, p]
            # else:
            #     best_per_k[str(k)] = [avg_auc, p]

    print("-"*64)
    print("Best AUC:", max_auc)
    print("Best p:", best_p)
    print("Best k:", best_k)
    print("-"*64)
    print("Best per p", best_per_p)
    # print("Best per k", best_per_k)

    return max_auc, best_p, best_k

def test_prediction(x_dev, y_train_df, x_test, p, k):
    y_labels = y_train_df['Coarse Label'].tolist()
    y_dev = np.array([0 if label == 'Key Stage 2-3' else 1 for label in y_labels])
    pipe = pipe = make_knn_pipeline(k, p)
    pipe.fit(x_dev, y_dev)
    y_hat = pipe.predict_proba(x_test)[:, 1]
    np.savetxt('yproba_knn.txt', y_hat)

def main():
    metrics = ['char_count', 'word_count',
       'sentence_count', 'avg_word_length', 'avg_sentence_length',
       'type_token_ratio', 'pronoun_freq', 'function_words_count',
       'punctuation_frequency', 'sentiment_polarity', 'sentiment_subjectivity',
       'readability_Kincaid', 'readability_ARI', 'readability_Coleman-Liau',
       'readability_FleschReadingEase', 'readability_GunningFogIndex',
       'readability_LIX', 'readability_SMOGIndex', 'readability_RIX',
       'readability_DaleChallIndex', 'info_characters_per_word',
       'info_syll_per_word', 'info_words_per_sentence',
       'info_type_token_ratio', 'info_characters', 'info_syllables',
       'info_words', 'info_wordtypes']
    metric_combinations = []
    best_combination, max_auc, best_p, best_k = [], 0, 0, 0
    for r in range(len(metrics)):
        metric_combinations = itertools.combinations(metrics, r)
        for metric_list in metric_combinations:
            x_dev, x_train_df, y_train_df, x_test = load_data(metrics)
            auc, p, k = hyperparameter_selection(x_dev, x_train_df, y_train_df)
        
            if auc > max_auc:
                    best_combination, max_auc, best_p, best_k = metric_list, auc, p, k

    print("#"*64)
    print("Best metrics:", best_combination)
    print("Best AUC:", max_auc)
    print("Best p:", best_p)
    print("Best k:", best_k)
    print("#"*64)

    x_dev, x_train_df, y_train_df, x_test = load_data(best_combination)
    test_prediction(x_dev, y_train_df, x_test, best_p, best_k)

if __name__ == "__main__":
    main()