import os
import numpy as np
import pandas as pd
import os

import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import seaborn as sns

import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing

from load_BERT_embeddings import load_arr_from_npz


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

def make_mlp_pipeline(layers, activation, solver, alpha, batch_size, learning_rate):
    pipeline = sklearn.pipeline.Pipeline(
        steps= [
            ('logit', sklearn.neural_network.MLPClassifier(hidden_layer_sizes=layers,
                                                           activation=activation,
                                                           solver=solver,
                                                           alpha=alpha,
                                                           batch_size=batch_size,
                                                           learning_rate=learning_rate
            ))
        ]
    )
    return pipeline

def load_data(metrics=[]):
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    additional_train_features = np.asarray([x_train_df[metric] for metric in metrics]).T
    additional_test_features = np.asarray([x_test_df[metric] for metric in metrics]).T
    # stdscaler = sklearn.preprocessing.StandardScaler(copy=False, with_mean=True, with_std=True)
    scaler = StandardScaler()
    additional_train_features = scaler.fit_transform(additional_train_features)
    additional_test_features = scaler.transform(additional_test_features)

    x_train = load_arr_from_npz(os.path.join(data_dir, 'x_train_BERT_embeddings.npz'))

    x_train = np.append(x_train, additional_train_features, axis=1)

    x_test = load_arr_from_npz(os.path.join( data_dir, 'x_test_BERT_embeddings.npz'))

    x_test = np.append(x_test, additional_test_features, axis=1)

    return x_train, x_train_df, y_train_df, x_test


def hyperparameter_selection(x_dev, x_train_df, y_train_df):
    # Get text and target
    y_labels = y_train_df['Coarse Label'].tolist()
    y_dev = np.array(y_labels)

    max_auc, best_c = 0, 0
    cat = x_train_df["author"].values
    # Optimize C with cross validation 
    kf = sklearn.model_selection.GroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    for c in np.logspace(-4, 4, 17):
        auc_sum = 0
        pipe = make_mlp_pipeline(layers=[4, 4], activation='relu', solver='adam', alpha=c, batch_size=64, learning_rate='invscaling')
        for train_ind, val_ind in kf.split(x_dev, y_dev, cat):
            pipe.fit(x_dev[train_ind], y_dev[train_ind])
            y_hat = pipe.predict_proba(x_dev[val_ind])[:, 1]
            auc = sklearn.metrics.roc_auc_score(y_dev[val_ind], y_hat)
            auc_sum += auc
        avg_auc = auc_sum / 10

        print(f"AUC {avg_auc:.6f} c {c:e}")

        # Prioritize a lower c value
        if avg_auc > max_auc:
            best_c, max_auc = c, avg_auc

    print("Best AUC:", max_auc)
    print("Best C:", best_c)

    return best_c

def test_prediction(x_dev, x_train_df, y_train_df, x_test_df, c):
    y_labels = y_train_df['Coarse Label'].tolist()

    pipe = make_mlp_pipeline(layers=[4, 4], activation='relu', solver='adam', alpha=c, batch_size=64, learning_rate='invscaling')
    pipe.fit(x_dev[train_ind], y_dev[train_ind])
    y_hat = pipe.predict_proba(x_dev[val_ind])[:, 1]
    np.savetxt('yproba1_test.txt', y_hat)
    print("y_hat[:5]:", y_hat[:5])


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
    x_dev, x_train_df, y_train_df, x_test = load_data(metrics)
    c = hyperparameter_selection(x_dev, x_train_df, y_train_df)
    # test_prediction(x_dev, x_train_df, y_train_df, x_test, c)

if __name__ == "__main__":
    main()

