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
            ('rescaler', sklearn.preprocessing.MinMaxScaler()),
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

    x_train = load_arr_from_npz(os.path.join(data_dir, 'x_train_BERT_embeddings.npz'))

    x_train = np.append(x_train, additional_train_features, axis=1)

    x_test = load_arr_from_npz(os.path.join( data_dir, 'x_test_BERT_embeddings.npz'))

    x_test = np.append(x_test, additional_test_features, axis=1)

    return x_train, x_train_df, y_train_df, x_test


def hyperparameter_selection(x_dev, x_train_df, y_train_df):
    # Get text and target
    y_labels = y_train_df['Coarse Label'].tolist()
    y_dev = np.array(y_labels)

    max_auc, best_c, best_bs, best_layer = 0, 0, 0, []
    best_per_layer = {}
    cat = x_train_df["author"].values
    # Optimize C with cross validation 
    kf = sklearn.model_selection.GroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    # for layer in ([2, 2], [4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]):
    for layer in ([16, 16], [32, 32]):
        for bs in [32, 64, 128]:
            for c in np.logspace(-4, 4, 17)[0:-5]:
                auc_sum = 0
                pipe = make_mlp_pipeline(layers=layer, activation='relu', solver='adam', alpha=c, batch_size=bs, learning_rate='invscaling')
                for train_ind, val_ind in kf.split(x_dev, y_dev, cat):
                    pipe.fit(x_dev[train_ind], y_dev[train_ind])
                    y_hat = pipe.predict_proba(x_dev[val_ind])[:, 1]
                    auc = sklearn.metrics.roc_auc_score(y_dev[val_ind], y_hat)
                    auc_sum += auc
                avg_auc = auc_sum / 10

                print(f"AUC {avg_auc:.6f} with layers {layer}, batch size {bs}, c {c:e}")

                # Prioritize a lower c value
                if avg_auc > max_auc:
                    best_c, max_auc, best_bs, best_layer = c, avg_auc, bs, layer
                if str(layer) in best_per_layer.keys():
                    if avg_auc > best_per_layer[str(layer)][0]:
                        best_per_layer[str(layer)] = [max_auc, best_c.item(), best_bs]
                else:
                    best_per_layer[str(layer)] = [max_auc, best_c.item(), best_bs]

    print("-"*64)
    print("Best AUC:", max_auc)
    print("Best C:", best_c)
    print("Best hidden layers:", best_layer)
    print("Best batch size", best_bs)
    print("-"*64)
    print(best_per_layer)

    return best_c, best_bs, best_layer, best_per_layer

def test_prediction(x_dev, y_train_df, x_test, c, bs, layer):
    y_labels = y_train_df['Coarse Label'].tolist()
    y_dev = np.array(y_labels)
    pipe = make_mlp_pipeline(layers=layer, activation='relu', solver='adam', alpha=c, batch_size=bs, learning_rate='invscaling')
    pipe.fit(x_dev, y_dev)
    y_hat = pipe.predict_proba(x_test)[:, 1]
    np.savetxt('yproba_minmax.txt', y_hat)

# '[2, 2]': [0.31622776601683794, 0.7042195481927992, 32, [2, 2]],
# '[4, 4]': [3.1622776601683795, 0.7727808871099182, 128, [4, 4]],
# '[8, 8]': [3.1622776601683795, 0.7729369697170639, 128, [8, 8]],
# '[16, 16]': [1.0, 0.7732375038973036, 64, [16, 16]],
# '[32, 32]': [3.1622776601683795, 0.773644880570182, 128, [32, 32]]

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
    c, bs, layer, best_per_layer = hyperparameter_selection(x_dev, x_train_df, y_train_df)
    test_prediction(x_dev, y_train_df, x_test, c, bs, layer)

if __name__ == "__main__":
    main()