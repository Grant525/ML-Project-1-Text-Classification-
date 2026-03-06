import os
import numpy as np
import pandas as pd
import os

import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import seaborn as sns

from load_BERT_embeddings import load_arr_from_npz
from CNN import CNN, MyDataset
from KNN import make_knn_pipeline
from Standard import make_mlp_pipeline

import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 68
torch.manual_seed(RANDOM_SEED)


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

    scaler = sklearn.preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_labels = y_train_df['Coarse Label'].tolist()
    y_train = np.array([0 if label == 'Key Stage 2-3' else 1 for label in y_labels])

    add_train = scaler.fit_transform(additional_train_features)
    add_test = scaler.transform(additional_test_features)

    return x_train, y_train, x_test, add_train, add_test

def predict(x_test, add_test, knn, mlp, cnn, polling):
    knn_probs = knn.predict_proba(x_test)[:, 1]
    mlp_probs = mlp.predict_proba(x_test)[:, 1]
    cnn_probs = cnn(torch.tensor(x_test.reshape((x_test.shape[0], 1, x_test.shape[1])).astype(np.float32)))[:, 1].detach().numpy()

    polling_test = np.vstack((knn_probs, mlp_probs, cnn_probs)).reshape(1197, 3)
    polling_test = np.hstack((polling_test, add_test))
    yhat = polling.predict_proba(polling_test)[:, 1]
    np.savetxt('yproba_polling.txt', yhat)

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
    x_dev, y_dev, x_test, add_train, add_test = load_data(metrics)

    # Best AUC: 0.7735294743372317
    # Best p: 1
    # Best k: 237
    knn = make_knn_pipeline(p=1, k=237)
    knn.fit(x_dev, y_dev)
    knn_probs = knn.predict_proba(x_dev)[:,1]
    print(sklearn.metrics.roc_auc_score(y_dev, knn_probs))

    # Best AUC: 0.779
    mlp = make_mlp_pipeline(layers=[8, 4], activation='relu', solver='adam', alpha=3.1622776601683795, batch_size=64, learning_rate='invscaling')
    mlp.fit(x_dev, y_dev)
    mlp_probs = mlp.predict_proba(x_dev)[:, 1]
    print(sklearn.metrics.roc_auc_score(y_dev, mlp_probs))

    # Best loss 0.5541490912437439 with accuracy 0.7410071942446043 and auc 0.8183999451566463
    cnn = CNN()
    state_dict = torch.load("cnn.pt")
    cnn.load_state_dict(state_dict)
    cnn.eval()
    loader = DataLoader(MyDataset(x_dev, y_dev), batch_size=x_dev.shape[0], shuffle=False)
    for inputs, _ in loader:
        inputs = inputs.reshape((inputs.shape[0], 1, inputs.shape[1]))
        cnn_probs = cnn(inputs)[:, 1].detach().numpy()
    print(sklearn.metrics.roc_auc_score(y_dev, cnn_probs))

    polling_train = np.vstack((knn_probs, mlp_probs, cnn_probs)).reshape(5557, 3)
    polling_train = np.hstack((polling_train, add_train))

    polling = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=[32, 8],
                                                   activation='relu',
                                                   solver='adam',
                                                   alpha=1e-4,
                                                   batch_size=64,
                                                   learning_rate='invscaling',
                                                   learning_rate_init=1e-3
                                                   )
    polling.fit(polling_train, y_dev)
    yhat = polling.predict_proba(polling_train)[:, 1]
    print(sklearn.metrics.roc_auc_score(y_dev, yhat))

    knn_probs = knn.predict_proba(x_test)[:, 1]
    mlp_probs = mlp.predict_proba(x_test)[:, 1]
    cnn_probs = cnn(torch.tensor(x_test.reshape((x_test.shape[0], 1, x_test.shape[1])).astype(np.float32)))[:, 1].detach().numpy()

    polling_test = np.vstack((knn_probs, mlp_probs, cnn_probs)).reshape(1197, 3)
    polling_test = np.hstack((polling_test, add_test))
    yhat = polling.predict_proba(polling_test)[:, 1]
    np.savetxt('yproba_polling.txt', yhat)


if __name__ == "__main__":
    main()