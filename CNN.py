import os
import numpy as np
import pandas as pd
import os

import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from matplotlib import pyplot as plt
import seaborn as sns

from load_BERT_embeddings import load_arr_from_npz

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

RANDOM_SEED = 68
torch.manual_seed(RANDOM_SEED)

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.long)
    def __len__(self):
        return self.x.size()[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
# Best loss 0.5610169172286987 with accuracy 0.7338129496402878
# C(3, 1, 1, 8) -> C(3, 1, 8, 8) -> C(3, 1, 8, 8) -> MP(2, 2) -> Flatten -> D(256) -> D(128) -> D(2)

# Best loss 0.558468222618103 with accuracy 0.7473021582733813
# C(3, 1, 1, 8) -> C(3, 1, 8, 8) -> C(3, 1, 8, 8) -> MP(2, 2) -> Flatten -> D(512) -> D(2)

# Best loss 0.5541490912437439 with accuracy 0.7410071942446043 and auc 0.8183999451566463
# C(3, 1, 1, 8) BN -> C(3, 1, 8, 8) BN -> C(3, 1, 8, 8) BN -> MP(2, 2) -> Flatten -> D(512) -> D(2)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.LazyConv1d(out_channels=8, kernel_size=3),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyConv1d(out_channels=8, kernel_size=3),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyConv1d(out_channels=8, kernel_size=3),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.LazyLinear(out_features=512),
            nn.ReLU(),
            nn.LazyLinear(out_features=2)
        )

    def forward(self, x):
        logits = self.seq(x)
        probs = F.softmax(logits, dim=1)
        return probs

def train(x_train, y_train, x_val, y_val, model, num_train_epochs, batch_size, lr, weight_decay):
    """
    args:
      x_train: `np.array((N, D))`, training data of N instances and D features.
      y_train: `np.array((N, C))`, training labels of N instances and C fitting targets 
      x_val: `np.array((N1, D))`, validation data of N1 instances and D features.
      y_val: `np.array((N1, C))`, validation labels of N1 instances and C fitting targets 
      model: a torch module
      num_train_epochs: int, the number of training epochs.
      batch_size: int, the batch size 
      lr: float, learning rate
      weight_decay: float, weight decay for regularization 
    """
    trainloader = DataLoader(MyDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(MyDataset(x_val, y_val), batch_size=y_val.shape[0], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_func = torch.nn.CrossEntropyLoss()

    history = {"loss": [],
              "val_loss": [],
              "accuracy": [],
              "auc": []}

    best_val_loss = float('inf')
    best_accuracy = 0
    best_auc = 0

    for epoch in range(num_train_epochs):
        epoch_loss = 0
        for data in trainloader:
            model.train()
            inputs, labels = data
            inputs = inputs.reshape((inputs.shape[0], 1, inputs.shape[1]))

            optimizer.zero_grad()

            preds = model(inputs)
            loss = loss_func(preds, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        with torch.no_grad():
            for data in validationloader:
                model.eval()
                inputs, labels = data
                inputs = inputs.reshape((inputs.shape[0], 1, inputs.shape[1]))
                preds = model(inputs)
                val_loss = loss_func(preds, labels).detach()

                accuracy = sklearn.metrics.accuracy_score(labels.numpy(), torch.argmax(preds, dim=1).numpy())
                auc = sklearn.metrics.roc_auc_score(labels.numpy(), preds[:, 1])

                history['loss'].append(epoch_loss / (y_train.shape[0] / batch_size))
                history['val_loss'].append(val_loss)
                history['accuracy'].append(accuracy)
                history['auc'].append(auc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_accuracy = accuracy
                    best_auc = auc
                    torch.save(model.state_dict(), "cnn.pt")

                # Early stopping
                if len(history['val_loss']) >= 30:
                    last30 = history['val_loss'][-30:]
                    if best_val_loss not in last30:
                        print(f"Best loss {best_val_loss} with accuracy {best_accuracy} and auc {best_auc}")
                        return history

        if epoch == 0:
            print(f"Epoch [1/{num_train_epochs}], Train Loss: {history['loss'][epoch]:.4f}, Val Loss: {history['val_loss'][epoch]:.4f}, Acc Score: {history['accuracy'][epoch]:.4f}, AUC: {history['auc'][epoch]:.4f}")
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_train_epochs}], Train Loss: {history['loss'][epoch]:.4f}, Val Loss: {history['val_loss'][epoch]:.4f}, Acc Score: {history['accuracy'][epoch]:.4f}, AUC: {history['auc'][epoch]:.4f}")

    return history

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

    return x_train, y_train, x_test

def visualize(history):
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(history['accuracy'], label='val accuracy')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.show()

def predict(x_test):
    state_dict = torch.load("cnn.pt")
    model = CNN()
    model.load_state_dict(state_dict)
    model.eval()
    y_probs = model(torch.tensor(x_test.reshape((x_test.shape[0], 1, x_test.shape[1])).astype(np.float32)))[:, 1]
    np.savetxt('yproba_cnn.txt', y_probs.detach().numpy())

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
    x_dev, y_dev, x_test = load_data(metrics)
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_dev, y_dev, test_size=0.20, random_state=RANDOM_SEED)
    model = CNN()
    history = train(x_train, y_train, x_val, y_val, model, num_train_epochs=500,
                           batch_size=64,
                           lr=1e-4,
                           weight_decay=1e-3)
    visualize(history)
    predict(x_test)

if __name__ == "__main__":
    main()