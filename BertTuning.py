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
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer,  AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate 
from transformers import DataCollatorWithPadding
from scipy.special import softmax




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

def fine_tune_bert(x_train_df, y_train_df, x_val_df, y_val_df, epochs):
    import torch
    model_path = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    for name, param in model.base_model.named_parameters():
        param.requires_grad = False
    for name, param in model.base_model.named_parameters():
        if "pooler" in name:
            param.requires_grad = True

    def tokenize(texts):
        return tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    train_labels = torch.tensor([0 if l == "Key Stage 2-3" else 1 for l in y_train_df["Coarse Label"].tolist()])

    model.train()
    for epoch in range(epochs):
        inputs = tokenize(x_train_df["text"].tolist())
        inputs["labels"] = train_labels
        loss = model(**inputs).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        inputs = tokenize(x_val_df["text"].tolist())
        probs = softmax(model(**inputs).logits.numpy(), axis=1)[:, 1]
    val_labels = np.array([0 if l == "Key Stage 2-3" else 1 for l in y_val_df["Coarse Label"].tolist()])
    return sklearn.metrics.roc_auc_score(val_labels, probs)


def hyperparameter_selection(x_train_df, y_train_df):
    labels = np.array([0 if l == "Key Stage 2-3" else 1 for l in y_train_df["Coarse Label"].tolist()])
    authors = x_train_df["author"].values
    kf = sklearn.model_selection.GroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    max_auc, best_epochs = 0, 0

    for epochs in [1, 2, 3]:
        auc_sum = 0
        for train_ind, val_ind in kf.split(x_train_df, labels, authors):
            auc = fine_tune_bert(
                x_train_df.iloc[train_ind], y_train_df.iloc[train_ind],
                x_train_df.iloc[val_ind], y_train_df.iloc[val_ind],
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


def test_prediction(x_train_df, y_train_df, x_test_df, best_epochs):
    import torch
    model_path = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    for name, param in model.base_model.named_parameters():
        param.requires_grad = False
    for name, param in model.base_model.named_parameters():
        if "pooler" in name:
            param.requires_grad = True

    def tokenize(texts):
        return tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    train_labels = torch.tensor([0 if l == "Key Stage 2-3" else 1 for l in y_train_df["Coarse Label"].tolist()])

    model.train()
    for epoch in range(best_epochs):
        inputs = tokenize(x_train_df["text"].tolist())
        inputs["labels"] = train_labels
        loss = model(**inputs).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        inputs = tokenize(x_test_df["text"].tolist())
        probs = softmax(model(**inputs).logits.numpy(), axis=1)[:, 1]
    np.savetxt('yproba1_test.txt', probs)


def main():
    print("start")
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
    #best_epochs = hyperparameter_selection(x_train_df, y_train_df)
    #test_prediction(x_train_df, y_train_df, x_test_df, best_epochs)
    x_tr, x_val, y_tr, y_val = sklearn.model_selection.train_test_split(
        x_train_df, y_train_df, test_size=0.1, random_state=RANDOM_SEED)
    
    auc = fine_tune_bert(x_tr, y_tr, x_val, y_val, epochs=1)
    print("AUC:", auc)

if __name__ == "__main__":
        main()