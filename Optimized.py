import os
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import textwrap
import os


import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.feature_extraction.text import CountVectorizer

from matplotlib import pyplot as plt

import seaborn as sns
from load_BERT_embeddings import load_arr_from_npz 



if __name__ == '__main__':
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    N, n_cols = x_train_df.shape
    print("Shape of x_train_df: (%d, %d)" % (N, n_cols))
    print("Shape of y_train_df: %s" % str(y_train_df.shape))

    # Print out 8 random entries
    tr_text_list = x_train_df['text'].values.tolist()
    prng = np.random.RandomState(101)
    rows = prng.permutation(np.arange(y_train_df.shape[0]))
    for row_id in rows[:8]:
        text = tr_text_list[row_id]
        print("row %5d | %s BY %s | y = %s" % (
            row_id,
            y_train_df['title'].values[row_id],
            y_train_df['author'].values[row_id],
            y_train_df['Coarse Label'].values[row_id],
            ))

        line_list = textwrap.wrap(tr_text_list[row_id],
            width=70,
            initial_indent='  ',
            subsequent_indent='  ')
        print('\n'.join(line_list))
        print("")


def make_logit_pipeline_knnimpute(C=1.0, k=5):
    pipeline = sklearn.pipeline.Pipeline(
        steps=[
         #('imputer', sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')),
         ('imputer', sklearn.impute.KNNImputer(n_neighbors=k)),
         ('rescaler', sklearn.preprocessing.MinMaxScaler()),
         ('logit', sklearn.linear_model.LogisticRegression(solver="lbfgs", l1_ratio=0, C=C))
        ])
    return pipeline

#get text and target
tr_list_of_text = x_train_df['text'].values.tolist()
y_labels = y_train_df["Coarse Label"].tolist()
x_train = x_train_df.drop(columns = ["author", "title" , "passage_id", "text"])

RANDOM_SEED = 68
 
# Create a train/validation/test split
X_dev, X_test, y_dev, y_test = sklearn.model_selection.train_test_split(
    x_train, y_labels, test_size=0.15, random_state=RANDOM_SEED, stratify=y_labels
)



y_dev = np.array(y_dev)
X_dev = np.array(X_dev)
#Optimize C with cross validation 
max_auc = 0
best_c = 0
kf = sklearn.model_selection.KFold(n_splits= 10, shuffle=True, random_state=RANDOM_SEED)
for C in np.logspace(-4, 4, 17): 
    for k in range(1,18):
        auc_sum = 0
        for train_ind, val_ind in kf.split(X_dev, y_dev):
            pipe = make_logit_pipeline_knnimpute(C=C, k=k)
            pipe.fit(X_dev[train_ind], y_dev[train_ind])
            y_hat = pipe.predict_proba(X_dev[val_ind])[:, 1]
            auc = sklearn.metrics.roc_auc_score(y_dev[val_ind], y_hat)
            auc_sum+=auc
            print(auc)
        avg_auc = auc_sum/10
        if avg_auc == max_auc:
                if C < best_c:
                    best_c = C
                    max_auc = avg_auc

        if avg_auc > max_auc:
            best_c = C
            max_auc = avg_auc

print ("Best C:", best_c)
print ("Best AUC:", max_auc)

pipe = make_logit_pipeline_knnimpute(C= best_c)