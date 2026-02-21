import os
import numpy as np
import pandas as pd
import os

import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.feature_extraction.text import CountVectorizer

RANDOM_SEED = 68

def load_data():
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    N, n_cols = x_train_df.shape
    # print("Shape of x_train_df: (%d, %d)" % (N, n_cols))
    # print("Shape of y_train_df: %s" % str(y_train_df.shape))

    # Print out 8 random entries
    tr_text_list = x_train_df['text'].values.tolist()
    prng = np.random.RandomState(101)
    rows = prng.permutation(np.arange(y_train_df.shape[0]))
    # for row_id in rows[:8]:
    #     text = tr_text_list[row_id]
    #     print("row %5d | %s BY %s | y = %s" % (
    #         row_id,
    #         y_train_df['title'].values[row_id],
    #         y_train_df['author'].values[row_id],
    #         y_train_df['Coarse Label'].values[row_id],
    #         ))

    #     line_list = textwrap.wrap(tr_text_list[row_id],
    #         width=70,
    #         initial_indent='  ',
    #         subsequent_indent='  ')
    #     print('\n'.join(line_list))
    #     print("")

    return x_train_df, y_train_df


def hyperparameter_selection(x_train_df, y_train_df):
    #get text and target
    tr_list_of_text = x_train_df['text'].values.tolist()
    y_labels = y_train_df["Coarse Label"].tolist()

    best_num_feats, best_max_df, best_min_df, max_auc, best_c = 100, 0, 0, 0, 0
    for num_feats in np.linspace(1000, 10000, 10, dtype=int).tolist() + [None]:
        for min_df in np.linspace(1, 10, 10, dtype=int):
            for max_df in np.linspace(0.1, 1, 19):
                #turn into vector, tokenize text, remove stop words, remove punctuation and non alphabet symbols 
                vectorizer = CountVectorizer(
                    lowercase=True,
                    token_pattern=r'\b[a-z]+\b',
                    stop_words='english',
                    min_df=min_df,
                    max_df=max_df,
                    max_features=num_feats
                )

                X_train = vectorizer.fit_transform(tr_list_of_text)
                    
                # Create a train/validation/test split
                X_dev, X_test, y_dev, y_test = sklearn.model_selection.train_test_split(
                    X_train, y_labels, test_size=0.15, random_state=RANDOM_SEED, stratify=y_labels
                )
                y_dev = np.array(y_dev)

                #Optimize C with cross validation 
                kf = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
                for c in np.logspace(-4, 4, 17):
                    auc_sum = 0
                    for train_ind, val_ind in kf.split(X_dev, y_dev):
                        pipe = sklearn.linear_model.LogisticRegression(solver="liblinear", l1_ratio=1.0, C=c)
                        pipe.fit(X_dev[train_ind], y_dev[train_ind])
                        y_hat = pipe.predict_proba(X_dev[val_ind])[:, 1]
                        auc = sklearn.metrics.roc_auc_score(y_dev[val_ind], y_hat)
                        auc_sum += auc
                    avg_auc = auc_sum / 10

                    print(f"AUC {avg_auc:.6f} @ Max Feats {num_feats}, min_df {min_df}, max_df {max_df:.2f}, c {c:e} ")

                    # Prioritize fewer features, then a lower c value
                    if avg_auc > max_auc:
                        best_c, max_auc, best_num_feats, best_max_df, best_min_df = c, avg_auc, num_feats, max_df, min_df
                    elif avg_auc == max_auc:
                        if num_feats == None and best_num_feats != None:
                            pass
                        elif num_feats != None and best_num_feats != None:
                            if num_feats < best_num_feats:
                                best_c, max_auc, best_num_feats, best_max_df, best_min_df = c, avg_auc, num_feats, max_df, min_df
                            elif num_feats == best_num_feats and c < best_c:
                                best_c, max_auc, best_num_feats, best_max_df, best_min_df = c, avg_auc, num_feats, max_df, min_df
                        elif num_feats == None and best_num_feats == None:
                            if c < best_c:
                                best_c, max_auc, best_num_feats, best_max_df, best_min_df = c, avg_auc, num_feats, max_df, min_df

    print ("Best AUC:", max_auc)
    print ("Best C:", best_c)
    print ("Best num_feats:", best_num_feats)
    print ("Best max_df:", best_max_df)
    print ("Best min_df:", best_min_df)

    return best_c, best_num_feats, best_max_df, best_min_df

def test_prediction(x_train_df, y_train_df, c, num_feats, max_df, min_df):
    tr_list_of_text = x_train_df['text'].values.tolist()
    y_labels = y_train_df["Coarse Label"].tolist()

    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r'\b[a-z]+\b',
        stop_words='english',
        min_df=min_df,
        max_df=max_df,
        max_features=num_feats
    )

    X_dev = vectorizer.fit_transform(tr_list_of_text)
    pipe = sklearn.linear_model.LogisticRegression(solver="liblinear", l1_ratio=1.0, C=c)
    pipe.fit(X_dev, y_labels)
    y_hat = pipe.predict_proba(X_dev)[:, 1]
    np.savetxt('yproba1_test.txt', y_hat)

def main():
    x_train_df, y_train_df = load_data()
    c, num_feats, max_df, min_df = hyperparameter_selection(x_train_df, y_train_df)
    # best result: auc = 0.8213904877785245, c = 1.0, num_feats = None, max_df = 0.65, min_df = 2
    test_prediction(x_train_df, y_train_df, c, num_feats, max_df, min_df)

if __name__ == "__main__":
    main()