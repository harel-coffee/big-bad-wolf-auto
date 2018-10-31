import argparse
import random
import os

import lxml.etree
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.utils import shuffle

import tqdm
import unidecode


def split_into_chunks(x, size=3000):
    for i in range(0, len(x), size):
        chunk = x[i: i + size]
        if len(chunk) == size:
            yield chunk


def error_plot(y_test, y_pred, fname):
    plt.clf()
    label = "$R^2 = {:0.3f}$".format(r2_score(y_test, y_pred))
    ax = sns.scatterplot(y_test, y_pred, hue=abs(y_test - y_pred),
                         legend=False, label=label, s=25)
    ax.set_ylim(1825, 2020)
    ax.set_xlim(1825, 2020)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", label='ideal fit')
    ax.set_xlabel("True years")
    ax.set_ylabel("Predicted years")
    sns.despine(ax=ax, offset=10)
    ax.set_aspect(1.)
    plt.legend(loc="upper left")

    divider = make_axes_locatable(ax)
    residuals = y_pred - y_test
    rax = divider.append_axes("bottom", size=0.7, pad=0.6, sharex=ax)
    sns.scatterplot(y_pred, residuals, hue=abs(residuals), label=label,
                    legend=False, ax=rax, s=25)
    rax.axhline(y=0, color="black", ls="--", c=".3")
    rax.set_ylim(-70, 70)
    rax.set_ylabel("Residuals")
    rax.set_xlabel("Predicted years")
    sns.despine(ax=rax, offset=10)
    plt.savefig(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df[df['reprint'] == 'no']

    stories, ns = {}, {'tei': 'http://www.tei-c.org/ns/1.0'}
    for entry in os.scandir(args.corpus):
        if entry.name.endswith('.xml'):
            tree = lxml.etree.parse(entry.path)
            text = tree.find('//tei:text', namespaces=ns)
            stories[entry.name[:-4]] = ' '.join(text.itertext())

    knowns, dates = zip(*df[df['exact_date']][['id', 'year_corrected']].values)
    unknowns, unkdates = zip(*df[~df['exact_date']][['id', 'year_corrected']].values)

    vectorizer = TfidfVectorizer(
        analyzer='char', ngram_range=(1, 4), max_df=1.0,
        lowercase=False, sublinear_tf=True, min_df=2, norm='l1', use_idf=False
    )

    X = [unidecode.unidecode(' '.join(stories[id].split())) for id in knowns]
    X, dates = zip(*[(s, d) for x, d in zip(X, dates) for s in split_into_chunks(x)])
    vectorizer.fit(X)
    X = vectorizer.transform(X)
    y = np.array(dates).astype(np.float64)
    X, y = shuffle(X, y, random_state=1900)

    regressor = make_pipeline(
        SelectPercentile(score_func=f_regression, percentile=67),
        ElasticNetCV(l1_ratio=0.85, tol=0.01, cv=5))

    y_scaler = RobustScaler()
    regressor = TransformedTargetRegressor(
        regressor=regressor, transformer=y_scaler)

    bins = np.linspace(1840, 2020, 5)
    y_binned = np.digitize(y, bins)
    skf = StratifiedKFold(n_splits=5, random_state=1900)
    trues, preds = [], []
    for train_index, test_index in tqdm.tqdm(skf.split(np.zeros(y.shape[0]), y_binned)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        y_pred[y_pred > y.max()] = y.max()
        trues.extend(list(y_test))
        preds.extend(list(y_pred))

    y_test, y_pred = np.array(trues), np.array(preds)
    print("R2 ", r2_score(y_test, y_pred))
    AE = np.abs(y_test - y_pred)
    print("MAE", AE.mean(), AE.std())

    error_plot(y_test, y_pred, "scatterplot-train.pdf")

    regressor.fit(X, y)
    X_target = [unidecode.unidecode(' '.join(stories[id].split())) for id in unknowns]
    X_target = vectorizer.transform(X_target)
    y_test = np.array(unkdates) 

    y_pred = regressor.predict(X_target)
    y_pred[y_pred > y.max()] = y.max()
    AE = np.abs(y_test - y_pred)
    print("MAE", AE.mean(), AE.std())

    error_plot(y_test, y_pred, "scatterplot-eval.pdf")

    preds = {id: pred for id, pred in zip(unknowns, y_pred)}
    df['year_estimated'] = df['year_corrected']
    unk_rows = df['id'].isin(unknowns)
    df.loc[unk_rows, 'year_estimated'] = df.loc[unk_rows, 'id'].apply(lambda id: preds.get(id))
    assert (df.loc[~unk_rows, 'year_corrected'] == df.loc[~unk_rows, 'year_estimated']).all()
    df.fillna('').to_csv(args.dataset, index=False)
