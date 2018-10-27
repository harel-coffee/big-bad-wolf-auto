import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import lxml.etree
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler
import unidecode

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--n_jobs', type=int, default=1)
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

X = [unidecode.unidecode(' '.join(stories[id].split())) for id in knowns]
y = np.array(dates)

pipeline = Pipeline([
    ('vec', TfidfVectorizer()),
    ('tt_svr', TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=MinMaxScaler()))
])

grid = {
    'vec__analyzer': ['char', 'char_wb'],
    'vec__ngram_range': [(2, 4), (2, 5)],
    'vec__min_df': [1, 5, 10, 20, 50],
    'vec__sublinear_tf': [True, False],
    'vec__max_df': [1.0, 0.9, 0.8, 0.7],
    'vec__use_idf': [True, False]
}

bins = np.linspace(1840, 2020, 15)
y_binned = np.digitize(y, bins)
skf = StratifiedKFold(n_splits=5).split(np.zeros(y.shape[0]), y_binned)

grid_cv = GridSearchCV(
    pipeline, grid, scoring=['r2', 'neg_mean_absolute_error'],
    refit='r2', cv=skf, n_jobs=args.n_jobs, verbose=1, error_score=np.nan)

grid_cv.fit(X, y)
results = pd.DataFrame(grid_cv.cv_results_)
results.to_csv("results-lr.csv")
