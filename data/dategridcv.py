import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import lxml.etree
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler


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

X = [stories[id] for id in knowns]
y = np.array(dates)

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, knowns, test_size=0.1, shuffle=True, random_state=1983)

pipeline = Pipeline([
    ('vec', TfidfVectorizer()),
    ('tt_svr', TransformedTargetRegressor(
        regressor=Ridge(), transformer=RobustScaler()))
])

grid = {
    'vec__analyzer': ['char'],
    'vec__ngram_range': [(1, 3), (1, 4), (2, 3), (2, 4)],
    'vec__min_df': [1, 2, 5, 10, 20, 50],
    'vec__max_df': [1.0, 0.9, 0.8, 0.7],
    'vec__use_idf': [True, False],
    'vec__lowercase': [True, False],
    'tt_svr__regressor__alpha': np.linspace(0.0001, 0.1, 10),
    'tt_scr__regressor__tol': [1e-2, 1e-3, 1e-4]
}

grid_cv = GridSearchCV(
    pipeline, grid, scoring=['r2', 'neg_mean_absolute_error'],
    refit='r2', cv=10, n_jobs=args.n_jobs, verbose=1, error_score=np.nan)

grid_cv.fit(X_train, y_train)
results = pd.DataFrame(grid_cv.cv_results_)
results.to_csv("results-ridge.csv")

test_pred = grid_cv.predict(X_test)

abs_error = np.abs(y_test - test_pred)
print("Mean Absolute Error", mean_absolute_error(y_test, test_pred))
print("R2", r2_score(y_test, test_pred))

