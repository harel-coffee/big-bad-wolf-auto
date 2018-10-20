import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import lxml.etree
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from sklearn.utils import shuffle


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

X = [stories[id] for id in knowns]
y = np.array(dates)

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, knowns, test_size=0.1, shuffle=True, random_state=1983)

vectorizer = FeatureUnion([
        ('raw_chars', TfidfVectorizer(analyzer='char', ngram_range=(1, 4),
                                      lowercase=False, use_idf=True, min_df=2)),
        ('clean_chars', TfidfVectorizer(analyzer='char', ngram_range=(1, 4), min_df=2)),
        ('puctuation', TfidfVectorizer(analyzer='word',  use_idf=False,
                                       token_pattern=r'[^\w\s]+'))])

X_train = vectorizer.fit_transform(X_train)

pipeline = Pipeline([
    ('tt_svr', TransformedTargetRegressor(regressor=LinearSVR(),
                                          transformer=RobustScaler()))
])

grid = {
    'tt_svr__regressor__loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
    'tt_svr__regressor__dual': [True, False],
    'tt_svr__regressor__tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'tt_svr__regressor__C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
    'tt_svr__regressor__epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
}

grid_cv = GridSearchCV(
    pipeline, grid, scoring=['r2', 'neg_mean_absolute_error'],
    refit='r2', cv=10, n_jobs=4, verbose=1, error_score=0.0)

grid_cv.fit(X_train, y_train)
results = pd.DataFrame(grid_cv.cv_results_)
results.to_csv("results-svr.csv")

X_test = vectorizer.transform(X_test)
test_pred = grid_cv.predict(X_test)

abs_error = np.abs(y_test - test_pred)
print("Mean Absolute Error", mean_absolute_error(y_test, test_pred))
print("R2", r2_score(y_test, test_pred))

