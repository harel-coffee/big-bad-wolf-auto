import argparse
import os

import lxml.etree
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler


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

vectorizer = FeatureUnion([
    ('raw_chars', TfidfVectorizer(analyzer='char', ngram_range=(1, 4),
                                  lowercase=False, use_idf=True, min_df=2)),
    ('clean_chars', TfidfVectorizer(analyzer='char', ngram_range=(1, 4), min_df=2)),
    ('puctuation', TfidfVectorizer(analyzer='word',  use_idf=False,
                                   token_pattern=r'[^\w\s]+')),
])

X = [stories[id] for id in knowns]
vectorizer.fit(stories.values())
X = vectorizer.transform(X)
y = np.array(dates)

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, knowns, test_size=0.1, shuffle=True, random_state=1983)

# Average CV score on the training set was:0.8450677049610829
regressor_pipeline = make_pipeline(
    # SelectPercentile(score_func=f_regression, percentile=82),
    # SelectPercentile(score_func=f_regression, percentile=16),
    LinearSVR(C=25.0, dual=True, epsilon=0.0001, loss="epsilon_insensitive", tol=0.1)
)

y_scaler = RobustScaler()
regressor = TransformedTargetRegressor(regressor=regressor_pipeline,
                                       transformer=y_scaler)

regressor.fit(X_train, y_train)
test_pred = regressor.predict(X_test)

abs_error = np.abs(y_test - test_pred)
print("Mean Absolute Error", mean_absolute_error(y_test, test_pred))
print("R2", r2_score(y_test, test_pred))

regressor.fit(X, y)

X_target = [stories[id] for id in unknowns]
X_target = vectorizer.transform(X_target)

preds = regressor.predict(X_target)
print("Mean Absolute Error", mean_absolute_error(unkdates, preds))
print("R2", r2_score(unkdates, preds))

