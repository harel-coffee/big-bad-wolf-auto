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
from tpot import TPOTRegressor
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

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_df=0.8,
                             lowercase=False, sublinear_tf=True, min_df=20,
                             use_idf=False)

X = [stories[id] for id in knowns]
X = vectorizer.fit_transform(X)
y = np.array(dates)

bins = np.linspace(1840, 2020, 15)
y_binned = np.digitize(y, bins)
skf = StratifiedKFold(n_splits=5).split(np.zeros(y.shape[0]), y_binned)

scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).squeeze()

tpot = TPOTRegressor(generations=100, population_size=100, verbosity=3, cv=skf,
                     config_dict=regressor_config_sparse, n_jobs=30,
                     scoring='r2',
                     periodic_checkpoint_folder='tpot')
tpot.fit(X, y)
tpot.export('tpot_datereg_pipeline.py')
