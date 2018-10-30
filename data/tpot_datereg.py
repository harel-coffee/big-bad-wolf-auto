import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import lxml.etree
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tpot import TPOTRegressor
import unidecode

from tpot_config import regressor_config_dict


parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--verbosity', type=int, default=2)
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

def split_into_chunks(x, size=3000):
    for i in range(0, len(x), size):
        chunk = x[i: i + size]
        if len(chunk) == size:
            yield chunk

vectorizer = TfidfVectorizer(
    analyzer='char', ngram_range=(1, 4), max_df=1.0,
    lowercase=False, sublinear_tf=True, min_df=2, norm='l1', use_idf=False
)

X = [unidecode.unidecode(' '.join(stories[id].split())) for id in knowns]
X, dates = zip(*[(s, d) for x, d in zip(X, dates) for s in split_into_chunks(x)])
y = np.array(dates)
X = vectorizer.fit_transform(X)
y = np.array(dates)
X, y = shuffle(X, y)

bins = np.linspace(1840, 2020, 15)
y_binned = np.digitize(y, bins)
skf = list(StratifiedKFold(n_splits=5).split(np.zeros(y.shape[0]), y_binned))

scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).squeeze()

tpot = TPOTRegressor(
    generations=100, population_size=100, verbosity=args.verbosity, cv=skf,
    config_dict='TPOT sparse',
    n_jobs=args.n_jobs, scoring='r2', periodic_checkpoint_folder='tpot'
)
tpot.fit(X, y)
tpot.export('tpot_datereg_pipeline.py')
