import argparse
import os

import lxml.etree
import numpy as np
import pandas as pd

from tpot import TPOTRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from tpot_config import tpot_config


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

vectorizer.fit(stories.values())
X = [stories[id] for id in knowns]
X = vectorizer.transform(X)
y = np.array(dates)

tpot = TPOTRegressor(generations=100, population_size=100, verbosity=3, cv=10,
                     config_dict=tpot_config, njobs=30, scoring='neg_mean_absolute_error',
                     periodic_checkpoint_folder='tpot')
tpot.fit(X, y)
tpot.export('tpot_datereg.py')
