import argparse
import os

import lxml.etree
import numpy as np
import pandas as pd

import sklearn.linear_model as lm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.metrics


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
unknowns = df[~df['exact_date']]['id'].values

vectorizer = TfidfVectorizer(analyzer='char', norm='l2', ngram_range=(2, 5), min_df=2)

X = [stories[id] for id in knowns]
X = vectorizer.fit_transform(X)
scaler = MinMaxScaler()
y = scaler.fit_transform(np.array(dates).reshape(-1, 1))[:,0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=True, random_state=1983)

tpot = TPOTRegressor(generations=100, population_size=100, verbosity=2,
                     config_dict='TPOT sparse', njobs=40, scoring='r2',
                     periodic_checkpoint_folder='tpot')
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_datereg.py')
