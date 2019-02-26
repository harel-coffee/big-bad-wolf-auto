# Cultural Entrenchment of Folktales is Encoded in Language 

[full paper](https://www.nature.com/articles/s41599-019-0234-9)

The repository hosts the code and data to reproduce the analysis presented in Karsdorp,
F. & Fonteyn, L. (2019). 'Cultural Entrenchment of Folktales is Encoded in Language'. All
data annotations can be found in the `data/` folder. The analyses are implemented in R and
can be found in `analyses/definite.R`. The script `scripts/datereg.py` can be used to
estimate dates of publication to the data collection.

## Requirements

### R (3.5.1)

```
OneR==2.2, bayesplot==1.6.0, brms==2.7.0, car==3.0-2, cowplot==0.9.3, data.table==1.11.4,
effects==4.0-3, gridExtra==2.3, lme4==1.1-18-1, projpred==1.1.0, purrr==0.2.5,
scales==1.0.0, sjstats==0.17.1, standardize==0.2.1, tidybayes==1.0.1.9000, tidyr==0.8.1
```

### Python (>=3.6)

```
lxml==4.2.1, matplotlib==3.0.2, numpy==1.15.4, pandas==0.23.4, seaborn==0.9.0, tqdm==4.28.1
```

## Citation

``` bibtex
@article{karsdorp_fonteyn:2019,
  author = {Folgert Karsdorp and Lauren Fonteyn},
  title = {Cultural Entrenchment is Encoded in Language},
  journal = {Palgrave Communications},
  volume = {5},
  year = {2019},
  doi = {10.1057/s41599-019-0234-9},
  url = {https://www.nature.com/articles/s41599-019-0234-9}
}
```
