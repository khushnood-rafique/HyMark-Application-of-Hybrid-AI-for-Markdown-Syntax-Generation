#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd

def preprocess(x):
    return pd.Series(x).replace(r'\b([A-Za-z])\1+\b', '', regex=True)\
        .replace(r'\b[A-Za-z]\b', '', regex=True)

