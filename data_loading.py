# %%
import pandas as pd
from typing import List

df: pd.DataFrame = pd.read_json('~/Downloads/yelp_dataset/yelp_academic_dataset_review.json', lines=True, chunksize=8192)


def prep(x: pd.DataFrame) -> pd.DataFrame:
    x['text'] = x['text'].astype('string')
    x['stars'] = x['stars'].astype('category')
    return x[['stars', 'text', 'date']]


preped_chunks: List[pd.DataFrame] = []

for chunk in df:
    preped_chunks.append(prep(chunk))

#%%
final = pd.concat(preped_chunks)
