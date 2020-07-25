# %%
import pandas as pd
from typing import List

# %%
def prep(x: pd.DataFrame) -> pd.DataFrame:
  """Drops all columns except for stars, text and date and optimizes datatypes."""
  x['text'] = x['text'].astype('string')
  x['stars'] = x['stars'].astype('category')
  return x[['stars', 'text', 'date']]

# %%
"""
Loads raw json file of reviews and preprocesses it using the prep function.
CAUTION: Long running time!
"""
path_to_json = "../data/yelp_academic_dataset_review.json"

df: pd.DataFrame = pd.read_json(path_to_json, lines=True, chunksize=8192)
preped_chunks: List[pd.DataFrame] = []

for chunk in df:
    preped_chunks.append(prep(chunk))

data = pd.concat(preped_chunks)

data.to_pickle('../data/reviews_optimized.pickle')
print("Sucessfully saved to disk!")