# %%
import pandas as pd
from typing import List


def prepare_raw_data(path_to_json: str = None) -> pd.DataFrame:
    """
    Loads raw json file of reviews and preprocesses it using the prep function.
    CAUTION: Long running time!
    """
    if path_to_json is None:
        path_to_json = "~/Downloads/yelp_dataset/yelp_academic_dataset_review.json"

    df: pd.DataFrame = pd.read_json(path_to_json, lines=True, chunksize=8192)
    preped_chunks: List[pd.DataFrame] = []

    for chunk in df:
        preped_chunks.append(prep(chunk))

    return pd.concat(preped_chunks)


def prep(x: pd.DataFrame) -> pd.DataFrame:
    """Drops all columns except for stars, text and date and optimizes datatypes."""
    x['text'] = x['text'].astype('string')
    x['stars'] = x['stars'].astype('category')
    return x[['stars', 'text', 'date']]
