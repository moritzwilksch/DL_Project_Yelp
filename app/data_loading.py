# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# %%
path = "../data/"

def stars_to_category(stars: int) -> int:
    """Converts stars of 1 to 5 to categories negative (-1), neutral (0), positive (1)"""
    return -1 if stars <= 2 else 0 if stars == 3 else 1

"""
Subsamples dataset. The train set will have balanced classe, class distribution in validation and test set will follow the original distribution.
:param total_num_samples: Total number of samples over all 3 splits
:param train_ratio: Percent of `total_num_samples` to put in training set
:param validation_ratio: Percent of `total_num_samples` to put in validation set
:param test_ratio: Percent of `total_num_samples` to put in test set
:return: Dictionary with keys x_train, y_train, x_val, y_val, x_test, y_test each containing a dataframe (x_) or Series (y_)
"""

total_num_samples: int = 400_000
train_ratio: float = 0.75
validation_ratio: float = 0.15
test_ratio: float = 0.1

# %%
if train_ratio + validation_ratio + test_ratio != 1:
    raise ValueError("Train-, Validation- and Testratio have to sum to 1!")

path_to_pickle = path + 'reviews_optimized.pickle'

df = pd.read_pickle(path_to_pickle)

# %%
x_train, x_test, y_train, y_test = train_test_split(
    df.drop('stars', axis=1),
    df['stars'],
    test_size=1 - train_ratio
)

# Map stars to categories: positive, neutral, negative
y_train = y_train.apply(stars_to_category)
y_test = y_test.apply(stars_to_category)

# Split the (too) large test set into validation and test
x_val, x_test, y_val, y_test = train_test_split(
    x_test,
    y_test,
    test_size=test_ratio / (test_ratio + validation_ratio)
)

# %%
# Group by stars. Subsample an equal number of samples from each group
groups = x_train.groupby(y_train)
samples = []

for _, group in groups:
    samples.append(group.sample(int(total_num_samples * train_ratio // 3)))

x_train = pd.concat(samples)
y_train = y_train[x_train.index]

# Randomly sumbsample validation data
x_val = x_val.sample(int(total_num_samples * validation_ratio))
y_val = y_val[x_val.index]

# Randomly sumbsample test data
x_test = x_test.sample(int(total_num_samples * test_ratio))
y_test = y_test[x_test.index]

print("Successfully split data! Train/Validation/Test-Shapes are:")
for split in [x_train, x_val, x_test]:
    print(split.shape)

# %%
data = {
    'x_train': x_train,
    'y_train': y_train,
    'x_val': x_val,
    'y_val': y_val,
    'x_test': x_test,
    'y_test': y_test
}

joblib.dump(data, path + '3c_subsampled_data.pickle')
print("Sucessfully saved to disk!")
