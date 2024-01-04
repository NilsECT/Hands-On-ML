from housing_data import load_housing_data
import numpy as np
import pandas as pd

# import data
housing = load_housing_data()

# stratify based on housing income (see housing_data.py lines 73-90)
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

# split data into training and testing sets taking care of the stratum of median housing income
from sklearn.model_selection import train_test_split

strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

# throw away income category column as you won't use it anymore
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# find the numerical features of the dataset
housing_num = housing.select_dtypes(include=[np.number])

# imputing the data

from sklearn.pipeline import make_pipeline  # calls fit_transform() sequentially until it reaches the last estimator where it calls fit()
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# pipeline for numerical features: impute and scale
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

housing_num_prepared = num_pipeline.fit_transform(housing_num)
df_housing_num_prepared = pd.DataFrame(housing_num_prepared, columns=num_pipeline.get_feature_names_out(), index=housing_num.index)

