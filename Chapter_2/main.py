from housing_data import load_housing_data
import numpy as np

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