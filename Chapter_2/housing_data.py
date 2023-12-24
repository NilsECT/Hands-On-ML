from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

# creating the function to fetch the data separately use by calling
# from housing_data import load_housing_data

def load_housing_data():
    '''
    Fetches the housing data ffor chapter 2 of HOML
    '''

    # bellow is checking if the file exists, if it does not it goes to fetch it
    # if the file already exists, none matters and you go straight to return
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        # if it doesn't find the path or anything at the path

        Path("datasets").mkdir(parents=True, exist_ok=True)
        # creates the parents, like using mkdir normally.
        # exist_ok is to allow for existing names that are non-directories
        
        # fetching the data from the web
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        # retrieves the tarfile and puts it in the defined path

        # once we have fetched the tarfile we can extract its contents
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")

    return pd.read_csv(Path("datasets/housing/housing.csv"))

# to take a look at the data follow the following
# using the data and such will not be done here
# some or all of the following might be in the main project file

housing = load_housing_data()

# printing the first five rows using pd.head
print("Showing the top five rows of the housing data:")
print(housing.head())

print()
# printing info from each column
print("Showing the info for all the columns:")
print(housing.info())

print()
# ocean_proximity is dtype Object, take a closer look at it
print("Taking a closer look at the ocean_proximity feature:")
print(housing["ocean_proximity"].value_counts())

print()
# summary of the numerical attributes
print("Showing a numerical summary of the attributes (features):")
print(housing.describe())

# plotting the data in histograms, this can be done directly from the dataframe!
# needs matplotlib.pyplot

import matplotlib.pyplot as plt

fig = housing.hist(bins=50, figsize=(14, 10))
# plt.show()
plt.savefig("./figs/housing_hist.pdf")
plt.close()
print()
print("Created a histogram plot of the data in /figs named housing_hist.pdf")

# the housing income is an important feature to determine the housing price
# so in the test set this should be well represented
import numpy as np

# take the dataset and divide the median income into 5 groups
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
# this splits into 5 categories where category 1 contains all instances of values between 0 and 1.5
# category contains instances of values between 1.5 to 3 and so on

# How do these categories look?
fig = housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.savefig("./figs/income_categories.pdf")
plt.close()
print()
print("Created a barplot of the housing income categories in /figs named income_categories.pdf")

# can use StratifiedShuffledSplit from scikit-learn now or you can do it as in line 13 in main.py
# here I write code for future reference, as it seems good for cross-validation

from sklearn.model_selection import StratifiedShuffleSplit

# generates 10 different splits
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []

for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]

    strat_splits.append([strat_train_set_n, strat_test_set_n])

# use the first split
strat_train_set, strat_test_set = strat_splits[0]

# see if the splitting was done in correct proportions
print()
print("Taking a look at the proportions of each income category in the test set:")
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

##################################################################
# We've looked at the data set with a bird's eye view and split it
# We will now take a closer look at the data
##################################################################

# Visualise based on geographical location and density of points: 
fig = housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
# alpha is the blending value which lets you see the density of points.
# The blending is based on how many points are overlapping.
plt.savefig("./figs/housing_geo_density.pdf")
plt.close()
print()
print("Created a scatterplot of the geographic location of the datapoints with blending to see the density in /figs named housing_geo_density.pdf")

# Now to see housing price with geographic location and population size
fig = housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"] / 100, label="population", c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(13, 10))
# s is for the size of the datapoints in the scatterplot
# c is for their colour
plt.savefig("./figs/housing_geo_val_pop.pdf")
plt.close()
print()
print("Created a scatterplot of the geographic location of the datapoints visualising both the population size and the median housing value in /figs named housing_geo_val_pop.pdf")

# taking a look at the correlation
# Pearson's r coefficient is the standard
# we use Pearson's becuase we don't have too large a dataset

# we can look at correlations through a correlation matix
# Ocean proximity is still strings, can one-hot-encode, this will be done later
corr_matrix = housing.corr(numeric_only=True)

print()
print("Correlation, using Pearson's r, of the median income. 1 is strongly correlated, 0 is not correlated and -1 is strongly negatively correlated:")
print(corr_matrix["median_income"].sort_values(ascending=False))

# Or we can look at the correlation plots using the pandas plotting
# can probably figure out how to plot using seaborn but integrated pandas plotting is neat
# to avoid having 11^2 plots we look at the most promising attributes based on the values from the correlation matrix

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

plt.savefig("./figs/housing_scatter_matrix.pdf")
plt.close()
print()
print("Created a correlation scatterplot of some housing attributes. The figure can be found at ./figs/housing_scatter_matrix.pdf")

# taking a closer look at correlations of the median house value and relative number of rooms, bedrooms and population

housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)

print()
print("Correlation, using Pearson's r, of median house value. There are some new, more useful, attributes:")
print(corr_matrix["median_house_value"].sort_values(ascending=False))

##################################################################
# We've explored the data
# We now go through a more detailed process of cleaning the data
# this process will be repeated with a pipeline in the main file (muuuuuch quicker)
##################################################################
