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

fig = plt.figure()
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
fig = plt.figure()
fig = housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.savefig("./figs/income_categories.pdf")
plt.close()
print()
print("Created a barplot of the housing income categories in /figs named income_categories.pdf")
