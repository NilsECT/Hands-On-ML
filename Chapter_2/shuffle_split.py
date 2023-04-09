# making a train test split function, quite simple
import numpy as np

'''
Note:
This is just training code. Better to use scikit-learn's methods and functions such as:
from sklearn.model_selection import train_test_split

or other more suiting functions. A team of professionals have worked on those and probably do it better than you alone.
'''

def shuffle_and_split(data, test_ratio):
    '''
    Takes the data, shuffles it and splits it into a training set and a test set according to the given ratio.
    The ratio is for how much of the dataset is going to be the test set.

    returns [train], [test]
    '''

    shuffled_indices = np.random.permutation(len(data))
    # makes a permutation of all the numbers from 0 to the length of the data-1

    test_size = int(len(data) * test_ratio)
    # finds out how many indices we need in the test set

    # now we can find the indices we want to put into each set
    test_indices = shuffled_indices[:test_size] # all indices until test set is full
    train_indices = shuffled_indices[test_size:]    # the rest

    # return the sets from the indices of each
    return data.iloc[train_indices], data.iloc[test_indices]

# to make sure that the test data never contains data that has been trained on (in previous runs or whatever):
# shuffle and choose based on instance's ID
# if they don't have an ID you've got to choose a way to ID them in a way that is unique and immutable

# this is for future reference, it doesn't look like chapter 2 in HOML uses the following functions

from zlib import crc32
'''
From https://www.geeksforgeeks.org/zlib-crc32-in-python/ (accessed april 9, 2023):

With the help of zlib.crc32() method, we can compute the checksum for crc32 (Cyclic Redundancy Check) to a particular data. It will give 32-bit integer value as a result by using zlib.crc32() method.
'''

def is_id_in_test_set(identifier, test_ratio):
    '''
    I'll be honest, I don't really know what's happening here.
    It is clearly a way to check whether an ID is in the test set.

    My interpretation without fundemental understanding:
    int64 turns the identifier into a 64bit integer.
    We check if the 32bit checksum is lower than the test ratio converted into a 32bit number.
    '''
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    '''
    See page 56 of HOML for explanation.

    Consistently splits into training and test sets based on ID so you don't ever have a test set containing data that has been trained on previously.
    '''
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))

    return data.loc[~in_test_set], data[in_test_set]    # the tilde means not

#############################################################
# For the housing dataset we give them an ID based on their location, that is latitude and longitude
# if you want to add this in the data set use:

from housing_data import load_housing_data

housing = load_housing_data()
housing["id"] = housing["longitude"] * 1000 + housing["latitude"]   # * 1000 to separate the values
