from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

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