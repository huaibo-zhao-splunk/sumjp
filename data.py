from pathlib import Path
import re
import time
import requests
import pandas as pd
import urllib
from bs4 import BeautifulSoup
from tqdm import tqdm

data_dir_path = Path('/home/ubuntu/20220812/data')

def download_data(url, data_dir_path):
    
    file_path = data_dir_path.joinpath(Path(url).name)

    data = requests.get(url).content
    with open(file_path, 'wb') as file:
        file.write(data)

url = 'https://raw.githubusercontent.com/KodairaTomonori/ThreeLineSummaryDataset/master/data/train.csv'
download_data(url=url, data_dir_path=data_dir_path)
print("train data downloaded")

url = 'https://raw.githubusercontent.com/KodairaTomonori/ThreeLineSummaryDataset/master/data/test.csv'
download_data(url=url, data_dir_path=data_dir_path)
print("test data downloaded")