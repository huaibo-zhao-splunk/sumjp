from pathlib import Path
import re
import time
import requests
import pandas as pd
import urllib
from bs4 import BeautifulSoup
from tqdm import tqdm

data_dir_path = Path('/home/ubuntu/20220812/data')

# def download_data(url, data_dir_path):
    
#     file_path = data_dir_path.joinpath(Path(url).name)

#     data = requests.get(url).content
#     with open(file_path, 'wb') as file:
#         file.write(data)

# url = 'https://raw.githubusercontent.com/KodairaTomonori/ThreeLineSummaryDataset/master/data/train.csv'
# download_data(url=url, data_dir_path=data_dir_path)
# print("train data downloaded")

# url = 'https://raw.githubusercontent.com/KodairaTomonori/ThreeLineSummaryDataset/master/data/test.csv'
# download_data(url=url, data_dir_path=data_dir_path)
# print("test data downloaded")

def anti_join(data1, data2, by):

    joined_data = data1.copy()
    target_data = data2.copy()
    target_data['flag_tmp'] = 1

    if type(by) is str:
        by = [by]

    joined_data = pd.merge(
        joined_data, target_data[by + ['flag_tmp']].drop_duplicates(),
        on=by, how='left'
    ).query('flag_tmp.isnull()', engine='python').drop(columns='flag_tmp').copy()

    return joined_data

columns = ['year', 'month', 'category', 'article_id', 'type_label']

articles = pd.DataFrame()
for data_name in ['train.csv', 'test.csv']:
    data = pd.read_csv(data_dir_path.joinpath(data_name))
    tmp = data.columns.tolist()
    if data_name == 'train.csv':
        data.columns = columns[:-1]
        data = pd.concat([
            data, pd.DataFrame([tmp], columns=columns[:-1])
        ], axis=0)
        data['type_label'] = None
    else:
        data.columns = columns
        data = pd.concat([
            data, pd.DataFrame([tmp], columns=columns)
        ], axis=0)
        
    articles = pd.concat([articles, data], axis=0)

articles = articles.assign(
    year=lambda x: x.year.astype(int),
    article_id=lambda x: x.article_id.map(lambda y: re.sub(r'[a-z\.]', '', str(y))).astype(int)
)

# if body_data_file_path.exists():
#     articles = anti_join(
#         articles,
#         pd.read_csv(body_data_file_path).assign(article_id=lambda x: x.article_id.astype(int)),
#         by='article_id'
#     )

waiting_time = 3              # スクレイピングの間隔
n_writing_data = 10000        # 取得する件数 
article_url = 'http://news.livedoor.com/article/detail/{}/'
body_data_file_path = data_dir_path.joinpath('body_data.csv')
summary_data_file_path = data_dir_path.joinpath('summary_data.csv')

target_articles = articles.sort_values(
    'year', ascending=False
).head(min(len(articles), n_writing_data))

def read_url_to_soup(url):
    
    try:
        response = urllib.request.urlopen(url)
        html = response.read().decode(response.headers.get_content_charset(), errors='ignore')
        soup = BeautifulSoup(html, 'html.parser')
    except Exception:
        soup = None
    
    return soup

def write_data(data, file_path):
    if file_path.exists():
        data = pd.concat([data, pd.read_csv(file_path)]).assign(
            article_id=lambda x: x.article_id.astype(int)
        ).drop_duplicates()
    data.to_csv(file_path, index=False)

body_data = []
summary_data = []
i = 1
for article_id in tqdm(target_articles['article_id']):

    url = article_url.format(article_id)

    soup = read_url_to_soup(url)
    if soup is None or soup.find(class_='articleBody') is None or soup.find(class_='summaryList') is None:
        body_data.append((article_id, None, None))
        summary_data.append((article_id, None))
    else:

        title = soup.find(id='article-body').find('h1').text.strip()

        body = soup.find(class_='articleBody').find('span', {'itemprop': 'articleBody'}).text
        body = re.sub('\n+', '\n', body)
        body_data.append((article_id, title, body))

        summary_list = soup.find(class_='summaryList').find_all('li')
        summary_list = list(map(lambda x: x.text.strip(), summary_list))

        summary_data.extend([(article_id, summary) for summary in summary_list])
    
    if i % 50 == 0:        
        body_data = pd.DataFrame(body_data, columns=['article_id', 'title', 'text'])
        summary_data = pd.DataFrame(summary_data, columns=['article_id', 'text'])
        write_data(data=body_data, file_path=body_data_file_path)
        write_data(data=summary_data, file_path=summary_data_file_path)
        body_data = []
        summary_data = []        

    i += 1
    time.sleep(waiting_time)

if len(body_data) > 0:
    body_data = pd.DataFrame(body_data, columns=['article_id', 'title', 'text'])
    summary_data = pd.DataFrame(summary_data, columns=['article_id', 'text'])
    write_data(data=body_data, file_path=body_data_file_path)
    write_data(data=summary_data, file_path=summary_data_file_path)
