import pandas as pd
from paths import *


# группируем основной фрейм по юзерам и сайтам, оставляем только те сайты, где более 1000 посещений
data_pd = pd.read_parquet(f'{LOCAL_DATA_PATH}/{DATA_FILE}')
# reqs by urls top sites
grouped = data_pd.groupby(['user_id', 'url_host']).agg({
    'request_cnt': sum
})
grouped = grouped.reset_index()
top_sites = grouped.groupby('url_host')[['request_cnt']].sum()\
    .sort_values(by='request_cnt', ascending=False).query('request_cnt > 1000').index
grouped = grouped[grouped['url_host'].isin(top_sites)]
pt_table = pd.pivot_table(data=grouped, index='user_id', columns='url_host', values='request_cnt', aggfunc=sum)
pt_table.columns = [str(i) for i in pt_table.columns]
pt_table.to_parquet(f'{LOCAL_DATA_PATH}/user_site_pivot.parquet')
