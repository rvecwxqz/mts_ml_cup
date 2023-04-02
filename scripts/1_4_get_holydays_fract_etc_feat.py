import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import requests
import json
from paths import *


data = pq.read_table(f'{LOCAL_DATA_PATH}/{DATA_FILE}')
data = data.select(['user_id', 'region_name', 'city_name']).to_pandas()
# Скачивание статы по городам
geo_stat_url = "https://raw.githubusercontent.com/hflabs/city/master/city.csv"
geo_stat = pd.read_csv(geo_stat_url)
# Выделяем уникальные названия регионов в датафрейме конкурса и в файле со статой
# Так как названия регионов отличаются между файлами, то переименовываю их на единый манер
# в файле со статой
geo_stat_regions = geo_stat.region.unique()
df_regions = data.region_name.unique()
# Вот тут само перименовывание
for geo in geo_stat_regions:
    true_name = list(filter(lambda x: geo in x, df_regions))
    if len(true_name) > 0:
        geo_stat.loc[geo_stat.region==geo, 'region'] = true_name[0]
geo_stat = geo_stat[['region', 'federal_district', 'city', 'population', 'capital_marker']]
geo_stat.rename(columns={'city': 'city_name', 'region': 'region_name'}, inplace=True)
geo_stat = geo_stat.drop_duplicates(subset=['city_name'], keep='first')
# Добавление статы по городам к основному df
data = data.merge(geo_stat[['federal_district', 'city_name', 'population', 'capital_marker']],
                  how='left', on='city_name')
# Добавление статы по регионам к основному df
geo_stat_gr = geo_stat.groupby('region_name', as_index=False)[['population']].sum()
geo_stat_gr.columns = ['region_name', 'region_population']
data = data.merge(geo_stat_gr, how='left', on='region_name')
# Подсчет агрегатов по числовым метрикам
data = pa.Table.from_pandas(data)
df_users_features = data.select(['user_id', 'population', 'capital_marker', 'region_population']).\
                    group_by(['user_id']).aggregate([('population', "mean"),
                                                     ('capital_marker', "mean"),
                                                     ('region_population', "mean")]).to_pandas()
# Подсчет агрегатов по имени федерального округа

fo = data.select(['user_id', 'federal_district']).\
                    group_by(['user_id', 'federal_district']).aggregate([('federal_district', "count")]).to_pandas()
fo['federal_district_count_max'] = fo.groupby(['user_id'], as_index=False)["federal_district_count"].transform('max')
fo = fo.loc[fo.federal_district_count==fo.federal_district_count_max]

df_users_features = df_users_features.merge(fo[['user_id', 'federal_district']], how='left')
# Сохраняем фрейм с фичами по популяционной стате по пользователям

df_users_features.to_csv(f'{FRACTION_FEAT_PATH}users_population_features.csv', index=False)
df_users_features.head()
data = pq.read_table(f'{LOCAL_DATA_PATH}/{DATA_FILE}')
data = data.select(['user_id', 'request_cnt', 'date']).to_pandas()
data['date'] = pd.to_datetime(data['date'])
# Не проверял этот календарь, но надеюсь, в нем все корректно)
# Тут даты во фрейме бинарно маркируются: выходной, не выходной
url21 = 'https://raw.githubusercontent.com/d10xa/holidays-calendar/master/json/consultant' + '2021' + '.json'
r21 = requests.get(url21)
cal21 = json.loads(r21.text)
url22 = 'https://raw.githubusercontent.com/d10xa/holidays-calendar/master/json/consultant' + '2022' + '.json'
r22 = requests.get(url22)
cal22 = json.loads(r22.text)


def is_holiday(date):
    if date.year == 2021:
        return cal21['holidays'].count(str(date)[0:-9])
    else:
        return cal22['holidays'].count(str(date)[0:-9])


data['date'] = data['date'].apply(lambda date: is_holiday(date))
data.head()
# Подсчет долей выходных долей по пользовтаелям
data['date'] = data['date'].astype('category')
df = data.groupby(['user_id','date'],group_keys=False)['request_cnt'].sum().reset_index()
df['request_cnt_sum'] = pd.Series(np.repeat(df.groupby('user_id')['request_cnt'].sum().reset_index(drop=True).values,2,axis=0))
df['holyday_fraction'] = df['request_cnt']/df['request_cnt_sum']
df['holyday_fraction'] = df['holyday_fraction'].astype('float32')
df = df.loc[df['date']==1].drop(['request_cnt_sum','request_cnt','date'],axis=1).reset_index(drop=True)
# Сохранение
df.to_csv(f'{FRACTION_FEAT_PATH}users_holyday_fraction.csv', index=False)
df.head()

# male sites fraction
data = pq.read_table(f'{LOCAL_DATA_PATH}/{DATA_FILE}')
data = data.select(['user_id', 'request_cnt', 'url_host']).to_pandas()
# Из датафрейма удалются сайты, у которых было менее 50 уникальных пользователей
sites = (data.groupby('url_host', as_index=False)
           .agg(request_cnt=('request_cnt', 'sum'), unique_user_id=('user_id', 'nunique'))
           .sort_values(by='request_cnt', ascending=False))
drop_sites_50 = list(sites.loc[sites.unique_user_id<=50].url_host.unique())
data = data.loc[~data['url_host'].isin(drop_sites_50)]
# Подгружаются таргеты по полу
# Для каждого сайта считается число запросов, число запросов мужчин
# и доля от этих двух агрегатов
target = pq.read_table(f'{LOCAL_DATA_PATH}/{TARGET_FILE}').to_pandas()[['user_id', 'is_male']]
target['user_id'] = target['user_id'].astype('int32')
target['is_male'] = target['is_male'].astype('object')
data = data.merge(target[['is_male','user_id']], on = 'user_id', how = 'inner')
data = data.loc[~(data['is_male'].isna()) & (data['is_male'] != 'NA')]
data['is_male'] = data['is_male'].astype('int8')
data['male_request_cnt'] = data['is_male'] * data['request_cnt']
df = data.groupby('url_host', as_index=False).agg({'request_cnt':'sum','male_request_cnt':'sum'})
df['male_fraction'] = df['male_request_cnt'] / df['request_cnt']
df.to_csv(f'{FRACTION_FEAT_PATH}sites_male_fraction.csv', index=False)
df.head()

df = pd.read_csv('./context_data/custom_data/sites_male_fraction.csv')
# Заново загружается основной фрейм, так как выше он редактировался
data = pq.read_table(f'{LOCAL_DATA_PATH}/{DATA_FILE}')
#data = data.select(['user_id', 'request_cnt', 'url_host']).to_pandas()
# К сгруппированному основному фрейму мержится датафрейм с male_fraction из ячеек выше
# Далее группировка по пользователям
#data = pa.Table.from_pandas(data)
data = data.group_by(['user_id', 'url_host']).aggregate([('request_cnt', "sum")]).to_pandas()
data = data.merge(df[['url_host', 'male_fraction']], how='left')
data['male_fraction_cum'] = data['male_fraction'] * data['request_cnt_sum']
users_male_fraction = data.groupby(['user_id'], as_index=False)[['male_fraction', 'male_fraction_cum']].mean()
users_male_fraction.to_csv(f'{FRACTION_FEAT_PATH}users_male_fraction.csv', index=False)
users_male_fraction.head()
# users_age_distribution
# Загрузка основного фрейма, удаление сайтов с менее 50 уников
# Примердживание таргета по возрасту
data = pq.read_table(f'{LOCAL_DATA_PATH}/{DATA_FILE}')\
    .select(['user_id', 'request_cnt', 'url_host']).to_pandas()
sites = (data.groupby('url_host', as_index=False)
           .agg(request_cnt=('request_cnt', 'sum'), unique_user_id=('user_id', 'nunique'))
           .sort_values(by='request_cnt', ascending=False))
drop_sites_50 = list(sites.loc[sites.unique_user_id<=50].url_host.unique())
data = data.loc[~data['url_host'].isin(drop_sites_50)]
target = pq.read_table(f'{LOCAL_DATA_PATH}/{TARGET_FILE}').to_pandas()[['user_id', 'age']]
target['user_id'] = target['user_id'].astype('int32')
# добавляем target, удаляем nan
data = data.merge(target[['age','user_id']], on = 'user_id', how = 'inner')
data = data.loc[~(data['age'].isna()) & (data['age'] > 18) & (data['age'] != 'NA')]
data['age'] = data['age'].astype('int16')
# Группировка по url и подсчет агрегатов по возрасту

def q10(x): return x.quantile(0.1)
def q25(x): return x.quantile(0.25)
def q75(x): return x.quantile(0.75)
def q90(x): return x.quantile(0.9)

data = data[['user_id', 'url_host', 'age']].drop_duplicates(subset=['user_id', 'url_host'], keep='first')
df = data.groupby('url_host', as_index=False).agg(median_age=('age', 'median'),
                                                  q10_age=('age', q10),
                                                  q25_age=('age', q25),
                                                  q75_age=('age', q75),
                                                  q90_age=('age', q90),
                                                  avg_age=('age', 'mean'))
df.to_csv('sites_age_distribution.csv', index=False)

# Снова подгрузка основного фрейма и джоин фрейма выше с агрегатами возрастов
# Группировка по пользователям
data = pq.read_table(f'{LOCAL_DATA_PATH}/{DATA_FILE}')\
    .select(['user_id', 'request_cnt', 'url_host'])
data = data.group_by(['user_id', 'url_host']).aggregate([('request_cnt', "sum")]).to_pandas()
data = data.merge(df[['url_host', 'median_age', 'q10_age', 'q25_age', 'q75_age', 'q90_age', 'avg_age']],
                  how='left')
users_age_distribution = data.groupby(['user_id'], as_index=False)[['median_age', 'q10_age', 'q25_age',
                                                                    'q75_age', 'q90_age', 'avg_age']].mean()
users_age_distribution.to_csv(f'{FRACTION_FEAT_PATH}users_age_distribution.csv', index=False)
