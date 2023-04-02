import pandas as pd
import numpy as np
import bisect
import pyarrow as pa
import pyarrow.parquet as pq
from paths import *

SEED = 42


def age_bucket(x):
    return bisect.bisect_left([18, 25, 35, 45, 55, 65], x)


id_to_submit = pq.read_table(f'{LOCAL_DATA_PATH}/{SUBMISSION_FILE}').to_pandas()
data = pq.read_table(f'{LOCAL_DATA_PATH}/{DATA_FILE}')
usr_targets = pq.read_table(f'{LOCAL_DATA_PATH}/{TARGET_FILE}').to_pandas()
# получаем индексы
train_index = usr_targets['user_id']
test_index = id_to_submit['user_id']
# создаем датафрейм
df = pd.DataFrame(data['user_id'].to_pandas().drop_duplicates())

# число запросов по времени суток
agg_by_part_of_day = data.select(['user_id', 'part_of_day', 'request_cnt']). \
    group_by(['user_id', 'part_of_day']).aggregate([('request_cnt', "sum")])
data_count_by_part = pd.pivot_table(agg_by_part_of_day.to_pandas(), index='user_id', columns='part_of_day',
                                    values='request_cnt_sum', fill_value=0)  # aggfunc='sum')
data_count_by_part = data_count_by_part.apply(lambda x: x / data_count_by_part.sum(axis=1))
df = df.merge(data_count_by_part, how='left', left_on='user_id', right_index=True)

# количество дней в данных max-min
days_with_us = data.select(['user_id', 'date']). \
    group_by(['user_id']).aggregate([
    ('date', "min"),
    ('date', 'max')
])
days_with_us = days_with_us.to_pandas()
days_with_us['days_with_us'] = (days_with_us['date_max'] - days_with_us['date_min']).dt.days + 1
days_with_us.drop(['date_min', 'date_max'], axis=1, inplace=True)
days_with_us.loc[days_with_us['days_with_us'] == 0, 'days_with_us'] = 1
df = df.merge(days_with_us, how='left', on='user_id')

# количество активных дней
active_days = data.select(['user_id', 'date']). \
    group_by(['user_id']).aggregate([
    ('date', "count_distinct")
]).to_pandas()
active_days.columns = ['active_days', 'user_id']
df = df.merge(active_days, how='left', on='user_id')

# Отношение активных дней к количеству дней всего
df['active_to_all_ratio'] = df['active_days'] / df['days_with_us']
df.loc[
    (df['days_with_us'] < 5) & (df['days_with_us'] == df['active_days']), 'active_to_all_ratio'
] = -1

# Цена устройства
price = data.select(['user_id', 'price']). \
    group_by(['user_id']).aggregate([
    ('price', "mean")
]).to_pandas()
df = df.merge(price, how='left', on='user_id')
df['price_mean'].fillna(0, inplace=True)

# Количество уникальных регионов, городов, url_hos
regions_count = data.select(['user_id', 'region_name', 'city_name', 'url_host']). \
    group_by(['user_id']).aggregate([
    ('region_name', "count_distinct"),
    ('city_name', "count_distinct"),
    ('url_host', "count_distinct")
]).to_pandas()
df = df.merge(regions_count, how='left', on='user_id')

# Количество запросов за активный день в среднем
df['req_per_active_day'] = (df['morning'] + df['day'] + df['evening'] + df['night']) / df['active_days']

# наличие дорогого телефона
prices = data.select(['user_id', 'price']) \
    .group_by(['user_id']).aggregate([('price', 'mean')])
ids_expansive_phones = prices.filter(pa.compute.greater(prices['price_mean'], 57000)) \
    .select(['user_id']).to_pandas()
del prices
df.loc[:, 'have_expansive_phone'] = df['user_id'].apply(lambda x: x in ids_expansive_phones['user_id'])
ids_low_iphones = data.filter \
    (pa.compute.not_equal(data['cpe_model_os_type'], 'Android')) \
    .select(['user_id', 'price']).to_pandas().query('price < 35000')['user_id'].drop_duplicates()
df.loc[:, 'have_low_price_iphone'] = df['user_id'].apply(lambda x: x in ids_low_iphones.to_numpy())
# part of day мода
df = df.merge(data.select(['user_id', 'part_of_day']).to_pandas()
              .groupby('user_id').agg(lambda x: pd.Series.mode(x)[0]).reset_index(), on='user_id', how='left')

# mean/min/max/std запросов
data_0 = data.select(['user_id', 'date', 'request_cnt']). \
    group_by(['user_id', 'date']).aggregate([
    ('request_cnt', "sum")
])
data_4 = data_0.select(['user_id', 'request_cnt_sum']). \
    group_by(['user_id']).aggregate([
    ('request_cnt_sum', "mean"),
    ('request_cnt_sum', "max"),
    ('request_cnt_sum', "min"),
    ('request_cnt_sum', "stddev")
]).to_pandas()
df = df.merge(data_4, how='left', on='user_id')
# телефон, город, регион
df = df.merge(data.select(['user_id', 'cpe_model_name', 'city_name',
                           'region_name', 'cpe_type_cd', 'cpe_manufacturer_name', 'cpe_model_os_type']).to_pandas()
              .groupby('user_id').agg(lambda x: pd.Series.mode(x)[0]).reset_index(), how='left', on='user_id')

# разобьем выборки на трейн и тест
df_train = df.merge(usr_targets, how='right')
df_test = df[df['user_id'].isin(test_index)]
# убираем пустые значения из трейна
df_train = df_train[df_train['age'].notna()]
df_train = df_train[df_train['is_male'].notna()]
# добавим бакет по возрасту
df_train.loc[:, 'age'] = df_train['age'].map(age_bucket)
# уберем нулевой бакет из трейна
df_train.loc[df_train['age'] == 0, 'age'] = 1
# уберем пропуски из is_male
df_train = df_train[df_train['is_male'] != 'NA']
df_train['is_male'] = df_train['is_male'].map(int)

df_train.to_csv(f'{DATASETS_PATH}/df_train.csv')
df_test.to_csv(f'{DATASETS_PATH}/df_test.csv')
