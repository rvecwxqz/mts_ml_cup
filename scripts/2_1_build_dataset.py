import os
import pandas as pd
import functools as ft
from paths import *
SEED = 42


df_train = pd.read_csv(f'{DATASETS_PATH}/df_train.csv')
df_test = pd.read_csv(f'{DATASETS_PATH}/df_test.csv')


def merge_dfs(df):
    global df_train
    global df_test
    df_train = df_train.merge(df, on='user_id', how='left')
    df_test = df_test.merge(df, on='user_id', how='left')

dfs_to_merge = []
# подгружаем матрицу user-url
usr_url_table = pd.read_parquet(f'{LOCAL_DATA_PATH}/user_site_pivot.parquet')
usr_url_table = usr_url_table.fillna(0)
dfs_to_merge.append(usr_url_table)
# clust features
data_pd = pd.read_parquet(f'{LOCAL_DATA_PATH}/{DATA_FILE}')
lables = pd.read_csv(f'{CLUST_PATH}labels.csv')
lables = lables[['url_host', 'labels2']]
data_pd = data_pd[['url_host', 'request_cnt', 'user_id']]
labels = data_pd.merge(lables, how='left', on='url_host')
pt_table = pd.pivot_table(data=labels, index='user_id', columns='labels2', values='request_cnt', aggfunc='sum')
dfs_to_merge.append(pt_table)
# добавляем фичи male fraction, holydays, population, age distribution
path = f'{FRACTION_FEAT_PATH}'
dfs = []
for feat_file in os.listdir(path):
    feat = pd.read_csv(path + feat_file)
    if 'user_id' in feat.columns:
        dfs.append(feat)
male_fraction_feats = ft.reduce(lambda left, right: pd.merge(left, right, on='user_id'), dfs)
male_fraction_feats['federal_district'].fillna('None', inplace=True)
male_fraction_feats = male_fraction_feats.drop_duplicates(subset='user_id')
dfs_to_merge.append(male_fraction_feats)
# pseudo target features
dfs = [pd.read_csv(feat_file) for feat_file in os.listdir(PSEUDO_FEATS_PATH)]
pseudo_target_feats = ft.reduce(lambda left, right: pd.merge(left, right, on='user_id'), dfs)
dfs_to_merge.append(pseudo_target_feats)
# добавляем все фичи к датафрейму
[merge_dfs(df) for df in dfs_to_merge]
df_train.to_csv(f'{DATASETS_PATH}/df_train.csv')
df_test.to_csv(f'{DATASETS_PATH}/df_test.csv')
