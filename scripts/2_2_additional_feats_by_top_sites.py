import pandas as pd
from catboost import CatBoostClassifier
from paths import *
TO_DROP = ['user_id', 'age', 'is_male']


def get_top_sites_by_fi(target):
    df_train = pd.read_csv(f'{DATASETS_PATH}/df_train.csv')
    clf = CatBoostClassifier(task_type='GPU')
    X = df_train.drop(TO_DROP, axis=1)
    y = df_train[target]
    cat_features = X.select_dtypes(include='object').columns.to_list()
    clf.fit(X, y, cat_features=cat_features, verbose=False)
    fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    tf = fi.sort_values(ascending=False).index
    return [f for f in tf if '.' in f][:15]


# отбираем по 15 топ сайтов через feature_importances катбуста для каждого таргета
top_sites = get_top_sites_by_fi('age')
top_sites.extend(get_top_sites_by_fi('is_male'))
# убираем повторы, если они есть
top_sites = list(set(top_sites))
# считаем запросы по времени суток для топ сайтов
data_pd = pd.read_parquet(f'{LOCAL_DATA_PATH}/{DATA_FILE}')
grouped = data_pd[data_pd['url_host'].isin(top_sites)].groupby(['user_id', 'url_host', 'part_of_day'])[['request_cnt']]\
    .sum()
grouped.reset_index()
top_sites_pod = pd.pivot_table(data=grouped, index='user_id', columns=['url_host', 'part_of_day'], values='request_cnt')
top_sites_pod.columns = [f'{col[1]}_{col[0]}' for col in top_sites_pod.columns]
top_sites_pod = top_sites_pod.reset_index()
# добавим новые фичи
df_train = pd.read_csv(f'{DATASETS_PATH}/df_train.csv')
df_test = pd.read_csv(f'{DATASETS_PATH}/df_test.csv')
df_train = df_train.merge(top_sites_pod, on='user_id', how='left')
df_test = df_test.merge(top_sites_pod, on='user_id', how='left')
df_train.to_csv(f'{DATASETS_PATH}/df_train.csv')
df_test.to_csv(f'{DATASETS_PATH}/df_test.csv')
