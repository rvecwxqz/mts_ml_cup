from pyts.transformation import ROCKET
from tsfresh import extract_features
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
from parallel_pandas import ParallelPandas
from tqdm import tqdm
from pandarallel import pandarallel
import bisect
from tsfresh import select_features
from paths import *


def read_pqt(file, columns):
    return pq.read_table(file).select(columns).to_pandas()


def get_target(columns):
    return read_pqt(TARGET_PATH, columns)


def get_data(columns):
    return read_pqt(DATA_PATH, columns)


# Медианы возрастов
data = get_data(columns=['user_id', 'url_host'])
data.drop_duplicates(inplace=True)
# Из датафрейма удалются сайты, у которых было менее 20 уникальных пользователей
sites = data.groupby('url_host', as_index=False).agg(unique_user_id=('user_id', 'nunique'))
drop_sites_20 = list(sites.loc[sites.unique_user_id<=20].url_host.unique())
data = data.loc[~data['url_host'].isin(drop_sites_20)]
del sites
# Подгружаются таргеты по полу
# Для каждого сайта считается число людей, число людей мужчин
# и доля от этих двух агрегатов
target = get_target(columns=['user_id','is_male'])
target['user_id'] = target['user_id'].astype('int32')
target['is_male'] = target['is_male'].astype('object')
data = data.merge(target[['is_male','user_id']], on = 'user_id', how = 'inner')
data = data.loc[~(data['is_male'].isna()) & (data['is_male'] != 'NA')]
data['is_male'] = data['is_male'].astype('int8')
df = data.groupby('url_host', as_index=False).agg({'user_id':'nunique','is_male':'sum'})
df['male_fraction'] = df['is_male'] / df['user_id']
df.loc[df.user_id!=0]
df.to_csv(f'{PSEUDO_FEATS_PATH}sites_male_fraction.csv', index=False)
del data, target
data = get_data(columns=['user_id', 'url_host'])
data.drop_duplicates(inplace=True)
data = data.loc[~data['url_host'].isin(drop_sites_20)]
target = get_target(columns=['user_id','age'])
target['user_id'] = target['user_id'].astype('int32')
# добавляем target, удаляем nan
data = data.merge(target[['age','user_id']], on = 'user_id', how = 'inner')
data = data.loc[~(data['age'].isna()) & (data['age'] > 18) & (data['age'] != 'NA')]
data['age'] = data['age'].astype('int16')
df = data.groupby('url_host', as_index=False).agg(median_age=('age', 'median'))
df.loc[~df.median_age.isna()]
df.to_csv(f'{PSEUDO_FEATS_PATH}sites_age.csv', index=False)
# Генерация признаков Rocketdata
data = get_data(columns=['user_id', 'url_host', 'date', 'part_of_day'])
# data.drop_duplicates(inplace=True)
sites_pop = data.groupby('url_host', as_index=False)[['user_id']].nunique()
sites_pop.columns = ['url_host', 'user_id_count']
unique_users = list(data.user_id.unique())
data.part_of_day = data.part_of_day.replace({'night': 0, 'morning': 1, 'day': 2, 'evening': 3})
data.date = pd.to_datetime(data.date)
data = data.merge(sites_pop, how='left')
male_fraction = pd.read_csv('sites_male_fraction.csv').fillna(0)
male_fraction = male_fraction.loc[male_fraction.male_fraction != 0]
male_fraction['male_fraction'] = male_fraction['male_fraction'] - 0.5
male_fraction = male_fraction[['url_host', 'male_fraction']]

median_age = pd.read_csv('sites_age.csv').fillna(0)
median_age = median_age.loc[median_age.median_age != 0]
median_age['median_age'] = median_age['median_age'] - 18
median_age = median_age[['url_host', 'median_age']]

target = median_age.merge(male_fraction, how='left')
target = target.loc[~target.male_fraction.isna()]
target['target'] = target['median_age'] * target['male_fraction']
target = target[['url_host', 'target']]

data = data.sort_values(by=['date', 'part_of_day', 'user_id_count'])
data = data.merge(target, how='left')
data['target'] = data['target'].fillna(0)
# data = data[['user_id', 'target']]
df_rows = pd.DataFrame(columns=list(range(500)) + ['user_id'])

groups = data.groupby("user_id")

for user_id, group in tqdm(groups):
    group = group[-500:].reset_index()
    row = group[['target']].T
    row['user_id'] = user_id
    df_rows = df_rows.append(row)

df_rows = df_rows.fillna(0)
rocket = ROCKET(n_kernels=10, random_state=42)
rocket.fit(df_rows[list(range(500))])
rocket_features = rocket.transform(df_rows[list(range(500))])
rocket_features = pd.concat([df_rows[['user_id']].reset_index(), pd.DataFrame(rocket_features)], axis=1)
rocket_features.drop(columns=['index'], axis=1, inplace=True)
rocket_features
rocket_features.to_csv(f'{PSEUDO_FEATS_PATH}rocket_features.csv', index=False)
# Генерация признаков по придуманному таргету
data = get_data(columns=['user_id', 'url_host'])
data.drop_duplicates(inplace=True)
male_fraction = pd.read_csv('sites_male_fraction.csv').fillna(0)
male_fraction = male_fraction.loc[male_fraction.male_fraction != 0]
male_fraction['male_fraction'] = male_fraction['male_fraction'] - 0.5
male_fraction = male_fraction[['url_host', 'male_fraction']]
median_age = pd.read_csv('sites_age.csv').fillna(0)
median_age = median_age.loc[median_age.median_age != 0]
median_age['median_age'] = median_age['median_age'] - 18
median_age = median_age[['url_host', 'median_age']]
target = median_age.merge(male_fraction, how='left')
target = target.loc[~target.male_fraction.isna()]
target['target'] = target['median_age'] * target['male_fraction']
target = target[['url_host', 'target']]
data = data.merge(target, how='left')
data['target'] = data['target'].fillna(0)
# data = data[['user_id', 'target']]
def q10(x): return x.quantile(0.1)
def q25(x): return x.quantile(0.25)
def q75(x): return x.quantile(0.75)
def q90(x): return x.quantile(0.9)
def g1(x): return x[(x>=-10) & (x<-9)].count()
def g2(x): return x[(x>=-9) & (x<-8)].count()
def g3(x): return x[(x>=-8) & (x<-7)].count()
def g4(x): return x[(x>=-7) & (x<-6)].count()
def g5(x): return x[(x>=-6) & (x<-5)].count()
def g6(x): return x[(x>=-5) & (x<-4)].count()
def g7(x): return x[(x>=-4) & (x<-3)].count()
def g8(x): return x[(x>=-3) & (x<-2.5)].count()
def g9(x): return x[(x>=-2.5) & (x<-2)].count()
def g10(x): return x[(x>=-2) & (x<-1.5)].count()
def g11(x): return x[(x>=-1.5) & (x<-1)].count()
def g12(x): return x[(x>=-1) & (x<-0.5)].count()
def g13(x): return x[(x>=-0.5) & (x<-0.25)].count()
def g14(x): return x[(x>=-0.25) & (x<0.0)].count()
def g15(x): return x[(x>=0.0) & (x<0.25)].count()
def g16(x): return x[(x>=0.25) & (x<0.5)].count()
def g17(x): return x[(x>=0.5) & (x<1)].count()
def g18(x): return x[(x>=1) & (x<1.5)].count()
def g19(x): return x[(x>=1.5) & (x<2)].count()
def g20(x): return x[(x>=2) & (x<2.5)].count()
def g21(x): return x[(x>=2.5) & (x<3)].count()
def g22(x): return x[(x>=3) & (x<4)].count()
def g23(x): return x[(x>=4) & (x<5)].count()
def g24(x): return x[(x>=5) & (x<6)].count()
def g25(x): return x[(x>=6) & (x<7)].count()
def g26(x): return x[(x>=7) & (x<8)].count()
def g27(x): return x[(x>=8) & (x<9)].count()
def g28(x): return x[(x>=9) & (x<10)].count()

df = data.groupby('user_id', as_index=False).agg(t_max=('target', 'max'),
                                                 t_min=('target', 'min'),
                                                 t_avg=('target', 'mean'),
                                                 t_med=('target', 'median'),
                                                 t_q10=('target', q10),
                                                 t_q25=('target', q25),
                                                 t_q75=('target', q75),
                                                 t_q90=('target', q90),
                                                 t_std=('target', 'std'),
                                                 g1=('target', g1),
                                                 g2=('target', g2),
                                                 g3=('target', g3),
                                                 g4=('target', g4),
                                                 g5=('target', g5),
                                                 g6=('target', g6),
                                                 g7=('target', g7),
                                                 g8=('target', g8),
                                                 g9=('target', g9),
                                                 g10=('target', g10),
                                                 g11=('target', g11),
                                                 g12=('target', g12),
                                                 g13=('target', g13),
                                                 g14=('target', g14),
                                                 g15=('target', g15),
                                                 g16=('target', g16),
                                                 g17=('target', g17),
                                                 g18=('target', g18),
                                                 g19=('target', g19),
                                                 g20=('target', g20),
                                                 g21=('target', g21),
                                                 g22=('target', g22),
                                                 g23=('target', g23),
                                                 g24=('target', g24),
                                                 g25=('target', g25),
                                                 g26=('target', g26),
                                                 g27=('target', g27),
                                                 g28=('target', g28))

df['t_scope'] = (df['t_max'] + 100) - (df['t_min'] + 100)
df.to_csv(f'{PSEUDO_FEATS_PATH}pseudo_target_features.csv', index=False)
# Генерация признаков TSFRESH
data = get_data(columns=['user_id', 'url_host', 'date', 'part_of_day'])
sites_pop = data.groupby('url_host', as_index=False)[['user_id']].nunique()
sites_pop.columns = ['url_host', 'user_id_count']
unique_users = list(data.user_id.unique())
data.part_of_day = data.part_of_day.replace({'night': 0, 'morning': 1, 'day': 2, 'evening': 3})
data.date = pd.to_datetime(data.date)
data = data.merge(sites_pop, how='left')
male_fraction = pd.read_csv('sites_male_fraction.csv').fillna(0)
male_fraction = male_fraction.loc[male_fraction.male_fraction != 0]
male_fraction['male_fraction'] = male_fraction['male_fraction'] - 0.5
male_fraction = male_fraction[['url_host', 'male_fraction']]
median_age = pd.read_csv('sites_age.csv').fillna(0)
median_age = median_age.loc[median_age.median_age != 0]
median_age['median_age'] = median_age['median_age'] - 18
median_age = median_age[['url_host', 'median_age']]
target = median_age.merge(male_fraction, how='left')
target = target.loc[~target.male_fraction.isna()]
target['target'] = target['median_age'] * target['male_fraction']
target = target[['url_host', 'target']]
data = data.sort_values(by=['date', 'part_of_day', 'user_id_count'])
data = data.merge(target, how='left')
data['target'] = data['target'].fillna(0)
data.reset_index(inplace=True)
data['time'] = data.index
tqdm.pandas()
groups = data[['user_id', 'target', 'time']].groupby("user_id")
data_short = groups.parallel_apply(lambda x: x[-500:].reset_index(drop=True)).reset_index(drop=True)
del groups, data, median_age
features = extract_features(data_short, column_id="user_id", column_sort="time")
features['user_id'] = features.index
tmp = features.copy()

def age_bucket(x):
    return bisect.bisect_left([18,25,35,45,55,65], x)

target = get_target(columns=['user_id','age'])
target['user_id'] = target['user_id'].astype('int32')

# добавляем target, удаляем nan
tmp = tmp.merge(target[['age','user_id']], on = 'user_id', how = 'inner')
tmp = tmp.loc[~(tmp['age'].isna()) & (tmp['age'] > 18) & (tmp['age'] != 'NA')]
tmp['age'] = tmp['age'].astype('int16')
tmp['age'] = tmp['age'].map(age_bucket)
tmp = tmp.fillna(0)
filtered_features = select_features(tmp[[el for el in tmp.columns if el not in ['age', 'user_id']]], tmp['age'])
features[['user_id'] + list(filtered_features.columns)]
features[['user_id'] + list(filtered_features.columns)].to_csv(f'{PSEUDO_FEATS_PATH}tsfresh_features.csv', index=False)
