import pandas as pd
from catboost import CatBoostClassifier, Pool
from paths import *
TO_DROP = ['user_id', 'age', 'is_male']


def get_tfs_by_target(df, target):
    X = df.drop(TO_DROP, axis=1)
    y = df[target]
    cat_features = X.select_dtypes(include='object').columns.to_list()
    clf = CatBoostClassifier(task_type='GPU')
    clf.fit(X, y, cat_features=cat_features)
    fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    tfs = fi[fi > 0].index
    return list(tfs)


df_train = pd.read_csv(f'{DATASETS_PATH}/df_train.csv')
df_test = pd.read_csv(f'{DATASETS_PATH}/df_test.csv')
# age
tfs_age = get_tfs_by_target(df_train, 'age')
tfs_age.append('user_id')
df_test_age = df_test[tfs_age]
tfs_age.append('age')
df_train_age = df_train[tfs_age]
df_test_age.to_csv(f'{DATASETS_PATH}/df_test_age.csv')
df_train_age.to_csv(f'{DATASETS_PATH}/df_train_age.csv')
# male
tfs_male = get_tfs_by_target(df_train, 'is_male')
tfs_male.append('user_id')
df_test_male = df_test[tfs_male]
tfs_male.append('age')
df_train_male = df_train[tfs_male]
df_test_male.to_csv(f'{DATASETS_PATH}/df_test_male.csv')
df_train_male.to_csv(f'{DATASETS_PATH}/df_train_male.csv')