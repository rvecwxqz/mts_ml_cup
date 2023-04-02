import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from classes.stacking import MTSCupStacking
from paths import *

TO_DROP = ['age', 'is_male', 'user_id']
SEED = 42


def get_train_data(df, target):
    X = df.drop(TO_DROP, axis=1, errors='ignore')
    y = df[target]
    return X, y


def get_pipe(X, clf):
    cat_cols = X.select_dtypes(include='object').columns.to_list()
    num_cols = X.select_dtypes(include=['int', 'float64']).columns.to_list()

    num_pipe = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='constant', fill_value='None')),
        ('encoder', ce.CatBoostEncoder())
    ])

    col_trans = ColumnTransformer(transformers=[
        ('num_pipe', num_pipe, num_cols),
        ('cat_pipe', cat_pipe, cat_cols)
    ], remainder='passthrough', n_jobs=-1)

    clf = Pipeline(steps=[
        ('col_trans', col_trans),
        ('clf', clf)
    ])

    return clf


df_test_age = pd.read_csv(f'{DATASETS_PATH}/df_test_age.csv')
df_train_age = pd.read_csv(f'{DATASETS_PATH}/df_train_age.csv')
df_test_male = pd.read_csv(f'{DATASETS_PATH}/df_test_male.csv')
df_train_male = pd.read_csv(f'{DATASETS_PATH}/df_train_male.csv')

# age
target = 'age'
df_train_age = df_train_age.reset_index(drop=True)
df_test_age = df_test_age.reset_index(drop=True).drop(TO_DROP, axis=1, errors='ignore')
X, y = get_train_data(df_train_age, 'age')
y = y - 1
clf_age_params = {
    'max_depth': 6,
    'learning_rate': 0.08173539262501082,
    'n_estimators': 8644,
    'max_bin': 216,
    'min_data_in_leaf': 56,
    'l2_leaf_reg': 3.3576598254341588,
    'random_strength': 1.4,
    'task_type': 'GPU',
    'loss_function': 'MultiClass',
    'eval_metric': 'TotalF1',
    'random_seed': 42
}
cat_age = CatBoostClassifier(**clf_age_params)
xgb_params = {
    'tree_method': 'gpu_hist',
    'eta': 0.5405328189397176,
    'gamma': 0.4073363098708551,
    'max_depth': 2,
    'learning_rate': 0.20913559851683008,
    'n_estimators': 4091,
    'eval_metric': 'mlogloss',
    'lambda': 0.03505073778605376,
    'alpha': 0.017113713178727955,
    'colsample_bytree': 0.4,
    'subsample': 1.0,
    'random_state': 42,
    'min_child_weight': 134
}
xgb_age = get_pipe(X, XGBClassifier(**xgb_params))
lgbm_age_params = {
    'boosting_type': 'gbdt',
    'data_sample_strategy': 'bagging',
    'n_estimators': 4377,
    'learning_rate': 0.04281921143791494,
    'num_leaves': 1680,
    'max_depth': 6,
    'min_data_in_leaf': 5120,
    'lambda_l1': 20,
    'lambda_l2': 60,
    'min_gain_to_split': 0.8633486374801944,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'feature_fraction': 0.6000000000000001,
    'objective': 'MultiClass',
    'verbosity': -1,
    'random_seed': 42
}
lgbm_age = get_pipe(X, LGBMClassifier(**lgbm_age_params))
estimators = [
    ('cat', cat_age),
    ('cat_base', CatBoostClassifier(n_estimators=10000, task_type='GPU', random_state=42)),
    ('xgb', xgb_age),
    ('lgbm', lgbm_age)
]
final_estimator = CatBoostClassifier(verbose=False, n_estimators=10000, task_type='GPU', random_state=42)
stacking = MTSCupStacking(estimators, final_estimator, X, y, df_test_age, use_all_features=True)
stacking.get_and_save_methafeatures()
preds_age = stacking.predict_test()
df_test_age = pd.read_csv(f'{DATASETS_PATH}/df_test_age.csv')
preds_age = preds_age.merge(df_test_age[['user_id']], left_index=True, right_index=True)

# is_male
df_train_male = df_train_male.reset_index(drop=True)
df_test_male = df_test_male.reset_index(drop=True).drop(TO_DROP, axis=1, errors='ignore')
X, y = get_train_data(df_train_male, 'is_male')
clf_male_params = {
    'max_depth': 7,
     'learning_rate': 0.043111183398493846,
     'n_estimators': 11696,
     'max_bin': 48,
     'min_data_in_leaf': 179,
     'l2_leaf_reg': 38.01390680054509,
     'random_strength': 1.2,
     'task_type': 'GPU',
     'loss_function': 'Logloss',
     'eval_metric': 'Logloss',
     'random_seed': 42,
     'bootstrap_type': 'No'
}
cat_male_1 = CatBoostClassifier(**clf_male_params)
cat_male_2 = {
    'max_depth': 9,
    'learning_rate': 0.0639010402515796,
    'n_estimators': 5254,
    'max_bin': 233,
    'min_data_in_leaf': 254,
    'l2_leaf_reg': 54.34400864985853,
    'random_strength': 1.2,
    'task_type': 'GPU',
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'random_seed': 42,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.5221247671325702
}
cat_male_2 = CatBoostClassifier(**cat_male_2)
cat_male_3 = CatBoostClassifier(n_estimators=10000, task_type='GPU')
xgb_params_male_1 = {
    'tree_method': 'gpu_hist',
    'eta': 0.2957205625097929,
    'gamma': 1.1331901198132077,
    'max_depth': 4,
    'learning_rate': 0.014918106363927933,
    'n_estimators': 11954,
    'eval_metric': 'logloss',
    'lambda': 0.46693068535079346,
    'alpha': 4.591740187108138,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'random_state': 42,
    'min_child_weight': 28
}
xgb_male_1 = get_pipe(X, XGBClassifier(**xgb_params_male_1))
lgbm_male_1 = {
    'boosting_type': 'gbdt',
    'data_sample_strategy': 'goss',
    'n_estimators': 3746,
    'learning_rate': 0.2265052196968235,
    'num_leaves': 100,
    'max_depth': 2,
    'min_data_in_leaf': 3320,
    'lambda_l1': 20,
    'lambda_l2': 75,
    'min_gain_to_split': 0.45941943594568435,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 0.30000000000000004,
    'objective': 'binary',
    'random_seed': 42
}
lgbm_male_1 = get_pipe(X, LGBMClassifier(**lgbm_male_1))
lgbm_male_2 =  {
    'boosting_type': 'gbdt',
    'data_sample_strategy': 'goss',
    'n_estimators': 6805,
    'learning_rate': 0.20689833233525487,
    'num_leaves': 280,
    'max_depth': 0,
    'min_data_in_leaf': 3020,
    'lambda_l1': 25,
    'lambda_l2': 90,
    'min_gain_to_split': 1.7458071204228953,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 0.2,
    'objective': 'binary',
    'random_seed': 42
}
lgbm_male_2 = get_pipe(X, LGBMClassifier(**lgbm_male_2))
male_estimators = [
    ('cat_male_1', cat_male_1),
    ('cat_male_2', cat_male_2),
    ('cat_male_3', cat_male_3),
    ('xgb_male_1', xgb_male_1),
    ('lgbm_male_1', lgbm_male_1),
    ('lgbm_male_2', lgbm_male_2)
]
final_estimator = CatBoostClassifier(verbose=False, n_estimators=10000, task_type='GPU', random_state=42)
stacking = MTSCupStacking(male_estimators, final_estimator, X, y, df_test_male, use_all_features=True)
stacking.target = 'is_male'
stacking.get_and_save_methafeatures()
stacking.fit()
preds = stacking.predict_proba_test()
preds = preds[[1]]
preds.columns = ['is_male']
df_test_male = pd.read_csv(f'{DATASETS_PATH}/df_test_male.csv')
preds = preds.merge(df_test_male[['user_id']], left_index=True, right_index=True)
preds = preds.merge(preds_age, on='user_id')