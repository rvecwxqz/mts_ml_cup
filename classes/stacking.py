import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool
from datetime import datetime
from sklearn.model_selection import train_test_split
from classes.metrics import *
from classes.utils import *
from scripts.paths import *

SEED = 42


class MTSCupStacking:

    def __init__(self, estimators, final_estimator, X, y, X_test, use_all_features=False, seed=SEED,
                 err=None):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self._use_all_features = use_all_features
        self.X = X
        self.y = y
        self.X_test = X_test
        self.cv = 5
        self.err = err
        self.seed = SEED
        self.n = len(estimators)
        self.scores = {}
        self.times = {}
        self.target = 'age'
        self.X_meta = None
        self.X_test_meta = None

    def get_path(self, name):
        stacking_path = f'{LOCAL_DATA_PATH}stacking/'
        if not os.path.exists(stacking_path):
            os.mkdir(stacking_path)
        path = f'{stacking_path}{name}/'
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_and_save_methafeatures(self):
        X = self.X
        y = self.y
        X_test = self.X_test
        for name, clf in self.estimators:
            print(f'Model {name}')
            self.scores[name] = []
            self.times[name] = []
            X_meta_parts = []
            X_meta_test_list = []
            X_meta_test = pd.DataFrame(index=X_test.index)
            for i, (train_index, val_index) in enumerate(StratifiedKFold \
                                                                     (n_splits=self.cv, shuffle=True,
                                                                      random_state=self.seed).split(X, y)):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                start_time = datetime.now()
                # если катбуст - используем внутренние методы для категориальных
                if 'cat' in name:
                    cat_features = X_train.select_dtypes(include='object').columns.to_list()
                    pool = Pool(X_train[1:], y_train[1:], cat_features=cat_features)
                    clf.fit(pool, verbose=False)
                else:
                    clf.fit(X_train, y_train)
                preds = clf.predict_proba(X_val)
                # Добавляем в список из метапризнаков для трейна
                X_meta_parts.append(pd.DataFrame(index=X_val.index, data=preds))
                # Добавляем в список метапризнаки теста, потом усредним
                preds_test = clf.predict_proba(X_test)
                X_meta_test_list.append(pd.DataFrame(index=X_test.index, data=preds_test))
                # Предиктим часть тренировочной выборки
                score = get_score(clf, self.target, X_val, y_val)
                # Сохраняем скор
                self.scores[name].append(score)
                # Сохраняем модель
                path = f'{self.get_path(name)}models/'
                if not os.path.exists(path):
                    os.mkdir(path)
                save(clf, f'{path}/{name}_{i}.pkl', verbose=False)
                # Сохраняем затраченное время
                n_fold_time = datetime.now() - start_time
                self.times[name].append(n_fold_time)
                print(f'Fold {i}: {score}, estimated time {n_fold_time}')
            # объединяем метапризнаки трейна
            X_meta_train = pd.concat(X_meta_parts)
            # усредняем метапризнаки теста
            X_meta_test = sum(X_meta_test_list) / len(X_meta_test_list)
            path = self.get_path(name)
            X_meta_train.columns = [f'{name}_{col}' for col in X_meta_train.columns]
            X_meta_test.columns = [f'{name}_{col}' for col in X_meta_test.columns]
            X_meta_train.to_csv(f'{path}{self.target}_{name}_train.csv')
            X_meta_test.to_csv(f'{path}{self.target}_{name}_test.csv')
            # проверим размерность
            if X_meta_train.shape[0] != X.shape[0] or X_meta_test.shape[0] != X_test.shape[0]:
                print("Ошибка.")
            print(f'{name} estimated time {np.sum(self.times[name])}')

    def fit(self):
        X_meta_list = []
        X_test_meta_list = []
        for name, clf in self.estimators:
            path = self.get_path(name)
            X_meta_list.append(pd.read_csv(f'{path}{self.target}_{name}_train.csv', index_col='Unnamed: 0'))
            X_test_meta_list.append(pd.read_csv(f'{path}{self.target}_{name}_test.csv', index_col='Unnamed: 0'))
        self.X_meta = pd.concat(X_meta_list, axis=1)
        self.X_test_meta = pd.concat(X_test_meta_list, axis=1)
        # Сортируем данные по индексам
        self.X_meta = self.X_meta.sort_index()
        self.y = self.y.sort_index()
        if self.err is not None:
            # добавляем нормальный шум
            self.X_meta += self.err * np.random.randn(*self.X_meta.shape)
        if not self._use_all_features:
            self.final_estimator.fit(self.X_meta, self.y)
        else:
            self.X = self.X.sort_index()
            X_full = pd.concat([self.X_meta, self.X], axis=1)
            # кривая проверка на катбуст
            if str(type(self.final_estimator)) == "<class 'catboost.core.CatBoostClassifier'>":
                cat_features = X_full.select_dtypes(include='object').columns.to_list()
                pool = Pool(X_full, self.y, cat_features=cat_features)
                self.final_estimator.fit(pool)
            else:
                self.final_estimator.fit(X_full, self.y)
        return self

    def get_val_score_by_df(self, X, y):
        preds_list = []
        for name, clf in self.estimators:
            preds_model_list = []
            for i in range(self.cv):
                path = f'{self.get_path(name)}models/'
                clf = load(f'{path}/{name}_{i}.pkl')
                preds_df = pd.DataFrame(index=X.index, data=clf.predict_proba(X))
                preds_df.columns = [f'{name}_{i}' for i in preds_df.columns]
                preds_model_list.append(preds_df)
            preds_list.append(sum(preds_model_list) / len(preds_model_list))
        X_meta = pd.concat(preds_list, axis=1)
        print(get_score(self.final_estimator, self.target, pd.concat([X_meta, X], axis=1), y))

    def predict_test(self):
        if not self._use_all_features:
            preds = pd.DataFrame(index=self.X_test_meta.index, data=self.final_estimator.predict(self.X_test_meta))
        else:
            X_test_full = pd.concat([self.X_test_meta, self.X_test], axis=1)
            preds = pd.DataFrame(index=X_test_full.index, data=self.final_estimator.predict(X_test_full))
        if self.target == 'age':
            preds.columns = ['age']
            preds['age'] += 1
        else:
            preds.columns = ['is_male']
        return preds

    def predict_proba_test(self):
        if not self._use_all_features:
            preds = pd.DataFrame(index=self.X_test_meta.index,
                                 data=self.final_estimator.predict_proba(self.X_test_meta))
        else:
            X_test_full = pd.concat([self.X_test_meta, self.X_test], axis=1)
            preds = pd.DataFrame(index=X_test_full.index, data=self.final_estimator.predict_proba(X_test_full))
        return preds

    def get_meta_features(self):
        return self.X_meta, self.y
