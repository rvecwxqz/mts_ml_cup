import sklearn.metrics as m


def f1_weighted(y_true, y_pred):
    return m.f1_score(y_true, y_pred, average='weighted')


def classif_report(y_true, y_pred):
    return m.classification_report(y_true, y_pred,
                                   target_names=['19-25', '26-35', '36-45', '46-55', '56-65', '66+'])


def gini(y_true, y_proba):
    return 2 * m.roc_auc_score(y_true, y_proba) - 1


def get_score(clf, target, X_test, y_test):
    if target == 'age':
        return f1_weighted(y_test, clf.predict(X_test))
    else:
        return gini(y_test, clf.predict_proba(X_test)[:, 1])


METRICS_INFO = {
    'age': 'f1 weighted по возрасту ',
    'is_male': 'GINI по полу '
}
