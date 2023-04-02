import pandas as pd
import numpy as np
import torch
import random
import razdel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm
import random
from functools import partial
import matplotlib.pyplot as plt
import cuml.cluster as hdbscan
import umap
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm.notebook import trange
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials


# pip install transformers sentencepiece datasets razdel -q
# pip install hdbscan
# pip install umap-learn
# pip install cuml


df = pd.read_excel('/data/sites.xlsx')
df_w_descr = df.loc[~df.text.isna()]
replaces_re = [
['Быстрый ответ:.* К основным', 'К основным'],
['Добавлены результаты по запросу.*».', '']
]

replaces = [
'Преимущества. Потери трафика при переходе со смартфонов уменьшаются, даже если у пользователей медленный или нестабильный интернет.',
'Некоторые ссылки отсутствуют в результатах поиска в силу требований применимого законодательства.',
'Преимущества. Потери трафика при переходе со смартфонов уменьшаются, даже если у пользователей',
'Рост популярности Турбо-страниц у пользователей положительно влияет на посещаемость сайта.',
'К основным результатамОбратная связь о специальных возможностях',
'поиск картинки видео карты товары переводчик все сервисы',
'Волгоград Google Bing Сообщить об ошибке Настройки Я.ру',
'Владелец сайта предпочёл скрыть описание страницы.',
'Владелец сайта предпочёл описание страницы.',
'Обратная связь о специальных возможностях',
'Снижается нагрузка на хостинг и серверы',
'РКН: сайт нарушает закон РФ',
'К основным результатам',
'Навигационный ответ ',
'Навигационный ответ',
'Сообщить об ошибке',
'Статья в Википедии',
'все сервисы Меню ',
'Видимость сайта ',
'Исходящие ссылки',
'Смотрите также',
'Турбо-страница',
'Отменить Меню',
'Скрыть Меню ',
'Весь список',
'Скрыть Меню',
'Читать ещё',
'Люди ищут ',
'Подробнее',
'Отменить',
'Скрыть ',
'Меню',
]

for el in replaces_re:
    df_w_descr['text'] = df_w_descr['text'].str.replace(el[0], el[1])

for el in replaces:
    df_w_descr['text'] = df_w_descr['text'].str.replace(el, '')

bert_name = 'cointegrated/LaBSE-en-ru'
enc_tokenizer = AutoTokenizer.from_pretrained(bert_name)
encoder = AutoModel.from_pretrained(bert_name)

if torch.cuda.is_available():
    encoder.cuda()


def encode(texts, do_norm=True):
    encoded_input = enc_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = encoder(**encoded_input.to(encoder.device))
        embeddings = model_output.pooler_output
        if do_norm:
            embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings


x = torch.empty(size=(df_w_descr.shape[0], 768))
for i in tqdm(range(df_w_descr.shape[0])):
    x[i] = encode(df_w_descr['text'][i:i+1].values[0])

y = x.cpu().detach().numpy()
print(y.shape)
np.save('y.npy', y)


def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components,
                      min_cluster_size,
                      min_samples=None,
                      random_state=None):
    """
    Returns HDBSCAN objects after first performing dimensionality reduction using UMAP

    Arguments:
        message_embeddings: embeddings to use
        n_neighbors: int, UMAP hyperparameter n_neighbors
        n_components: int, UMAP hyperparameter n_components
        min_cluster_size: int, HDBSCAN hyperparameter min_cluster_size
        min_samples: int, HDBSCAN hyperparameter min_samples
        random_state: int, random seed

    Returns:
        clusters: HDBSCAN object of clusters
    """

    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors,
                                 n_components=n_components,
                                 metric='cosine',
                                 random_state=random_state)
                       .fit_transform(message_embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               min_samples=min_samples,
                               metric='euclidean',
                               gen_min_span_tree=True,
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters


def score_clusters(clusters, prob_threshold=0.05):
    """
    Returns the label count and cost of a given clustering

    Arguments:
        clusters: HDBSCAN clustering object
        prob_threshold: float, probability threshold to use for deciding
                        what cluster labels are considered low confidence

    Returns:
        label_count: int, number of unique cluster labels, including noise
        cost: float, fraction of data points whose cluster assignment has
              a probability below cutoff threshold
    """

    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num)

    return label_count, cost


def random_search(embeddings, space, num_evals):
    """
    Randomly search parameter space of clustering pipeline

    Arguments:
        embeddings: embeddings to use
        space: dict, contains keys for 'n_neighbors', 'n_components',
               and 'min_cluster_size' and values with
               corresponding lists or ranges of parameters to search
        num_evals: int, number of random parameter combinations to try

    Returns:
        df_result: pandas dataframe containing info on each evaluation
                   performed, including run_id, parameters used, label
                   count, and cost
    """

    results = []

    for i in trange(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])
        random_state = space['random_state']

        clusters = generate_clusters(embeddings,
                                     n_neighbors=n_neighbors,
                                     n_components=n_components,
                                     min_cluster_size=min_cluster_size,
                                     random_state=random_state)

        label_count, cost = score_clusters(clusters, prob_threshold=0.05)

        results.append([i, n_neighbors, n_components, min_cluster_size, label_count, cost])

    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components',
                                               'min_cluster_size', 'label_count', 'cost'])

    return result_df.sort_values(by='cost')


def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperopt to minimize

    Arguments:
        params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'random_state' and
               their values to use for evaluation
        embeddings: embeddings to use
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters

    Returns:
        loss: cost function result incorporating penalties for falling
              outside desired range for number of clusters
        label_count: int, number of unique cluster labels, including noise
        status: string, hypoeropt status

        """

    clusters = generate_clusters(embeddings,
                                 n_neighbors=params['n_neighbors'],
                                 n_components=params['n_components'],
                                 min_cluster_size=params['min_cluster_size'],
                                 random_state=params['random_state'])

    label_count, cost = score_clusters(clusters, prob_threshold=0.05)

    # 15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.15
    else:
        penalty = 0

    loss = cost + penalty

    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}


def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayesian search on hyperparameter space using hyperopt

    Arguments:
        embeddings: embeddings to use
        space: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', and 'random_state' and
               values that use built-in hyperopt functions to define
               search spaces for each
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters
        max_evals: int, maximum number of parameter combinations to try

    Saves the following to instance variables:
        best_params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'min_samples', and 'random_state' and
               values associated with lowest cost scenario tested
        best_clusters: HDBSCAN object associated with lowest cost scenario
                       tested
        trials: hyperopt trials object for search

        """

    trials = Trials()
    fmin_objective = partial(objective,
                             embeddings=embeddings,
                             label_lower=label_lower,
                             label_upper=label_upper)

    best = fmin(fmin_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    best_params = space_eval(space, best)
    print('best:')
    print(best_params)
    print(f"label count: {trials.best_trial['result']['label_count']}")

    best_clusters = generate_clusters(embeddings,
                                      n_neighbors=best_params['n_neighbors'],
                                      n_components=best_params['n_components'],
                                      min_cluster_size=best_params['min_cluster_size'],
                                      random_state=best_params['random_state'])

    return best_params, best_clusters, trials


def combine_results(df_ground, cluster_dict):
    """
    Returns dataframe of all documents and each model's assigned cluster

    Arguments:
        df_ground: dataframe of original documents with associated ground truth
                   labels
        cluster_dict: dict, keys as column name for specific model and value as
                      best clusters HDBSCAN object

    Returns:
        df_combined: dataframe of all documents with labels from
                     best clusters for each model

    """

    df_combined = df_ground.copy()

    for key, value in cluster_dict.items():
        df_combined[key] = value.labels_

    return df_combined


def summarize_results(results_dict, results_df):
    """
    Returns a table summarizing each model's performance compared to ground
    truth labels and the model's hyperparametes

    Arguments:
        results_dict: dict, key is the model name and value is a list of:
                      model column name in combine_results output, best_params and best_clusters
                      for each model (e.g. ['label_use', best_params_use, trials_use])
        results_df: dataframe output of combine_results function; dataframe of all documents
                    with labels from best clusters for each model

    Returns:
        df_final: dataframe with each row including a model name, calculated ARI and NMI,
                  loss, label count, and hyperparameters of best model

    """

    summary = []

    for key, value in results_dict.items():
        ground_label = results_df['category'].values
        predicted_label = results_df[value[0]].values

        ari = np.round(adjusted_rand_score(ground_label, predicted_label), 3)
        nmi = np.round(normalized_mutual_info_score(ground_label, predicted_label), 3)
        loss = value[2].best_trial['result']['loss']
        label_count = value[2].best_trial['result']['label_count']
        n_neighbors = value[1]['n_neighbors']
        n_components = value[1]['n_components']
        min_cluster_size = value[1]['min_cluster_size']
        random_state = value[1]['random_state']

        summary.append([key, ari, nmi, loss, label_count, n_neighbors, n_components,
                        min_cluster_size, random_state])

    df_final = pd.DataFrame(summary, columns=['Model', 'ARI', 'NMI', 'loss',
                                              'label_count', 'n_neighbors',
                                              'n_components', 'min_cluster_size',
                                              'random_state'])

    return df_final.sort_values(by='NMI', ascending=False)


def plot_clusters(embeddings, clusters, n_neighbors=15, min_dist=0.1):
    """
    Reduce dimensionality of best clusters and plot in 2D

    Arguments:
        embeddings: embeddings to use
        clusteres: HDBSCAN object of clusters
        n_neighbors: float, UMAP hyperparameter n_neighbors
        min_dist: float, UMAP hyperparameter min_dist for effective
                  minimum distance between embedded points

    """
    umap_data = umap.UMAP(n_neighbors=n_neighbors,
                          n_components=2,
                          min_dist=min_dist,
                          # metric='cosine',
                          random_state=42).fit_transform(embeddings)

    point_size = 100.0 / np.sqrt(embeddings.shape[0])

    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = clusters.labels_

    fig, ax = plt.subplots(figsize=(14, 8))
    outliers = result[result.labels == -1]
    clustered = result[result.labels != -1]
    plt.scatter(outliers.x, outliers.y, color='lightgrey', s=point_size)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=point_size, cmap='jet')
    plt.colorbar()
    plt.show()


hspace = {
    "n_neighbors": hp.choice('n_neighbors', range(3,16)),
    "n_components": hp.choice('n_components', range(3,16)),
    "min_cluster_size": hp.choice('min_cluster_size', range(10,16)), # 2, 16
    "random_state": 42
}

label_lower = 30
label_upper = 100
max_evals = 100

best_params, best_clusters, trials = bayesian_search(y,
                                                    space=hspace,
                                                    label_lower=label_lower,
                                                    label_upper=label_upper,
                                                    max_evals=max_evals)
# то что в принте выше топ параметры
trials.best_trial
best_clusters.labels_[0:100]
best_n_neighbors = best_params['n_neighbors']
best_params
plot_clusters(y, best_clusters)
df_w_descr['labels'] = best_clusters.labels_
df_w_descr.head()
df_w_descr.groupby('labels', as_index=False)[['url_host']].count().sort_values(by='url_host', ascending=False)
df_w_descr.loc[df_w_descr.labels==5]
df_w_descr.loc[df_w_descr.labels==15]
labels_gr = df_w_descr.loc[df_w_descr.labels != -1].groupby(['labels'], as_index=False)[['url_host']].count()

x2 = np.empty([labels_gr.shape[0], 768])

for i, lab in enumerate(labels_gr.labels.unique()):
    rows = list(df_w_descr.loc[df_w_descr.labels == lab].index)
    x2[i] = y[rows, :].mean(axis=0)

y2 = x2.copy()
labels_gr.head()
print(y2.shape)
labels_gr.to_csv('labels_gr.csv', index=False)
np.save('y2.npy', y2)

hspace = {
    "n_neighbors": hp.choice('n_neighbors', range(3,16)), # range(3,16)
    "n_components": hp.choice('n_components', range(3,16)),
    "min_cluster_size": hp.choice('min_cluster_size', range(2,16)), # 2, 16
    "random_state": 42
}

label_lower = 30
label_upper = 100
max_evals = 100


best_params, best_clusters, trials = bayesian_search(y2,
                                                    space=hspace,
                                                    label_lower=label_lower,
                                                    label_upper=label_upper,
                                                    max_evals=max_evals)
labels_gr['labels2'] = best_clusters.labels_
df_w_descr = df_w_descr.merge(labels_gr[['labels', 'labels2']], how='left')
df_w_descr.groupby('labels2', as_index=False)[['labels']].count().sort_values(by='labels', ascending=False)
df_w_descr.loc[df_w_descr.labels2==5]
df_w_descr.labels2 = df_w_descr.labels2.fillna(-1).astype(int)
umap_embeddings = (umap.UMAP(n_neighbors = best_n_neighbors,
                            n_components = 24,
                            metric = 'cosine',
                            random_state = 42)
                        .fit_transform(y))
df_umap_embeddings = pd.DataFrame(umap_embeddings)
df_umap_embeddings.head()
umap_embeddings.shape
umap_embeddings[0]
np.save('y_umap.npy', umap_embeddings)
df_w_descr.to_csv('labels.csv', index=False)