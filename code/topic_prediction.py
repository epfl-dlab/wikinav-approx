import fasttext
import numpy as np
from numpy.linalg import norm
from scipy import stats
import time
import pickle
import sys, os

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def read_articles(fname, model_list, err_fname):
    articles = []; vocab = {}
    ferr = open(err_fname, "w")
    for emb_type in emb_types:
        vocab[emb_type] = set(model_list[emb_type].words)
    for lnum, line in enumerate(open(fname)):
        if lnum == 0:
            header = line.strip().split(",")
            continue
        line = line.strip()
        article_entries = line.split(",")
        page_id = article_entries[0]
        toDiscard = False
        for emb_type in emb_types:
            if page_id not in vocab[emb_type]:
                ferr.write(f'Embeddings for the article {page_id} is missing in model learned from {emb_type} paths!\n')
                toDiscard=True
        if not toDiscard:
            articles.append(line)
    ferr.close()
    return header, articles

def prepare_data(articles, model):
    Ids = []; X = []; Y = []
    vocab = set(model.words)
    for lnum, article in enumerate(articles):
        article_entries = article.split(",")
        page_id = article_entries[0]
        if page_id not in vocab:
            continue
        emb = list(model.get_word_vector(page_id))
        Ids.append(page_id); X.append(emb); Y.append(article_entries[1:])
        if lnum%1000000 == 0:
            print(f'Processed {lnum} articles')
    return np.array(Ids), np.array(X), np.array(Y, dtype=int)

def metrics(true, pred):
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)
    return precision, recall, f1

def classify(clf, X_train, X_val, X_test, y_train, y_val, y_test, categories):
    micro_true_val = []; micro_pred_val = []
    macro_precision_val = []; macro_recall_val = []; macro_f1_val = []

    micro_true = []; micro_pred = []
    macro_precision = []; macro_recall = []; macro_f1 = []

    for idx in range(0,len(categories)):
        category = categories[idx]
        clf.fit(X_train, y_train[:,idx])

        prediction_val = clf.predict(X_val)
        precision_val, recall_val, f1_val = metrics(y_val[:,idx], prediction_val)
        micro_true_val += list(y_val[:,idx]); micro_pred_val += list(prediction_val)
        macro_precision_val.append(precision_val); macro_recall_val.append(recall_val); macro_f1_val.append(f1_val)

        prediction = clf.predict(X_test)
        precision, recall, f1 = metrics(y_test[:,idx], prediction)
        micro_true += list(y_test[:,idx]); micro_pred += list(prediction)
        macro_precision.append(precision); macro_recall.append(recall); macro_f1.append(f1)
        if idx%20 == 0:
            print(f'Completed training for {idx} classes!')

    return macro_precision_val, macro_recall_val, macro_f1_val, micro_true_val, micro_pred_val, macro_precision, macro_recall, macro_f1, micro_true, micro_pred

def evaluate(macro_precision, macro_recall, macro_f1, micro_true, micro_pred):
    micro_precision, micro_recall, micro_f1 = metrics(np.array(micro_true), np.array(micro_pred))

    macro_stats = []; micro_stats = []
    macro_stats.append(np.mean(macro_precision)); macro_stats.append(np.mean(macro_recall)); macro_stats.append(np.mean(macro_f1))
    micro_stats.append(micro_precision); micro_stats.append(micro_recall); micro_stats.append(micro_f1)

    return macro_stats, micro_stats

root_dir = os.path.abspath(os.path.join(os.getcwd(),os.pardir))
PATH_IN = root_dir
PATH_OUT = os.path.join(root_dir, 'downstream_tasks', 'topic_prediction_results')
langlist = ['en', 'ru', 'ja', 'de', 'fr', 'it', 'pl', 'fa']
emb_types = ['real_nav', 'gen_clickstream_private', 'gen_clickstream_public', 'gen_graph']

penalizeMajority = 1

results = {}
for lang in langlist:
    results[lang] = {}
    model_path = os.path.join(PATH_IN, 'data', 'navigation_embeddings', lang)
    fout_topicpred_results = open(os.path.join(PATH_OUT, f'{lang}wiki_topic_pred.tsv'), "w")
    fout_topicpred_results.write(f'EmbeddingType\tRegParam\tMicroPrecisionVal\tMicroRecallVal\tMicroF1Val\tMacroPrecisionVal\tMacroRecallVal\tMacroF1Val\t\tMicroPrecisionTest\tMicroRecallTest\tMicroF1Test\tMacroPrecisionTest\tMacroRecallTest\tMacroF1Test\n')

    articles_fname = os.path.join(PATH_IN, 'data', 'topic_prediction', lang, f'{lang}wiki_topic_prediction.csv')
    articles_err_fname = os.path.join(PATH_IN, 'data', 'topic_prediction', lang, f'{lang}wiki_missing_embeddings_topic_prediction.err')

    model_list = {}
    for emb_type in emb_types:
        tmodel = time.time()
        model_fname = os.path.join(model_path, f'article_representations_{emb_type}_128_5.bin')
        model = fasttext.load_model(model_fname)
        print(f'Loaded model {model_fname} in {time.time()-tmodel} seconds!')
        model_list[emb_type] = model

    header, articles = read_articles(articles_fname, model_list, articles_err_fname)
    print(f'#Articles in {lang}wiki = {len(articles)}')

    articles_train, articles_test = train_test_split(articles, random_state=42, test_size=0.2, shuffle=True)
    articles_val, articles_test = train_test_split(articles_test, random_state=42, test_size=0.5, shuffle=True)
    print(f'#Train = {len(articles_train)}, #Val = {len(articles_val)}, #Test = {len(articles_test)}')

    for emb_type in emb_types:
        results[lang][emb_type] = {}
        model = model_list[emb_type]
        tprep = time.time()
        Ids_train, X_train, y_train = prepare_data(articles_train, model)
        Ids_val, X_val, y_val = prepare_data(articles_val, model)
        Ids_test, X_test, y_test = prepare_data(articles_test, model)
        print(X_train.shape, X_val.shape, X_test.shape)
        print(y_train.shape, y_val.shape, y_test.shape)
        print(f'Data prep done in {time.time() - tprep} seconds')

        for clf_reg in [1]:
            print(f'Regularization param: {clf_reg}')
            if penalizeMajority == 1:
                print(f'Penalize majority class by setting class_weight=balanced')
                logreg = LogisticRegression(random_state=42, max_iter=1000, C=clf_reg, class_weight='balanced')
            else:
                print(f'No penalty on majority class, class_weight=None')
                logreg = LogisticRegression(random_state=42, max_iter=1000, C=clf_reg)
            tclf = time.time()
            categories = header[1:]
            macro_precision_val, macro_recall_val, macro_f1_val, micro_true_val, micro_pred_val, macro_precision, macro_recall, macro_f1, micro_true, micro_pred = classify(logreg, X_train, X_val, X_test, y_train, y_val, y_test, categories)
            print(f'Model training done in {time.time() - tclf} seconds')

            teval = time.time()
            macro_stats_val,  micro_stats_val = evaluate(macro_precision_val, macro_recall_val, macro_f1_val, micro_true_val, micro_pred_val)
            print(f'Eval done in {time.time() - teval} seconds')

            print(f'Validation Macro-avg: Precision = {macro_stats_val[0]}, Recall = {macro_stats_val[1]}, F1 = {macro_stats_val[2]}')
            print(f'Validation Micro-avg: Precision = {micro_stats_val[0]}, Recall = {micro_stats_val[1]}, F1 = {micro_stats_val[2]}')

            teval = time.time()
            macro_stats,  micro_stats = evaluate(macro_precision, macro_recall, macro_f1, micro_true, micro_pred)
            print(f'Eval done in {time.time() - teval} seconds')

            print(f'Test Macro-avg: Precision = {macro_stats[0]}, Recall = {macro_stats[1]}, F1 = {macro_stats[2]}')
            print(f'Test Micro-avg: Precision = {micro_stats[0]}, Recall = {micro_stats[1]}, F1 = {micro_stats[2]}')

            fout_topicpred_results.write(f'{emb_type}\t{clf_reg}\t{micro_stats_val[0]}\t{micro_stats_val[1]}\t{micro_stats_val[2]}\t{macro_stats_val[0]}\t{macro_stats_val[1]}\t{macro_stats_val[2]}\t{micro_stats[0]}\t{micro_stats[1]}\t{micro_stats[2]}\t{macro_stats[0]}\t{macro_stats[1]}\t{macro_stats[2]}\n')

            results[lang][emb_type][clf_reg] = (macro_stats_val, macro_stats, micro_stats_val, micro_stats)
            fout_topicpred_results.flush()
    fout_topicpred_results.close()

pickle.dump(results, open(os.path.join(PATH_OUT, f'topic_prediction_results_master.pickle'), 'wb'))
