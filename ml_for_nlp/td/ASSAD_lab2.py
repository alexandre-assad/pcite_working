#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

class Examples:
    """
    a batch of examples:
    One example is 
    - a BOW vector represented as a python dictionary, for features with non-null values only
    - a gold class

    dict_vectors = list of dictionary BOW vector
    gold_classes = list of gold classes
    """
    def __init__(self):
        self.gold_classes = []
        self.dict_vectors = []

class KNN:
    """
    K-NN for document classification (multiclass)

    members = 

    X_train = matrix of training example vectors
    Y_train = list of corresponding gold classes

    K = maximum number of neighbors to consider

    """
    def __init__(self, X, Y, K=3, weight_neighbors: bool = False, cos_or_dist: bool = True, verbose=False):
        self.X_train = X   # (nbexamples, d)
        self.Y_train = Y   # list of corresponding gold classes

        # nb neighbors to consider
        self.K = K
        self.weight_neighbors = weight_neighbors
        self.cos_or_dist = cos_or_dist
        self.verbose = verbose

    def get_nearest_neighbors(self, x: np.ndarray, k: int | None = None):
        if k is None:
            k = self.K

        if self.cos_or_dist:
            dot = self.X_train @ x
            norm_train = np.linalg.norm(self.X_train, axis=1)
            norm_x = np.linalg.norm(x) + 1e-12
            sims = dot / (norm_train * norm_x)

            idx = np.argsort(-sims)[:k]
            return idx, sims[idx]
        else:
            dists = np.linalg.norm(self.X_train - x, axis=1)
            idx = np.argsort(dists)[:k]  
            sims = 1 / (1 + dists[idx]) 
            return idx, sims

    def predict(self, x: np.ndarray, k: int | None = None):
        idx, sims = self.get_nearest_neighbors(x, k)
        class_scores = {}

        for i, neighbor_idx in enumerate(idx):
            label = self.Y_train[neighbor_idx]
            weight = sims[i] if self.weight_neighbors else 1.0
            class_scores[label] = class_scores.get(label, 0.0) + weight

        return max(class_scores.items(), key=lambda kv: kv[1])[0]

    def evaluate_on_test_set(self, X: np.ndarray, Y: list[str], max_k: int | None = None) -> list[float]:
        """
        Evaluate accuracy on test set
        If max_k is set, test all k from 1..max_k
        """
        if max_k is None:
            max_k = self.K

        results = []
        for k in range(1, max_k + 1):
            correct = 0
            for i in range(X.shape[0]):
                pred = self.predict(X[i], k)
                if pred == Y[i]:
                    correct += 1
            acc = correct / len(Y)
            results.append(acc)
            print(f"Accuracy with k={k}: {acc:.4f}")
        return results


def read_examples(infile):
    """ Reads a .examples file and returns an Examples instance.
    """

    stream = open(infile)
    examples = Examples()
    dict_vector = None
    gold_class = None

    while True:
        line = stream.readline()
        if not line:
            break
        line = line.strip()
        if line.startswith("EXAMPLE_NB"):
            if dict_vector is not None:
                examples.dict_vectors.append(dict_vector)
            dict_vector = {}
            cols = line.split('\t')
            gold_class = cols[3]
            examples.gold_classes.append(gold_class)
        elif line and dict_vector is not None:
            (wordform, val) = line.split('\t')
            dict_vector[wordform] = float(val)

    if dict_vector is not None and len(examples.gold_classes) > len(examples.dict_vectors):
        examples.dict_vectors.append(dict_vector)

    return examples

def build_matrices(examples: Examples, w2i: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    n, m = len(examples.dict_vectors), len(w2i)
    X = np.zeros((n, m))

    for i, vector in enumerate(examples.dict_vectors):
        for word, val in vector.items():
            if word in w2i:
                X[i, w2i[word]] = val

    Y = np.array(examples.gold_classes)
    return X, Y


def compute_idf(train_examples: Examples, w2i: dict[str, int]) -> np.ndarray:
    T = len(train_examples.dict_vectors)
    nb_doc = np.zeros(len(w2i))
    for vec in train_examples.dict_vectors:
        for word in vec.keys():
            if word in w2i:
                nb_doc[w2i[word]] += 1
    idf = np.log(T / (nb_doc + 1e-9))
    return idf


def apply_tfidf(X: np.ndarray, idf: np.ndarray) -> np.ndarray:
    return X * idf


def tune_knn(X_train, Y_train, X_dev, Y_dev, k_values, cos_or_dist_opts, weight_opts, use_idf=False, idf_vector=None):
    results = []
    combos = list(itertools.product(cos_or_dist_opts, weight_opts))
    for cos_opt, weight_opt in combos:
        for K in k_values:
            knn = KNN(X_train, Y_train, K=K, weight_neighbors=weight_opt, cos_or_dist=cos_opt)
            acc_list = knn.evaluate_on_test_set(X_dev, Y_dev, max_k=K)
            acc = acc_list[-1]
            results.append({
                "K": K,
                "cos_or_dist": "cosine" if cos_opt else "distance",
                "weight_neighbors": weight_opt,
                "accuracy": acc
            })
            print(f"K={K}, cos_or_dist={cos_opt}, weight={weight_opt} -> acc={acc:.4f}")

    df = pd.DataFrame(results)
    return df

usage = """ DOCUMENT CLASSIFIER using K-NN algorithm

  prog [options] TRAIN_FILE TEST_FILE

  In TRAIN_FILE and TEST_FILE , each example starts with a line such as:
EXAMPLE_NB	1	GOLD_CLASS	earn

and continue providing the non-null feature values, e.g.:
declared	0.00917431192661
stake	0.00917431192661
reserve	0.00917431192661
...

"""

parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('train_file', help='Examples\' file, used as neighbors', default=None)
parser.add_argument('test_file', help='Examples\' file, used for evaluation', default=None)
parser.add_argument("-k", '--k', default=3, type=int, help='Maximum number of nearest neighbors to consider (all values between 1 and K will be tested). Default=1')
parser.add_argument('-v', '--verbose',action="store_true",default=False,help="If set, triggers a verbose mode. Default=False")
parser.add_argument('-w', '--weight_neighbors', action="store_true", default=False,help="If set, neighbors will be weighted when scoring classes. Default=False")
parser.add_argument("--cos_or_dist", action="store_true", help="Use cosine similarity if set, else euclidean distance")
parser.add_argument("--tune", action="store_true", help="Perform grid search on hyperparameters")
parser.add_argument("--use_idf", action="store_true", help="Use TF-IDF weighting for vectors")

args = parser.parse_args()




#------------------------------------------------------------
# Reading training and test examples :
train_examples = read_examples(args.train_file)
test_examples = read_examples(args.test_file)
dev_examples = read_examples("reuters.train.examples") 
print(train_examples.dict_vectors[0])
#------------------------------------------------------------
# Building indices for vocabulary in TRAINING examples

w2i:dict[str, int] = dict()
i2w:dict[int, str] = dict()
index = 0
for vector in train_examples.dict_vectors:
    for key in vector:
        if key not in w2i:
            w2i[key] = index
            i2w[index] = key
            index += 1

#------------------------------------------------------------
# Organize the data into two matrices for document vectors
#                   and two lists for the gold classes

(X_train, Y_train) = build_matrices(train_examples, w2i)
(X_dev, Y_dev) = build_matrices(dev_examples, w2i)
(X_test, Y_test) = build_matrices(test_examples, w2i)

# Apply TF-IDF if needed
if args.use_idf:
    idf_vec = compute_idf(train_examples, w2i)
    X_train = apply_tfidf(X_train, idf_vec)
    X_dev = apply_tfidf(X_dev, idf_vec)
    X_test = apply_tfidf(X_test, idf_vec)
else:
    idf_vec = None

if args.tune:
    print("Running grid search on dev set...")
    k_values = range(1, 51, 5)
    cos_opts = [True, False]
    weight_opts = [True, False]

    df = tune_knn(X_train, Y_train, X_dev, Y_dev, k_values, cos_opts, weight_opts, args.use_idf, idf_vec)
    print(df)

    # Visualization
    pivot = df.pivot_table(index="K", columns=["cos_or_dist", "weight_neighbors"], values="accuracy")
    pivot.plot(title="KNN accuracy as function of K", figsize=(10,6))
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.legend(title="(cos_or_dist, weight)")
    plt.show()

    # Best configuration
    best_row = df.loc[df["accuracy"].idxmax()]
    print(f"\nBest configuration:\n{best_row}\n")

    # Evaluate on test with best params
    best_knn = KNN(X_train, Y_train,
                   K=int(best_row.K),
                   weight_neighbors=best_row.weight_neighbors,
                   cos_or_dist=(best_row.cos_or_dist == "cosine"))
    test_acc = best_knn.evaluate_on_test_set(X_test, Y_test, max_k=int(best_row.K))[-1]
    print(f"Final accuracy on test set: {test_acc:.4f}")