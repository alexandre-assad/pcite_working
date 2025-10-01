#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
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
    def __init__(self, X, Y, K=3, weight_neighbors=False, verbose=False):
        self.X_train = X   # (nbexamples, d)
        self.Y_train = Y   # list of corresponding gold classes

        # nb neighbors to consider
        self.K = K

        # if True, the nb of neighbors will be weighted by their similarity to the example to classify
        self.weight_neighbors = weight_neighbors

        self.verbose = verbose

    def get_nearest_neighbors(self, x: np.ndarray, k: int | None = None):
        if k is None:
            k = self.K

        dot = self.X_train @ x
        norm_train = np.linalg.norm(self.X_train, axis=1)
        norm_x = np.linalg.norm(x) + 1e-12
        sims = dot / (norm_train * norm_x)

        idx = np.argsort(-sims)[:k]
        return idx, sims[idx]

    def predict(self, x: np.ndarray, k: int | None = None):
        idx, sims = self.get_nearest_neighbors(x, k)
        class_scores = {}

        for i, neighbor_idx in enumerate(idx):
            label = self.Y_train[neighbor_idx]
            weight = sims[i] if self.weight_neighbors else 1.0
            class_scores[label] = class_scores.get(label, 0.0) + weight

        return max(class_scores.items(), key=lambda kv: kv[1])[0]

    def evaluate_on_test_set(self, X: np.ndarray, Y: list[str], max_k: int = None) -> float:
        """
        Evaluate accuracy on test set
        If max_k is set, test all k from 1..max_k
        """
        if max_k is None:
            max_k = self.K

        for k in range(1, max_k + 1):
            correct = 0
            for i in range(X.shape[0]):
                pred = self.predict(X[i], k)
                if pred == Y[i]:
                    correct += 1
            acc = correct / len(Y)
            print(f"Accuracy with k={k}: {acc:.4f}")
        return acc


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

args = parser.parse_args()




#------------------------------------------------------------
# Reading training and test examples :
train_examples = read_examples(args.train_file)
test_examples = read_examples(args.test_file)
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
(X_test, Y_test) = build_matrices(test_examples, w2i)
print(len(Y_test))
print(f"Training matrix has shape {X_train.shape}")
print(f" Testing matrix has shape {X_test.shape}")

myclassifier = KNN(X = X_train,
                   Y = Y_train,
                   K = args.k,
                   weight_neighbors = args.weight_neighbors,
                   verbose=args.verbose)

print("Evaluating on test...")
accuracies = myclassifier.evaluate_on_test_set(X_test, Y_test, max_k=args.k)
print(accuracies)