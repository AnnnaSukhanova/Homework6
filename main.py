import numpy as np
import pandas as pd
import sys
import os
from dataset import CandyDataset


class Utils:
    @staticmethod
    def learn_validate(X, Y, lp=0.7, vp=0.3):
        permutation = np.random.permutation(X.shape[0])
        Xrand = X[permutation]
        Yrand = Y[permutation]

        learnX = Xrand[0:int(lp * Xrand.shape[0])]
        validateX = Xrand[int((lp) * Xrand.shape[0]):int((lp + vp) * Xrand.shape[0])]

        learnY = Yrand[0:int(lp * Yrand.shape[0])]
        validateY = Yrand[int((lp) * Yrand.shape[0]):int((lp + vp) * Yrand.shape[0])]

        return learnX, learnY, validateX, validateY

    @staticmethod
    def to_one_hot(T):
        t_max = max(T)
        result = []
        for t in T:
            encoding = [1 if t == i else 0 for i in range(t_max + 1)]
            result.append(encoding)
        return np.array(result)

    @staticmethod
    def to_digits(onehotY):
        return np.argmax(onehotY, axis=1)

    @staticmethod
    def confusion_to_accuracy(confusion):
        true = 0
        for i in range(confusion.shape[0]):
            true += confusion[i, i]
        return true / np.sum(confusion)

    @staticmethod
    def confusion_to_precision(confusion):
        fp = 0
        tp = 0
        for i in range(conf.shape[0] // 2):
            for j in range(conf.shape[0] // 2):
                tp += confusion[i, j]
        for i in range(conf.shape[0] // 2, conf.shape[0]):
            for j in range(conf.shape[0] // 2):
                fp += confusion[i, j]
        return (tp) / (tp + fp)

    @staticmethod
    def confusion_to_recall(confusion):
        fn = 0
        tp = 0
        for i in range(conf.shape[0] // 2):
            for j in range(conf.shape[0] // 2):
                tp += confusion[i, j]
        for i in range(conf.shape[0] // 2):
            for j in range(conf.shape[0] // 2, conf.shape[0]):
                fn += confusion[i, j]
        return (tp) / (tp + fn)

    @staticmethod
    def confusion_to_f1_score(confusion):
        recall = Utils.confusion_to_recall(confusion)
        precision = Utils.confusion_to_precision(confusion)
        return 2 * (recall * precision) / (recall + precision)


class DecisionNode:
    @staticmethod
    def entropy(vec):
        uniq, probs = np.unique(vec, return_counts=True)
        all = np.sum(probs)
        probs = probs / all
        return -(probs @ np.log2(probs))

    def __init__(self, max_steps, leafe_min_count,
                 max_depth, min_entropy, target_func, depth):
        self._tau = None
        self._is_leafe = None
        self._left = None
        self._right = None
        self._column = None
        self._max_steps = max_steps
        self._depth = depth
        self._leafe_min_count = leafe_min_count
        self._max_depth = max_depth
        self._min_entropy = min_entropy
        self._target_func = target_func

    def fit(self, X, T, classes):
        if self._should_create_leaf(X, T):
            self._is_leafe = True
            self._leafe_fit(X, T, classes)
        else:
            self._is_leafe = False
            self._node_fit(X, T, classes)

    def _leafe_fit(self, X, T, classes):
        uniq, probs = np.unique(T, return_counts=True)
        all = sum(probs)
        self._probabilities = np.zeros((classes))
        probs = probs / all
        for i in range(probs.shape[0]):
            self._probabilities[uniq[i]] = probs[i]

    def _node_fit(self, X, T, classes):
        best_tau = None
        best_column = None
        best_entropy = None
        for colindex in range(X.shape[1]):
            col = X[:, colindex]
            xmin = np.min(col)
            xmax = np.max(col)
            if xmin == xmax:
                continue
            uniq = np.unique(col)
            tries = min(uniq.shape[0], self._max_steps)
            possible_tau = np.random.choice(uniq, tries, replace=False)
            for tau in possible_tau:
                lT = T[col < tau]
                rT = T[col >= tau]
                lP = (lT.shape[0] / (lT.shape[0] + rT.shape[0]))
                rP = (rT.shape[0] / (lT.shape[0] + rT.shape[0]))
                if lT.shape[0] == 0 or rT.shape[0] == 0:
                    continue
                entropy = lP * self._target_func(lT) + rP * self._target_func(rT)
                if best_entropy == None \
                        or (best_entropy > entropy \
                            and tau != possible_tau[0] \
                            and tau != possible_tau[-1]
                ):
                    best_entropy = entropy
                    best_column = colindex
                    best_tau = tau
        self._tau = best_tau
        self._column = best_column

        self._left = DecisionNode(
            depth=self._depth + 1,
            max_steps=self._max_steps,
            max_depth=self._max_depth,
            leafe_min_count=self._leafe_min_count,
            min_entropy=self._min_entropy,
            target_func=self._target_func
        )
        self._right = DecisionNode(
            depth=self._depth + 1,
            max_steps=self._max_steps,
            max_depth=self._max_depth,
            leafe_min_count=self._leafe_min_count,
            min_entropy=self._min_entropy,
            target_func=self._target_func
        )
        lX = self._get_leftX_part(X)
        rX = self._get_rightX_part(X)
        lT = self._get_leftT_part(T, X)
        rT = self._get_rightT_part(T, X)
        self._left.fit(lX, lT, classes)
        self._right.fit(rX, rT, classes)

    def _should_create_leaf(self, X, T):
        return X.shape[0] < self._leafe_min_count \
               or self.entropy(T) < self._min_entropy \
               or self._depth >= self._max_depth

    def _get_leftX_part(self, X):
        return X[X[:, self._column] < self._tau]

    def _get_leftT_part(self, T, X):
        return T[X[:, self._column] < self._tau]

    def _get_rightX_part(self, X):
        return X[X[:, self._column] >= self._tau]

    def _get_rightT_part(self, T, X):
        return T[X[:, self._column] >= self._tau]

    def predict(self, X):
        if self._is_leafe:
            return np.full(X.shape[0], np.argmax(self._probabilities)), \
                   np.tile(self._probabilities, (X.shape[0], 1))
        else:
            lmask = X[:, self._column] < self._tau
            rmask = X[:, self._column] >= self._tau
            lclasses, lprobs = self._left.predict(X[lmask])
            rclasses, rprobs = self._right.predict(X[rmask])

            resulted_classes = np.zeros(lclasses.shape[0] + rclasses.shape[0])
            resulted_classes[lmask] = lclasses
            resulted_classes[rmask] = rclasses

            resulted_probs = np.zeros((lprobs.shape[0] + rprobs.shape[0], rprobs.shape[1]))
            resulted_probs[lmask] = lprobs
            resulted_probs[rmask] = rprobs
            return resulted_classes.astype(int), resulted_probs


class DecissionTree:
    @classmethod
    def confusion(cls, Y, T):
        classes = np.max(T) + 1
        predicted_digits = Y
        real_digits = T
        result = np.zeros((classes, classes))
        for i in range(len(predicted_digits)):
            real = real_digits[i]
            pred = predicted_digits[i]
            result[real, pred] += 1
        return result

    @classmethod
    def find_best_model(cls, Xlearn, Tlearn, Xval, Tval, tries=10):
        depth_max = 20
        depth_min = 2
        steps_min = 10
        steps_max = 1000
        leafecount_min = 1
        leafecount_max = Xlearn.shape[0]
        entropy_min = 0.0001
        target_func = DecisionNode.entropy
        best_accuracy = 0
        best_model = None
        for i in range(tries):
            print(f"{i} TRY")
            max_depth = np.random.randint(depth_min, depth_max)
            max_steps = np.random.randint(steps_min, steps_max)
            leafe_min_count = np.random.randint(leafecount_min, leafecount_max)
            min_entropy = np.random.uniform(entropy_min, target_func(Tlearn))
            model = DecissionTree(max_depth, max_steps, leafe_min_count, min_entropy, target_func)
            model.fit(Xlearn, Tlearn)
            accuracy = model.test(Xval, Tval)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        return best_model

    def __init__(self, max_depth=10, max_steps=100, leafe_min_count=10,
                 min_entropy=0.1, target_func=DecisionNode.entropy):
        self._root = None
        self._node_parameters = {
            "max_depth": max_depth,
            "max_steps": max_steps,
            "leafe_min_count": leafe_min_count,
            "min_entropy": min_entropy,
            "target_func": target_func
        }

    def fit(self, X, T):
        self._root = DecisionNode(**self._node_parameters, depth=0)
        self._root.fit(self._standartize(X), T, max(T) + 1)

    def __str__(self):
        return f"max depth = {self._node_parameters['max_depth']}\n" \
               + f"max steps = {self._node_parameters['max_steps']}\n" \
               + f"leafe minimal set = {self._node_parameters['leafe_min_count']}\n" \
               + f"leafe minimal target functoin value = {self._node_parameters['min_entropy']}\n"

    def _standartize(self, X):
        X = np.copy(X)
        for i in range(X.shape[1]):
            if X[:, i].std() != 0:
                X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
        return X

    def predict(self, X):
        return self._root.predict(self._standartize(X))

    def test(self, X, T):
        digits, probs = self.predict(X)
        conf = DecissionTree.confusion(digits, T)
        return Utils.confusion_to_accuracy(conf)

cd = CandyDataset('candy-data.csv')
X = cd.get_classification_data()[0]
T = cd.get_classification_data()[1]
Xlearn, Tlearn, Xval, Tval = Utils.learn_validate(X, T)
model = DecissionTree(target_func=DecisionNode.entropy)
model.fit(Xlearn, Tlearn)
digits, probs = model.predict(Xval)

print("For validation samples:")
conf = DecissionTree.confusion(digits, Tval)
print(f"Confusion matrix = \n{conf}")
print(f"Accuracy = {Utils.confusion_to_accuracy(conf)}")
print(f"Precision = {Utils.confusion_to_precision(conf)}")
print(f"Recall = {Utils.confusion_to_recall(conf)}")
print(f"F1 score = {Utils.confusion_to_f1_score(conf)}\n")

print("For Learn samples:")
conf2 = DecissionTree.confusion(digits, Tlearn)
print(f"Confusion matrix =  \n{conf2}")
print(f"Accuracy = {Utils.confusion_to_accuracy(conf2)}")
print(f"Precision = {Utils.confusion_to_precision(conf2)}")
print(f"Recall = {Utils.confusion_to_recall(conf2)}")
print(f"F1 score = {Utils.confusion_to_f1_score(conf2)}\n")

print("Start find best model parameters")
model = DecissionTree.find_best_model(Xlearn, Tlearn, Xval, Tval, tries=100)
print(f"Parameters of best model: \n{model}")
