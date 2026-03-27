import numpy as np

def precision (gold, y):
    """
    Computes precision for predicted array y given gold array.
    Values are binary: 1 = positive, 0 = negative.
    """
    gold = np.array(gold)
    y = np.array(y)

    tp = np.sum((gold == 1) & (y == 1))
    fp = np.sum((gold == 0) & (y == 1))

    if (tp + fp > 0):
        precision = tp/(tp+fp)
    else:
        precision = 0

    return precision

def recall (gold, y):
    """
    Computes recall for predicted array y given gold array.
    """
    gold = np.array(gold)
    y = np.array(y)

    tp = np.sum((gold == 1) & (y == 1))
    fn = np.sum((gold == 1) & (y == 0))

    if (tp + fn > 0):
        recall= tp/(tp+fn)
    else:
        recall = 0

    return recall

def f1_measure (precison, recall):
    """
    Computes F1 score from macro precision and macro recall.
    """
    f1_m =  2*(precison*recall)/(precison+recall)
    return f1_m

def accuracy(gold, y):
    """
    Computes accuracy for predicted array y given gold array.
    """

    gold = np.array(gold)
    y = np.array(y)

    accuracy = np.mean(gold == y)
    return accuracy

def kappa(gold, y):
    """
    Computes Cohen's kappa using tp, tn, fp, fn (formula from Wikipedia).
    """

    gold = np.array(gold)
    y = np.array(y)

    tp = np.sum((gold == 1) & (y == 1))
    tn = np.sum((gold == 0) & (y == 0))
    fn = np.sum((gold == 1) & (y == 0))
    fp = np.sum((gold == 0) & (y == 1))

    kappa = 2*(tp*tn - fp*fn)/((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn))
    return kappa