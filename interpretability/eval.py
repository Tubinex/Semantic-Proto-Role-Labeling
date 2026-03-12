import numpy as np

def precision (gold, y):
    """
    Berechnet Precision eines Arrays y bei gegebenem Goldwert-Array mit Werten 1(=true) oder 0(=false)
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
    Berechnet Recall eines Arrays y bei gegebenem Goldwert-Array 
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
    Berechnet f1-Score aus macro precison und macro recall
    """
    f1_m =  2*(precison*recall)/(precison+recall)
    return f1_m

def accuracy(gold, y):
    """
    Berechnet Accuracy von Array y bei gegebenem Goldwert array
    """

    gold = np.array(gold)
    y = np.array(y)

    accuracy = np.mean(gold == y)
    return accuracy

def kappa(gold, y):
    """
    Berechnet Kappa score mit fp, fn, tp, tn (Formel hab ich von Wikipedia)
    """

    gold = np.array(gold)
    y = np.array(y)

    tp = np.sum((gold == 1) & (y == 1))
    tn = np.sum((gold == 0) & (y == 0))
    fn = np.sum((gold == 1) & (y == 0))
    fp = np.sum((gold == 0) & (y == 1))

    kappa = 2*(tp*tn - fp*fn)/((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn))
    return kappa