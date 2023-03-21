from sklearn.metrics import f1_score

def f1_score_function(labels, preds):
    return f1_score(y_true=labels, y_pred=preds, average='macro')