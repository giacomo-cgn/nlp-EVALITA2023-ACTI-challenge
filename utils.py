from sklearn.metrics import f1_score
import torch

def f1_score_function(labels, preds):
    return f1_score(y_true=labels, y_pred=preds, average='macro')


def get_device():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device