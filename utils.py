from sklearn.metrics import f1_score
import torch
from sklearn.metrics import accuracy_score

# Calculate F1 score with macro average
def f1_score_function(labels, preds):
    return f1_score(y_true=labels, y_pred=preds, average='macro')

# Get available device for training (CPU or GPU)
def get_device():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


# Train Transformer based classifier for 1 epoch
def train_clf(model, tr_dataloader, loss_function, optimizer, scheduler, device='cpu'):
    model.to(device)

    # Put the model into training mode
    model.train()

    loss_total = 0
    predictions, labels = [], []

    for step, batch in enumerate(tr_dataloader):
        # Load batch to GPU
        b_input_ids = batch[0].to(device)
        b_attn_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Zero out any previously calculated gradients
        model.zero_grad()

        # Perform a forward pass
        raw_preds = model(b_input_ids, b_attn_mask)

        loss = loss_function(raw_preds, b_labels)
        loss_total += loss.item()

        # Discretize classes
        _, b_preds = torch.max(raw_preds, dim=1)

        # Perform a backward pass to calculate gradients
        loss.backward()

        # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update optimizer and scheduler
        optimizer.step()
        scheduler.step()

        # Move preds and labels to CPU
        b_preds = b_preds.detach().cpu().tolist()
        b_labels = b_labels.detach().cpu().tolist()
        
        # Store predictions and true labels
        predictions += b_preds
        labels += b_labels

    # Calculate scores and avg loss
    acc_score = accuracy_score(labels, predictions)
    f1_score = f1_score_function(labels, predictions)
    avg_epoch_loss_tr = loss_total / len(tr_dataloader)  

    return avg_epoch_loss_tr, acc_score, f1_score, model, optimizer, scheduler



# Evaluate transformer based classifier
def eval_clf(model, eval_dataloader, loss_function, device='cpu'):
    # Put model into evaluation mode
    model.eval()

    loss_total = 0
    predictions, labels = [], []

    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            
            # Perform a forward pass
            raw_preds = model(b_input_ids, b_attn_mask)

            loss = loss_function(raw_preds, b_labels)
            loss_total += loss.item()
            
            # Discretize classes
            _, b_preds = torch.max(raw_preds, dim=1)

            # Move preds and labels to CPU
            b_preds = b_preds.detach().cpu().tolist()
            b_labels = b_labels.detach().cpu().tolist()
            
            # Store predictions and true labels
            predictions += b_preds
            labels += b_labels

    # Calculate scores and avg loss
    acc_score = accuracy_score(labels, predictions)
    f1_score = f1_score_function(labels, predictions)
    avg_epoch_loss_eval = loss_total / len(eval_dataloader)

    return avg_epoch_loss_eval, acc_score, f1_score, predictions, labels


# Test transformer based classsifier (same to eval, but without labels)
def test_clf(model, test_dataloader, device='cpu'):
    # Put model into evaluation mode
    model.eval()

    predictions, raw = [], []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)
            
            # Perform a forward pass
            raw_preds = model(b_input_ids, b_attn_mask)

            # Discretize classes
            _, b_preds = torch.max(raw_preds, dim=1)

            # Move preds to CPU
            b_preds = b_preds.detach().cpu().tolist()
            
            # Store predictions
            predictions += b_preds

    return predictions