from torch import nn
import torch
from sklearn.metrics import accuracy_score
from transformers import BertModel
import transformers
import tqdm
import numpy as np
from utils import f1_score_function

class BertClassifier(nn.Module) :

    def __init__(self, bert_path = 'm-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0', fine_tune = True, head = None):
        
        super(BertClassifier, self).__init__()

        #BERT initialization
        self.bert = BertModel.from_pretrained(bert_path)

        #Classifier initialization
        if head is not None and head._modules['0'].in_features == 768:
            self.head = head
        
        #If the classifier is not valid, set a default classifier
        else:
            self.head = nn.Sequential(
                nn.Linear(768, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 2),
                )
        
        #Freezing for fine tuning
        if not fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs, mask):
        # Feeding the input to Bert
        last_hidden_states = self.bert(input_ids=inputs, attention_mask=mask)

        # Taking only the CLS : [ all sentences (:), only the first position (0), all hidden unit outputs (:) ] 
        bert_cls = last_hidden_states[0][:,0,:]

        # Classification layer forward
        softmax_preds = self.head(bert_cls)

        # The shape of the softmax_preds is #number of sentences * #number of classes
        return softmax_preds

    def fine_tune(self, enabled = True):
        #Freezing or unfreezing for fine tuning
        for param in self.bert.parameters():
            param.requires_grad = enabled

# Returns initialized model
def init_bert_clf(tr_steps, lr_rate=3e-5, scheduler_warmp_steps=0):
    bert_clf = BertClassifier()

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = transformers.AdamW(params = bert_clf.parameters(), lr=lr_rate, correct_bias=False)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=tr_steps, num_warmup_steps=scheduler_warmp_steps # TO DECIDE
                                                             )
    
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return bert_clf, loss_function, optimizer, scheduler, device




# Train Bert classifier for 1 epoch
def train_bert_clf(model, tr_dataloader, loss_function, optimizer, scheduler, device='cpu'):
    # Put the model into training mode
    model.train()

    loss_total = 0
    predictions, labels = []

    for step, batch in enumerate(tr_dataloader):
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

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
        b_preds = b_preds.detach().cpu().numpy()
        b_labels = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(b_preds)
        labels.append(b_labels)

    # Calculate scores and avg loss
    acc_score = accuracy_score(labels, predictions)
    f1_score = f1_score_function(labels, predictions)
    avg_epoch_loss_tr = loss_total / len(tr_dataloader)  

    return avg_epoch_loss_tr, acc_score, f1_score, model, optimizer, scheduler



# Evaluate
def eval_bert_clf(model, eval_dataloader, loss_function, device='cpu'):
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
            b_preds = b_preds.detach().cpu().numpy()
            b_labels = b_labels.to('cpu').numpy()
            
            # Store predictions and true labels
            predictions.append(b_preds)
            labels.append(b_labels)

    # Calculate scores and avg loss
    acc_score = accuracy_score(labels, predictions)
    f1_score = f1_score_function(labels, predictions)
    avg_epoch_loss_eval = loss_total / len(eval_dataloader)

    return avg_epoch_loss_eval, acc_score, f1_score, predictions, labels