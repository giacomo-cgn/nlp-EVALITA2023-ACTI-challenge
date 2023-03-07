from torch import nn
import torch
from sklearn.metrics import f1_score
from transformers import BertModel
import tqdm
import numpy as np

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
                nn.Softmax()
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
        logits = self.head(bert_cls)

        # The shape of the logits is #number of sentences * #number of classes
        return logits

    def fine_tune(self, enabled = True):
        #Freezing or unfreezing for fine tuning
        for param in self.bert.parameters():
            param.requires_grad = enabled


def f1_score_function(preds, labels):
    preds_copy = torch.tensor(preds)
    #numpy works on cpu, so we ensure that preds and labels are on cpu
    preds_flat = np.argmax(preds_copy.cpu(), axis=1).flatten()
    labels_flat = labels.cpu().flatten()
    return f1_score(labels_flat, preds_flat, average='macro')


def train(model, train_dataloader, optimizer, device='cpu', val_dataloader = None, epochs = 5, loss_function = nn.CrossEntropyLoss()):

    for epoch_i in tqdm(range(epochs)):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'F1 Train':^9} | {'Val Loss':^10} | {'Val Acc':^9} | {'F1 Val':^9} | {'Elapsed':^9}")
        print("-"*95)

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts, f1_value_train_batch, f1_value_train_tot  = 0, 0, 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            #b_labels = b_labels *1.0
            #b_labels = b_labels.unsqueeze(1)

            loss = loss_function(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            f1_value_train_batch+= f1_score_function(logits, b_labels) 
            f1_value_train_tot+= f1_score_function(logits, b_labels) 

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {f1_value_train_batch / batch_counts:^9.2f} | {'-':^10} | {'-':^9} | {'-':^9}")

                # Reset batch tracking variables
                batch_loss, batch_counts, f1_value_train_batch = 0, 0, 0

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        avg_f1_value = f1_value_train_tot / len(train_dataloader)

        print("-"*95)
        # =======================================
        #               Evaluation
        # =======================================
        if val_dataloader != None:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy, f1_value_validation = evaluate(model, val_dataloader)

            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {avg_f1_value:^9.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {f1_value_validation:^9.2f}")
            print("-"*95)
        print("\n")
    
    print("Training complete!")

def evaluate():
    pass
