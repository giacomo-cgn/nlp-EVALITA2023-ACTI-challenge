from torch import nn
import torch
from transformers import BertModel
import transformers
import numpy as np

class BertClassifier(nn.Module) :

    def __init__(self, bert_path = 'dbmdz/bert-base-italian-xxl-cased', fine_tune = True, head = None):
        
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
      
                nn.Linear(128, 2),
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
def init_bert_clf(tr_steps, lr_rate=1e-5, scheduler_warmp_steps=None, head=None):
    bert_clf = BertClassifier(head=head)

    if scheduler_warmp_steps == None:
        scheduler_warmp_steps = int(tr_steps/10)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = transformers.AdamW(params = bert_clf.parameters(), lr=lr_rate, correct_bias=False)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_training_steps=tr_steps, num_warmup_steps=scheduler_warmp_steps)
    
    
    return bert_clf, loss_function, optimizer, scheduler