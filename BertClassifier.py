from torch import nn
from transformers import BertModel
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
                nn.Linear(32, 2)
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

        return logits

    def fine_tune(self, ft = True):
        #Freezing or unfreezing for fine tuning
        for param in self.bert.parameters():
            param.requires_grad = ft
