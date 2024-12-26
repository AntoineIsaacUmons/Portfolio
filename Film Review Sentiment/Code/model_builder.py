import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel

class ModelBuilder:
    def __init__(self):
        pass

    def build_bert_classifier(self, n_classes, dropout=0.3):
        class BertSentimentClassifier(nn.Module):
            def __init__(self, n_classes, dropout):
                super(BertSentimentClassifier, self).__init__()
                self.bert = BertModel.from_pretrained('bert-base-uncased')
                for param in self.bert.parameters():
                    param.requires_grad = False
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                x = self.dropout(pooled_output)
                logits = self.fc(x)
                return logits

        return BertSentimentClassifier(n_classes, dropout)

    def build_distilbert_classifier(self, n_classes, dropout=0.3):
        class DistilBertSentimentClassifier(nn.Module):
            def __init__(self, n_classes, dropout):
                super(DistilBertSentimentClassifier, self).__init__()
                self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
                for param in self.distilbert.parameters():
                    param.requires_grad = False
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(self.distilbert.config.hidden_size, n_classes)

            def forward(self, input_ids, attention_mask):
                outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                # DistilBERT does not have a pooler_output, so we take the last hidden state
                last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token representation
                x = self.dropout(last_hidden_state)
                logits = self.fc(x)
                return logits

        return DistilBertSentimentClassifier(n_classes, dropout)

    def get_model(self, model_name, n_classes, dropout=0.3):
        if model_name == 'bert':
            return self.build_bert_classifier(n_classes, dropout)
        elif model_name == 'distilbert':
            return self.build_distilbert_classifier(n_classes, dropout)
        else:
            raise ValueError(f"Model {model_name} not recognized.")