# EVALITA 2023 ACTI challenge - HLT project
This repository contains an analysis on various methods for solving the EVALITA 2023 ACTI ("Automatic Conspiracy Theory Identification") challenge.

The ACTI challenge is an Italian NLP challenge about recognizing Conspiracy theories in Italian Telegram messages. It is composed by two subtasks:
- **[SubtaskA](https://www.kaggle.com/competitions/acti-subtask-a) - Conspiratorial Content Classification:** the model must recognize if the message contains conspiratorial theories or not.
- **[SubtaskB](https://www.kaggle.com/competitions/acti-subtask-b) - Conspiracy Category Classification:** the model must discriminate to which conspiracy theory a message belongs from 4 possible conspiracy topics (Covid, Qanon, Flat Earth, Pro-Russia).

The published paper will soon be published.

## Models
The following model are compared to solve both subtasks:
- **BERT:** we employed [bert-base-italian-xxl-cased](https://huggingface.co/dbmdz/bert-base-italian-xxl-cased), an exclusively Italian-pretrained variant of base BERT and finetuned it with a custom classification head.
- **XLM-RoBERTa**: we employed [XLM-RoBERTa-large](https://huggingface.co/xlm-roberta-large) the multilingual variant of RoBERTa and finetuned it with a custom classification head.
- **LLama**: we employed [Llama 7B](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) as a feature extractor on the samples and used a custom MLP classifier to trained on Llama extracted features.

_Only for SubtaskB:_
- **topic-specific tfidf baseline:** Top _K_ specific keyword to each conspiracy topic are found with _topic-specific tfidf_ (metric calculated using base tfidf). Occurrences of top _K_ keywords for each topic are then used as input of a Random Forest classifier.

### Implementation
We used mainly Pytorch and Transformers libraries.

Each model best hyperparams were found based on their performance over an hold-out validation set.

## Results
The used metric is F1 score macro-averaged. Reported scores are test set results.
### SubtaskA
| Model  | Test score |
| ------------- | ------------- |
| BERT  | 0.8257  |
| XLM-RoBERTa  | 0.8203  |
| Llama  | 0.8022  |

### SubtaskB
| Model  | Test score |
| ------------- | ------------- |
| BERT  | 0.8265  |
| XLM-RoBERTa  | 0.8532  |
| Llama  | 0.7389  |
| Topic-specific tfidf  | 0.7520  |

