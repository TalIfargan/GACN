import numpy as np
import pandas as pd
import json
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import preprocessing
from torch import cuda

# config

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 8
EPOCHS = 19
LEARNING_RATE = 1e-05
SEED = 42
RUN_NAME = 'bert fine tuned for classification using natural clinical notes'

# setting seed for reproducibility

preprocessing.set_seed(SEED)

# Setting up the device for GPU usage

device = 'cuda' if cuda.is_available() else 'cpu'

# Setting up the model name

model_name = 'bert-base-uncased'

# importing train and eval data

data_files = {
    'train': './data/train.csv',
    'eval': './data/eval.csv'
}
train_dataframe = pd.read_csv(data_files['train'])
eval_dataframe = pd.read_csv(data_files['eval'])
train_dataframe['labels'] = train_dataframe['labels'].apply(lambda x: json.loads(x))
eval_dataframe['labels'] = eval_dataframe['labels'].apply(lambda x: json.loads(x))


class TextClassifierDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = self.data.text
        self.labels = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = str(self.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True

        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.float)
        }


# setting the datasets for train and eval

tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
train_set = TextClassifierDataset(train_dataframe, tokenizer, MAX_LEN)
eval_set = TextClassifierDataset(eval_dataframe, tokenizer, MAX_LEN)

# set model

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=16,
                                                           problem_type="multi_label_classification")


# define metric function

def metric_fn(predictions):
    model_prediction = np.array(predictions.predictions) >= 0.5
    labels = predictions.label_ids
    return {'f1_macro': f1_score(model_prediction, labels, average='macro'),
            'f1_micro': f1_score(model_prediction, labels, average='micro'),
            'accuracy': accuracy_score(model_prediction, labels)}


# set model arguments and huggingface's Trainer


args = TrainingArguments(output_dir="./results",
                         overwrite_output_dir=True,
                         per_device_train_batch_size=TRAIN_BATCH_SIZE,
                         per_device_eval_batch_size=VALID_BATCH_SIZE,
                         save_strategy='no',
                         metric_for_best_model='eval_f1_micro',
                         greater_is_better=True,
                         evaluation_strategy='epoch',
                         do_train=True,
                         num_train_epochs=EPOCHS,
                         # report_to='wandb',
                         run_name=RUN_NAME)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    compute_metrics=metric_fn
)

trainer.train()
trainer.save_model(output_dir='./outputs/classification_model')

