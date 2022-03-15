import numpy as np
import pandas as pd
import json
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    T5ForConditionalGeneration, DataCollatorForSeq2Seq
import preprocessing
from torch import cuda

# config

MAX_LEN = 512
SOURCE_LEN = 32
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-4
SEED = 42
RUN_NAME = 't5 fine tuning for clinical note generation'

# setting seed for reproducibility

preprocessing.set_seed(SEED)

# Setting up the device for GPU usage

device = 'cuda' if cuda.is_available() else 'cpu'

# Setting up the model name

model_name = 't5-base'

# importing train and eval data

# data preprocessing
LABEL_COLUMNS = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'Hypercholesterolemia',
                 'Hypertension', 'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous Insufficiency']

train_dataframe = pd.read_csv('./data/train.csv', names=["target_text", "input_text"], skiprows=1, header=None)
eval_dataframe = pd.read_csv('./data/eval.csv', names=["target_text", "input_text"], skiprows=1, header=None)
train_dataframe['input_text'] = train_dataframe['input_text'].apply(
    lambda x: ', '.join([y for i, y in enumerate(LABEL_COLUMNS) if json.loads(x)[i] == 1]))
eval_dataframe['input_text'] = eval_dataframe['input_text'].apply(
    lambda x: ', '.join([y for i, y in enumerate(LABEL_COLUMNS) if json.loads(x)[i] == 1]))
train_dataframe = train_dataframe.reset_index(drop=True)
eval_dataframe = eval_dataframe.reset_index(drop=True)


class ClinicalNotesGenDataSet(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the neural network for finetuning the model

    """

    def __init__(self, dataframe, tokenizer, source_len, target_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.output_len = target_len
        self.target_text = self.data['target_text']
        self.source_text = self.data['input_text']

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = 'clinical_note: ' + str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data to ensure data is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus([source_text], max_length=self.source_len, pad_to_max_length=True,
                                                  truncation=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_text], max_length=self.output_len, pad_to_max_length=True,
                                                  truncation=True, padding="max_length", return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'input_ids': source_ids.to(dtype=torch.long),
            'attention_mask': source_mask.to(dtype=torch.long),
            'labels': target_ids.to(dtype=torch.long)
            # 'decoder_input_ids': target_ids.to(dtype=torch.long),
            # 'decoder_attention_mask': target_mask.to(dtype=torch.long)
        }


# setting the datasets for train and eval

tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
train_set = ClinicalNotesGenDataSet(train_dataframe, tokenizer, SOURCE_LEN, MAX_LEN)
eval_set = ClinicalNotesGenDataSet(eval_dataframe, tokenizer, SOURCE_LEN, MAX_LEN)

# set model

model = T5ForConditionalGeneration.from_pretrained(model_name)

data_collator = DataCollatorForSeq2Seq(tokenizer, model)

# define metric function

# def metric_fn(predictions):
#     model_prediction = np.array(predictions.predictions) >= 0.5
#     labels = predictions.label_ids
#     return {'f1_macro': f1_score(model_prediction, labels, average='macro'),
#             'f1_micro': f1_score(model_prediction, labels, average='micro'),
#             'accuracy': accuracy_score(model_prediction, labels)}


# set model arguments and huggingface's Trainer


args = TrainingArguments(output_dir="./results/t5_clinical_notes_generator",
                         optim='adafactor',
                         overwrite_output_dir=True,
                         per_device_train_batch_size=TRAIN_BATCH_SIZE,
                         per_device_eval_batch_size=VALID_BATCH_SIZE,
                         evaluation_strategy='steps',
                         eval_steps=100,
                         learning_rate=LEARNING_RATE,
                         num_train_epochs=EPOCHS,
                         logging_strategy='steps',
                         logging_steps=100,
                         report_to='wandb',
                         run_name=RUN_NAME)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(output_dir='./outputs/generation_model')
