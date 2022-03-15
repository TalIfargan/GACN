import json
import os
import numpy as np
import pandas as pd
import torch
from torch import cuda
from transformers import AutoTokenizer, T5ForConditionalGeneration
from tqdm import tqdm

LABEL_COLUMNS = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'Hypercholesterolemia',
                 'Hypertension', 'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous Insufficiency']

# Setting up the device for GPU usage

device = 'cuda' if cuda.is_available() else 'cpu'

# Setting up the model name

model_name = "t5-base"

# importing train and eval data

train_dataframe = pd.read_csv('./data/train.csv')
eval_dataframe = pd.read_csv('./data/eval.csv')
train_dataframe['labels'] = train_dataframe['labels'].apply(lambda x: json.loads(x))
eval_dataframe['labels'] = eval_dataframe['labels'].apply(lambda x: json.loads(x))

# randomize new diseases lists

train_p = len(train_dataframe)/(len(train_dataframe)+len(eval_dataframe))
eval_p = len(eval_dataframe)/(len(train_dataframe)+len(eval_dataframe))
disease_prob = (np.mean(train_dataframe['labels'].apply(lambda x: sum(x)).tolist())/(len(LABEL_COLUMNS)))*train_p + \
               (np.mean(eval_dataframe['labels'].apply(lambda x: sum(x)).tolist())/(len(LABEL_COLUMNS)))*eval_p
randomized_diseases_lists = []

zeros_list = [0] * 16
for i in range(len(train_dataframe)):
    randomized_diseases_list = (np.random.choice([0, 1], size=(16,), p=[1 - disease_prob, disease_prob])).tolist()
    while randomized_diseases_list == zeros_list:
        randomized_diseases_list = (np.random.choice([0, 1], size=(16,), p=[1-disease_prob, disease_prob])).tolist()
    randomized_diseases_lists.append(randomized_diseases_list)
diseases_lists = []
for j in range(len(randomized_diseases_lists)):
    diseases_list = ', '.join([y for i, y in enumerate(LABEL_COLUMNS) if randomized_diseases_lists[j][i] == 1])
    diseases_lists.append(diseases_list)


def T5_clinical_notes_generator(model_path, tokenizer, diseases_lists):
    """
    Function to generate new clinical notes from a given list of diseases lists

    """
    task_prefix = 'clinical_note: '
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    clinical_notes = []
    inputs = tokenizer([task_prefix + diseases_list for diseases_list in diseases_lists], return_tensors="pt",
                       padding=True)
    input_ids = inputs['input_ids']
    attention_masks = inputs['attention_mask']
    with torch.no_grad():
        for index in tqdm(range(len(diseases_lists))):
            ids = torch.unsqueeze(input_ids[index].to(device, dtype=torch.long), 0)
            mask = torch.unsqueeze(attention_masks[index].to(device, dtype=torch.long), 0)

            generated_ids = model.generate(
                input_ids=ids,
                min_length=400,
                do_sample=True,
                attention_mask=mask,
                max_length=512,
                #                 num_beams=2,
                top_k=50,
                top_p=0.9,
                repetition_penalty=3.5,
                length_penalty=2.5,
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            clinical_notes.append([', '.join(preds), [diseases_lists[index]]])
        clinical_notes = pd.DataFrame(clinical_notes, columns=["text", "labels"])
        clinical_notes['labels'] = clinical_notes['labels'].apply(lambda x: ', '.join(x))
    return clinical_notes


# defining tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

# generating new synthetic clinical notes

generated_clinical_notes = T5_clinical_notes_generator('outputs/generation_model/pytorch_model.bin', tokenizer,
                                                       diseases_lists)
generated_clinical_notes['labels'] = generated_clinical_notes['labels'].apply(lambda x: [1 if i in x.split(', ') else 0 for i in LABEL_COLUMNS])

natural_and_synthetic_clinical_notes = pd.concat([generated_clinical_notes, train_dataframe])
natural_and_synthetic_clinical_notes.to_csv(os.path.join('./outputs/generation_model', 'combined_clinical_notes.csv'))
