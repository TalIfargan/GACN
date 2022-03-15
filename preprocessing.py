import os
import random
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import torch as torch
import yaml
from sklearn.model_selection import train_test_split

CONFIG_PATH = './config'


def xml_to_csv(x_path, y_path, file_name):
    X = pd.read_xml(x_path, xpath="/root/docs/doc")
    ids = pd.read_xml(y_path, xpath='/diseaseset/diseases/disease/doc').id.unique()
    tree = ET.parse(y_path)
    root = tree.getroot()[0]
    diseases = []
    for disease in root:
        diseases.append(disease.attrib['name'])
    disease_dict = {an_id: {d: 0 for d in diseases} for an_id in ids}
    questionable_id = []
    for disease in root:
        for doc in disease:
            if doc.attrib['judgment'] == "Y":
                disease_dict[int(doc.attrib['id'])][disease.attrib['name']] = 1
            elif doc.attrib['judgment'] == "Q":
                questionable_id.append(int(doc.attrib['id']))

    disease_dict_with_labels = {}
    for id in disease_dict:
        if id not in questionable_id:
            disease_dict_with_labels[id] = {'labels': list(disease_dict[id].values())}

    y = pd.DataFrame(disease_dict_with_labels).T.reset_index().rename(columns={"index": "id"})
    dataframe = pd.merge(X, y, on='id').set_index('id')
    dataframe.to_csv(f'./data/{file_name}.csv')


def split_train_test(path1, path2, path3):
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data3 = pd.read_csv(path3)
    full_data = pd.concat([data1, data2, data3]).reset_index(drop=True)
    train_data, test_data = train_test_split(full_data, test_size=0.2)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    train_data.to_csv(f'./data/train.csv')
    test_data.to_csv(f'./data/eval.csv')


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


# set seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    xml_to_csv('./data/obesity_patient_records_training.xml', './data/obesity_standoff_textual_annotations_training.xml', 'train1')
    xml_to_csv('./data/obesity_patient_records_training2.xml', './data/obesity_standoff_annotations_training_addendum3.xml', 'train2')
    xml_to_csv('./data/obesity_patient_records_test.xml', './data/obesity_standoff_annotations_test_textual.xml', 'test')
    split_train_test('./data/train1.csv', './data/train1.csv', './data/test.csv')