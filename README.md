# GACN
Please refer to the short paper we wrote regarding this work: [Paper](GACN.pdf)

## Introduction
This project is a final project of NLP course.

## Data
We took all the annotated data(train1, train2, test) combined it and randomly split to train and test.

** NOTE: The data for this project is not publicly available, for more details see: https://www.i2b2.org/NLP/Obesity/ **

## Preprocessing, Training and Generating notes
### first step:
preprocessing the data - run preprocessing.py

### second step:
train and evaluate natural clinical notes classification - run classifier_train.py

### third step:
train T5 model and generate new synthetic clinical notes - run generate_synthetic_clinical_notes.py

### fourth step:
train and evaluate natural and synthetic clinical notes classification - run combined_calassifier_train.py
