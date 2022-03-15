## DATA 
We took all the annotated data(train1, train2, test) combined it and randomly split to train and test.

** NOTE: The data for this project is not publiclly available, for more details see: https://www.i2b2.org/NLP/Obesity/ **

## Preprocessing, Training and Generating notes
### first step:
preprocessing the data - run main.py

### second step:
train and evaluate natural clinical notes classification - run classifier_train.py

### third step:
train T5 model and generate new synthetic clinical notes - run generate_synthetic_clinical_notes.py

### fourth step:
train and evaluate natural and synthetic clinical notes classification - run combined_calassifier_train.py

NOTE: all the other files necessary for the preprocessing.
