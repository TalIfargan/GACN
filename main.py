import preprocessing

if __name__ == '__main__':
    preprocessing.xml_to_csv('./data/obesity_patient_records_training.xml', './data/obesity_standoff_textual_annotations_training.xml', 'train1')
    preprocessing.xml_to_csv('./data/obesity_patient_records_training2.xml', './data/obesity_standoff_annotations_training_addendum3.xml', 'train2')
    preprocessing.xml_to_csv('./data/obesity_patient_records_test.xml', './data/obesity_standoff_annotations_test_textual.xml', 'test')
    preprocessing.split_train_test('./data/train1.csv', './data/train1.csv', './data/test.csv')
