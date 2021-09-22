from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import torch
import csv
# labels from file

# load a label 

path = "data/Second_DiCOVA_Challenge_Dev_Data_Release/AUDIO"
folders = ["breathing", "cough", "speech"]

label_file_path = "data/Second_DiCOVA_Challenge_Dev_Data_Release/metadata.csv"
feature_file_path="/Users/bmoell/Code/covid/data/Second_DiCOVA_Challenge_Dev_Data_Release/AUDIO/breathing"

def load_data(label_file_path, feature_file_path):
    data = []
    with open(label_file_path, 'r', newline='') as in_file:
        reader = csv.reader(in_file)
        # skip header
        next(reader)
        for row in reader:

            row = row[0].split(' ')
            # handle parsed row

            filename = row[0]
            label = 1 if row[1] == 'p' else 0

            # load features
            feature = torch.load(feature_file_path + "/" + filename + '.flac.16k.flachubert.pt')

            print(feature, label)

            data.append((feature, label))
    return data



def classify_model(X, y):
    X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    RandomForestClassifier(...)
    print(clf.predict([[0, 0, 0, 0]]))

load_data(label_file_path, feature_file_path)