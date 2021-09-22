from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import torch
import csv
import os
import pickle
import numpy
# labels from file

# load a label 

path = "data/Second_DiCOVA_Challenge_Dev_Data_Release/AUDIO"
folders = ["breathing", "cough", "speech"]

label_file_path = "data/Second_DiCOVA_Challenge_Dev_Data_Release/metadata.csv"
feature_file_path="/home/birger/Code/covid-challenge/data/Second_DiCOVA_Challenge_Dev_Data_Release/AUDIO/breathing"

def load_data(label_file_path, feature_file_path):
    features = []
    labels = []
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
            # hubert, mfcc, mel, smile
            full_file_path = feature_file_path + "/" + filename + '.flac.16k.flachubert.pt'
            # print("the path is", full_file_path)
            if os.path.isfile(full_file_path):
                feature = torch.load(feature_file_path + "/" + filename + '.flac.16k.flachubert.pt')
                # print(feature, label)
                features.append(feature)
                labels.append(label)
    return [features, labels]

def classify_model(X, y, X_test, y_test):
    X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X, y)
    print("inside classify model")

    filename = 'models/random_forest.sav'
    pickle.dump(model, open(filename, 'wb'))
    import pdb
    pdb.set_trace()

    result = model.score(X_test, y_test)
    print(result)


def evaluate_model(X_test,y_test, model_path="models/random_forest.sav"):
    # load the model from disk
    loaded_model = pickle.load(open(model_path, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)


full_data = load_data(label_file_path, feature_file_path)
print(full_data)

X = full_data[0]
y = full_data[1]


train_size = int(len(y)*0.8)
test_size = len(y) - train_size

y_train = y[0:train_size]
X_train = X[0:train_size]

y_test = y[train_size:len(y)]
X_test = X[train_size:len(y)]

classify_model(X_train,y_train, X_test, y_test)
# evaluate_model(X_test, y_test)
