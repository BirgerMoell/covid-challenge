from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import torch
import csv
import os
import pickle
import numpy
import torch.nn.functional as F
# labels from file

# load a label 

path = "data/Second_DiCOVA_Challenge_Dev_Data_Release/AUDIO"
folders = ["breathing", "cough", "speech"]

label_file_path = "data/Second_DiCOVA_Challenge_Dev_Data_Release/metadata.csv"
feature_file_path="data/Second_DiCOVA_Challenge_Dev_Data_Release/AUDIO/breathing"

longest_length = 1501

def load_data(label_file_path, feature_file_path):
    features = []
    labels = []
    with open(label_file_path, 'r', newline='') as in_file:
        reader = csv.reader(in_file)
        # skip header
        next(reader)

        longest = 0
        lengths = []

        for row in reader:

            #print("the row is", row)

            row = row[0].split(' ')
            # handle parsed row

            filename = row[0]
            label = 1 if row[1] == 'p' else 0

            # load features
            # hubert, mfcc, mel, smile
            full_file_path = feature_file_path + "/" + filename + '.flac.16k.flachubert.pt'
            
            # print("the full file path is", full_file_path)

            # print("the path is", full_file_path)
            if os.path.isfile(full_file_path):
                feature = torch.load(feature_file_path + "/" + filename + '.flac.16k.flachubert.pt')
                #print(len(feature[0]))
                padded_feature = padd_feature(feature, len(feature[0]), longest_length)

                # print(feature, label)
                features.append(padded_feature)
                labels.append(label)


    return [features, labels]

def padd_feature(feature, feature_length, max_length):

    padding_length = max_length - feature_length
    # right, left, top, bottom
    padded_feature = F.pad(feature, (0, 0, padding_length, 0))

    print(padded_feature)

    if len(padded_feature[0]) != max_length:
        print("the padded feature is not the same length as the max length")

        print(max_length)
    else:
        return padded_feature


def classify_model(X, y, X_test, y_test):
    X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X, y)
    print("inside classify model")

    filename = 'models/random_forest.sav'
    pickle.dump(model, open(filename, 'wb'))
  

    # X_test = X_test.reshape(-1,1)
    # y_test = y_test.reshape(-1,1)
    # X_test = numpy.array(X_test)
    # y_test = numpy.array(y_test)
    result = model.score(X_test, y_test)
    import pdb
    pdb.set_trace()
    print(result)


def evaluate_model(X_test,y_test, model_path="models/random_forest.sav"):
    # load the model from disk
    loaded_model = pickle.load(open(model_path, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(result)

full_data = load_data(label_file_path, feature_file_path)
# print(full_data)



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
