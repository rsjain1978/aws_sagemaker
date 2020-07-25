
import pandas as pd
import numpy as np
import os
import argparse

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def train_model(data_loc_dir, 
                max_iter_param,
               class_weight_param):
           
    filename = '/iris.csv'
    
    print ('---------------> Passed location for data is ->', data_loc_dir)
    print ('---------------> Files at this location are ->', os.listdir(data_loc_dir))
    print ('---------------> Reading data from ->',data_loc_dir+filename)
    
    data = pd.read_csv(data_loc_dir+filename, engine='python')
    
    X = data.iloc[:,1:5]
    y = data.iloc[:,5]
    
    train_x, test_x, train_y, test_y = train_test_split(X,y)
    
    print ('---------------> Starting to fit model')
    print ('\t---------------> Max Iteration = ', max_iter_param)
    print ('\t---------------> Class Weigth = ', class_weight_param)
    model = LogisticRegression(max_iter=max_iter_param,
                              class_weight=class_weight_param)
    
    model.fit(train_x, train_y)
    
    print ('---------------> Starting to predict on test data')
    pred_y = model.predict(test_x)
    
    print ('Test Accuracy: %s'%(accuracy_score(test_y, pred_y)))
    print ('Test F1-Score: %s'%(accuracy_score(test_y, pred_y)))
    
    return model

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--class_weight', type=str, default='balanced')
    
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    args, _ = parser.parse_known_args()

    max_iter = args.max_iter
    class_weight = args.class_weight
    
    print ('----> started model training')
    model = train_model(args.train, max_iter,class_weight)   
    print ('----> ended model training')
    
    print ('\t----> started model dump')
    joblib.dump(model, os.path.join(args.model_dir,'model.joblib'))
    print ('\t----> ended model dump')
    
def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf     
