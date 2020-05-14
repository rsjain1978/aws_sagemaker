
import pandas as pd
import numpy as np
import os
import argparse

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
def train_model(data_loc_dir, 
                max_depth_param,
               max_features_param):
           
    filename = '/cancer.csv'
    
    print ('---------------> Passed location for data is ->', data_loc_dir)
    print ('---------------> Files at this location are ->', os.listdir(data_loc_dir))
    print ('---------------> Reading data from ->',data_loc_dir+filename)
    
    data = pd.read_csv(data_loc_dir+filename, engine='python')
    
    X = data.iloc[:,1:31]
    y = data.iloc[:,31]
    
    train_x, test_x, train_y, test_y = train_test_split(X,y)
    
    print ('---------------> Starting to fit model')
    print ('\t---------------> Max Depth = ', max_depth_param)
    print ('\t---------------> Max Features = ', max_features_param)
    model = DecisionTreeClassifier(max_depth=max_depth_param,
                                  max_features=max_features_param)
    
    model.fit(train_x, train_y)
    
    print ('---------------> Starting to predict on test data')
    pred_y = model.predict(test_x)
    
    print ('\t---------------> F1-Score %s'%(f1_score(test_y, pred_y)))
    
    return model
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--max_features', type=int, default=20)
    
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    args, _ = parser.parse_known_args()

    max_depth = args.max_depth
    max_features = args.max_features
    
    print ('----> started model training')
    model = train_model(args.train, max_depth,max_features)   
    print ('----> ended model training')
    
    print ('\t----> started model dump')
    joblib.dump(model, os.path.join(args.model_dir,'cancer.model.joblib'))
    print ('\t----> ended model dump')
    
def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "cancer.model.joblib"))
    return clf     