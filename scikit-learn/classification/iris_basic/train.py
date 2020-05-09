
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.externals import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def prepare_dataset():

    #load iris dataset
    data = load_iris()

    #extract X & y
    X = data.data
    y = data.target
    y= y.reshape(-1,1)
    zipped_data = np.append(X,y,axis=1)
    
    #create df
    iris_df = pd.DataFrame(data = zipped_data,
                           columns=['sepal length','sepal width','petal length','petal width','class'])

    #check df looks ok
    iris_df.head()

    #persist df to file
    os.makedirs('data',exist_ok=True)
    iris_df.to_csv('./data/iris.csv')
    
#prepare_dataset()
def train_model(data_location):
    
    print ('----> created model')
    model = LogisticRegression()

    print ('-----> directory listing',os.listdir(data_location))
    print ('----> reading data file from ', data_location+'/iris.csv')
    data = pd.read_csv(data_location+'/iris.csv',engine='python')
    X = data.iloc[:,1:5]
    y = data.iloc[:,5]

    train_x, test_x, train_y, test_y = train_test_split(X,y)
    
    print ('----> starting model fit')
    model.fit(train_x, train_y)

    print ('----> starting model predictions on test data')
    pred_y = model.predict(test_x)
    print ('F1 score on test data is -',accuracy_score(test_y, pred_y))
    
    return model
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=int(5873/16))
    parser.add_argument('--val_steps', type=int, default=(1476/16))

    # input data and model directories
    #parser.add_argument('--sm-model-dir', type=str, default='.')
    #parser.add_argument('--train', type=str, default='./data/iris.csv')

    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    args, _ = parser.parse_known_args()

    batch_size = args.batch_size
    
    print ('----> started model training')
    model = train_model(args.train)   
    print ('----> ended model training')
    
    print ('----> started model dump')
    joblib.dump(model, os.path.join(args.model_dir,'model.joblib'))
    print ('----> ended model dump')
    
def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf     