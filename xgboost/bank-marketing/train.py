
import pandas as pd
import numpy as np
import logging

import os
import argparse
import pickle as pkl

from sagemaker_xgboost_container.data_utils import get_dmatrix
import xgboost as xgb

def _xgb_train(params, 
               dtrain, 
               evals, 
               num_boost_round, 
               model_dir, 
               is_master):
    """Run xgb train on arguments given with rabit initialized.

    This is our rabit execution function.

    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training, or is running single node training job. Note that rabit_run will include this argument.
    """
    booster = xgb.train(params=params, dtrain=dtrain, evals=evals, num_boost_round=num_boost_round)

    if is_master:
        model_location = model_dir + '/xgboost-model'
        pkl.dump(booster, open(model_location, 'wb'))
        logging.info("Stored trained model at {}".format(model_location))
        
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--num_round', type=int, default=10)
    parser.add_argument('--objective', type=int)
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    args, _ = parser.parse_known_args()

    print ('Number of rounds -', args.num_round)
    print ('Training data location -', args.train)
    print ('Test data location -', args.test)
    print ('Objective -', args.objective)

    dtrain = get_dmatrix(args.train, 'csv')
    dval = get_dmatrix(args.test, 'csv')

    train_hp = {
        'num_round':args.num_round,
        'objective':args.objective
    }
    
    xgb_train_args = dict(
        params = train_hp,
        dtrain = dtrain,
        evals  = dvalm,
        model_dir=args.model_dir,        
    )
            
    # If single node training, call training method directly.
    if dtrain:
        xgb_train_args['is_master'] = True
        _xgb_train(**xgb_train_args)
    else:
        raise ValueError("Training channel must have data to train model.")    

def model_fn(model_dir):
    """Deserialized and return fitted model.

    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    model_file = 'xgboost-model'
    booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    return booster 
