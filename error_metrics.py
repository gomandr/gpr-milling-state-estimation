def error_metrics(pred_train_y, train_y, pred_valid_y, valid_y, std_y, \
                   print_error=True, first=True, last=True):
    
    '''
    # Naive previous error
    naive_train = abs(train_y[1:] - train_y[:-1]).mean(axis=0)
    naive_valid = abs(valid_y[1:] - valid_y[:-1]).mean(axis=0)
    '''
    
    # Naive mean error
    naive_train = abs(train_y - train_y.mean(axis=0)).mean(axis=0)
    naive_valid = abs(valid_y - train_y.mean(axis=0)).mean(axis=0)
    
    # MAPE (Mean absolute percentage error)
    mape_train = abs(pred_train_y - train_y).sum() / abs(train_y).sum()
    mape_valid = abs(pred_valid_y - valid_y).sum() / abs(valid_y).sum()
    
    # MAE (mean absolute error)
    mae_train = abs(pred_train_y - train_y).mean(axis=0)
    mae_valid = abs(pred_valid_y - valid_y).mean(axis=0)
    
    # MASE (mean absolue scaled error)
    mase_train = mae_train/naive_train
    mase_valid = mae_valid/naive_valid
      
    return mape_train, mape_valid, mae_train, mae_valid, mase_train,\
        mase_valid, naive_train, naive_valid