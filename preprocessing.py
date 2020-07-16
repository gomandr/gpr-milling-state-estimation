import numpy as np
import pandas as pd

from create_dataset import create_dataset
from features_selection import get_const_features, get_corr_features
from sklearn.preprocessing import StandardScaler

def dt_to_keep(num_teeth = 8, n = 900/60, R = 5, vf = 2*1000/60, \
        phi_cut = np.pi/2, dt = 0.001):
    # n [rps], R [mm], vf [mm/s], phi_cut [rad], dt [s]
    
    # additionnal tool parameters
    f_z = vf/(num_teeth*n) # feed per tooth [mm] 
    ratio = np.ceil(R/f_z)
    
    # Check if there is always a tooth in the part
    if 2*np.pi/num_teeth < phi_cut : 
        if phi_cut > np.pi/2: 
            num_dt = int(np.ceil(R*(1+np.sin(phi_cut-np.pi/2))/(vf*dt)))   
        else:
            num_dt = int(np.ceil(R/(vf*dt)))                    
    else:
        phi_z = 2*np.pi/num_teeth - phi_cut         # remaining angle of tooth to reach the part surface [°]
        dt_cut = phi_cut / (2*np.pi) /n             # cutting time of one tooth within part [s]
        dt_z = phi_z / (2*np.pi) / n                # remaining time for one tooth to reach the part surface  [s]
        
        num_dt = int(np.ceil(ratio*dt_cut/dt + (ratio-1)+dt_z/dt))  # number of time steps to keep
    
    return num_dt

def split_data(length_set, ratio, seed=1, shuffle=True):

    indices = list(range(length_set))
    split = int(np.floor(length_set * ratio))
    
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    set1_idx, set2_idx = indices[split:], indices[:split]
    return set1_idx, set2_idx
    
def cross_validation_split(length_set, folds=1):  
    indices = list(range(length_set))
    np.random.seed(seed=1)
    np.random.shuffle(indices)

    fold_size = int(len(indices) / folds)+1
    idx_split = [indices[i * fold_size:(i + 1) * fold_size] for i in range((len(indices) + fold_size - 1) // fold_size )]         
        
    return idx_split

def preprocessing_part1(path_to_data=['DATASETS/Shift_x/*.npy', 'DATASETS/Random_action/*.npy'], \
                        data_augmentation=True, zeros_action='delete'):
    
    ##########################################################################
    ### Create dataset from data files and apply some corrections on ps = 0 cases
    print('Creating dataset ...')
    dataset = create_dataset(path_to_data, zeros_action=zeros_action)  

    ###########################################################################
    ### Dataset augmentation 
    if data_augmentation == True:
        print('Performing data augmentation ...')

        # Find how many previous time steps are needed
        num_dt = dt_to_keep()
        #num_dt = 200
        
        # Add #num_dt of previous powers to features of Dataframe
        dataset_augmented = dataset.copy()
        
        inputs = ['ps', 'px', 'py', 'pz']
        for inp in inputs:
            for i in range(num_dt):
                idx_ps = dataset_augmented.columns.get_loc(inp)+(i+1)
                val_1 = np.zeros(i+1)
                df1 = pd.DataFrame({inp: val_1})
                df2 = pd.DataFrame(dataset[inp][:-(i+1)])
                df_added = df1.append(df2, ignore_index=True)
                dataset_augmented.insert(idx_ps, '{}'.format(inp)+'_{}'.format(i+1), df_added.values)
    else:
        dataset_augmented = dataset.copy()
        
    
    ###########################################################################
    ### Split train/val/test
    print('Splitting the dataset ...')
    train_val_idx, test_idx = split_data(len(dataset_augmented), ratio=0.1, seed=1, shuffle=True)
    dataset_train_val_augmented = dataset_augmented.loc[train_val_idx]
    dataset_test_augmented = dataset_augmented.loc[test_idx]
    
    dataset_train_val_augmented = dataset_train_val_augmented.reset_index()
    del dataset_train_val_augmented['index']
    dataset_test_augmented = dataset_test_augmented.reset_index()
    del dataset_test_augmented['index']
    
    return dataset_train_val_augmented, dataset_test_augmented

def preprocessing_part2(dataset_train_augmented, dataset_val_augmented, \
                        dataset_test_augmented, threshold_corr=0.9):    
    
    ###########################################################################
    ### Features selection
    print('Selecting relevant features ...')

    # Correlated
    corr_features = get_corr_features(dataset_train_augmented.iloc[:,:-1], mode='spearman', threshold=threshold_corr)
    dataset_train_augmented.drop(labels=corr_features, axis=1, inplace=True)
    dataset_val_augmented.drop(labels=corr_features, axis=1, inplace=True)
    dataset_test_augmented.drop(labels=corr_features, axis=1, inplace=True)
    
    # Quasi-constant
    cst_features = get_const_features(dataset_train_augmented.iloc[:,:-1], threshold=0.98)  
    dataset_train_augmented.drop(labels=cst_features, axis=1, inplace=True)
    dataset_val_augmented.drop(labels=cst_features, axis=1, inplace=True)
    dataset_test_augmented.drop(labels=cst_features, axis=1, inplace=True)
    
    ###########################################################################
    ### Scaling
    
    print('Scaling the dataset ...')
    scaler = StandardScaler() 
    scaled_train_augmented = pd.DataFrame(scaler.fit_transform(dataset_train_augmented))
    mean = scaler.mean_
    var = scaler.var_
    scaled_val_augmented = pd.DataFrame(scaler.transform(dataset_val_augmented))
    scaled_test_augmented = pd.DataFrame(scaler.transform(dataset_test_augmented))
    
    ###########################################################################
    # Separate predictors and target
    
    headers = list(dataset_train_augmented.columns.values)
    
    X_train = scaled_train_augmented.iloc[:,:-1].to_numpy()
    Y_train = scaled_train_augmented.iloc[:,-1].to_numpy()
    
    X_val = scaled_val_augmented.iloc[:,:-1].to_numpy()
    Y_val = scaled_val_augmented.iloc[:,-1].to_numpy()
    
    X_test = scaled_test_augmented.iloc[:,:-1].to_numpy()
    Y_test = scaled_test_augmented.iloc[:,-1].to_numpy()
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, mean, var, headers