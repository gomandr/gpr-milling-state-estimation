import pandas as pd
import numpy as np
import glob 

def create_dataset(path_to_data, zeros_action='delete'):

    # Load and clean dataset   
    data = []
    
    for path in path_to_data:
        files_name = glob.glob(path) 
    
        for file in files_name:
            data.append(np.squeeze(np.load(file, allow_pickle = True)))
            
    #DATASETS without theta
    columns= ['x', 'y', 'z', 'ps', 'px', 'py', 'pz', \
                                       'vibx', 'viby', 'vibz', 'dt', 'Target']

    '''
    # DATASETS with theta
    columns= ['x', 'y', 'z', 'theta', 'ps', 'px', 'py', 'pz', \
                                       'vibx', 'viby', 'vibz', 'dt', 'Target']
    '''
 
    dataset = pd.DataFrame([],columns=columns)
    
    for i in range(len(data)):
        X = data[i][0]      
        Y = data[i][1]
        
        
        '''# DATASETS with theta
        dataset_i = pd.DataFrame({'x': [X[0] for X in X], 'y': [X[1] for X in X], 'z': [X[2] for X in X], \
                                 'theta': [X[3]%(np.pi*2) for X in X], \
                                'ps': [X[4] for X in X], 'px': [X[5] for X in X], 'py': [X[6] for X in X], \
                                'pz': [X[7] for X in X], 'vibx': [X[8] for X in X], 'viby': [X[9] for X in X], \
                                'vibz': [X[10] for X in X], 'dt': [X[11] for X in X], 'Target': Y})
         '''
        # DATASETS without theta
        dataset_i = pd.DataFrame({'x': [X[0] for X in X], 'y': [X[1] for X in X], 'z': [X[2] for X in X], \
                                'ps': [X[3] for X in X], 'px': [X[4] for X in X], 'py': [X[5] for X in X], \
                                'pz': [X[6] for X in X], 'vibx': [X[7] for X in X], 'viby': [X[8] for X in X], \
                                'vibz': [X[9] for X in X], 'dt': [X[10] for X in X], 'Target': Y})
        
        # a = dataset_i.to_numpy() # for visualisation when debugging
        
        # Step 1: Find all spindle powers ps <= 0 indicating that nothing is being cut
        idx = dataset_i[dataset_i['ps']<=0].index.tolist() 
        
        # Step 2: For each ind. with ps <= 0: look at x, y z, derivatives to check if tool moving along this direction
                # This would mean that the powers along these dimensions should also be zero

        if zeros_action == 'interpolate':   # If tool has been cuting before and is cutting after too, then interpolates between both         
          
            inputs = ['px', 'py', 'pz','ps']
            axis = ['x', 'y', 'z', 's']
    
            for inp, ax in zip(inputs, axis): # Iterate over px/x, py/y, pz/z and ps
                for k in idx[1:]: # Iterate over each line where ps = 0
                    to_change = False
                    if ax == 's':
                        to_change = True
                    elif dataset_i[ax][k] - dataset_i[ax][k-1] != 0:
                        to_change = True
                        
                    if to_change == True: # If moving along checked axis or ps
                        next_inp = 0
                        j = 1
                        check_zero = False
                        
                        while next_inp == 0:
                            if k+j > len(dataset_i)-1:
                                prec_inp = 0
                                next_inp = 0
                                check_zero = True
                                break
                            else:
                                next_inp = dataset_i[inp][k+j]
                                j += 1
                        
                        prec_inp = 0
                        if check_zero == False: # If no next value different than 0
                            j = 1
                            while prec_inp == 0:
                                if k-j < 0:
                                    prec_inp = 0
                                    next_inp = 0
                                    break
                                else:
                                    prec_inp = dataset_i[inp][k-j]
                                    j += 1
                                
                        dataset_i[inp][k] = (prec_inp + next_inp) / 2
                        
            # Delete remaining zeros (values before entering and after leaving part)
            idx_to_delete = dataset_i[dataset_i['ps']==0].index.tolist() 
            dataset_i = dataset_i.drop(idx_to_delete, axis=0)
                    
        elif zeros_action == 'delete': # Delete all this entries from dataset
                        dataset_i = dataset_i.drop(idx, axis=0)
                        
        dataset = dataset.append(dataset_i, ignore_index = True) 
    
    return dataset.apply(pd.to_numeric)