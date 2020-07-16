import numpy as np

def get_corr_features(x_train, mode='pearson', threshold=0.8):
    corr_features = set()
    
    corr_matrix = x_train.corr(mode)
    
    for i in range(len(corr_matrix.columns)-1): # -1 to remove target 
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                corr_features.add(colname)
    
    return corr_features

def get_const_features(x_train, threshold=0.98):
    quasi_constant_features = []
    
    for feature in x_train.columns:
    
        predominant = (x_train[feature].value_counts() / np.float(len(x_train))).sort_values(ascending=False).values[0]
        
        if predominant >= threshold:
            quasi_constant_features.append(feature)   
            
    return quasi_constant_features
