import pandas as pd
import numpy as np

train=pd.read_csv('../selected/train_addrank.csv')
test=pd.read_csv('../selected/test_addrank.csv')

new_feature=train[['EID','TARGET']]
new_feature1=test[['EID']]
#log特征/除特征
features=['ZCZB','FSTINUM','MPNUM','INUM','ETYPE','alter_times','HY','RGYEAR','rZCZB','right_last_year']
for i in range(len(features)):
    for j in range(i+1,len(features)):
        #new_feature['log_'+str(features[i])+'_'+str(features[j])]=np.log1p(train[features[i]]*train[features[j]])
        new_feature['divide_'+str(features[i])+'_'+str(features[j])]=train[features[i]]/(train[features[j]]+1)
        new_feature['divide_'+str(features[j])+'_'+str(features[i])]=train[features[j]]/(train[features[i]]+1)
        #new_feature1['log_'+str(features[i])+'_'+str(features[j])]=np.log1p(test[features[i]]*test[features[j]])
        new_feature1['divide_'+str(features[i])+'_'+str(features[j])]=test[features[i]]/(test[features[j]]+1)
        new_feature1['divide_'+str(features[j])+'_'+str(features[i])]=test[features[j]]/(test[features[i]]+1)
new_feature.to_csv('../selected/train_chufeature.csv',index=False)
new_feature1.to_csv('../selected/test_chufeature.csv',index=False)
