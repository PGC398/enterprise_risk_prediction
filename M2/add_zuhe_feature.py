import pandas as pd

train=pd.read_csv('../selected/train_addrank.csv')
test=pd.read_csv('../selected/test_addrank.csv')

train_log=pd.read_csv('../selected/train_logfeature.csv')
test_log=pd.read_csv('../selected/test_logfeature.csv')

train_chu=pd.read_csv('../selected/train_chufeature.csv')
test_chu=pd.read_csv('../selected/test_chufeature.csv')

train['log_HY_RGYEAR']=train_log['log_HY_RGYEAR']
train['log_ETYPE_RGYEAR']=train_log['log_ETYPE_RGYEAR']
train['log_INUM_RGYEAR']=train_log['log_INUM_RGYEAR']
train['log_HY_rZCZB']=train_log['log_HY_rZCZB']
train['log_INUM_HY']=train_log['log_INUM_HY']

test['log_HY_RGYEAR']=test_log['log_HY_RGYEAR']
test['log_ETYPE_RGYEAR']=test_log['log_ETYPE_RGYEAR']
test['log_INUM_RGYEAR']=test_log['log_INUM_RGYEAR']
test['log_HY_rZCZB']=test_log['log_HY_rZCZB']
test['log_INUM_HY']=test_log['log_INUM_HY']

train['divide_HY_RGYEAR']=train_chu['divide_HY_RGYEAR']
train['divide_ETYPE_RGYEAR']=train_chu['divide_ETYPE_RGYEAR']
train['divide_INUM_RGYEAR']=train_chu['divide_INUM_RGYEAR']
train['divide_ZCZB_RGYEAR']=train_chu['divide_ZCZB_RGYEAR']
train['divide_HY_rZCZB']=train_chu['divide_HY_rZCZB']

test['divide_HY_RGYEAR']=test_chu['divide_HY_RGYEAR']
test['divide_ETYPE_RGYEAR']=test_chu['divide_ETYPE_RGYEAR']
test['divide_INUM_RGYEAR']=test_chu['divide_INUM_RGYEAR']
test['divide_ZCZB_RGYEAR']=test_chu['divide_ZCZB_RGYEAR']
test['divide_HY_rZCZB']=test_chu['divide_HY_rZCZB']

train.to_csv('../selected/train.csv',index=False)
test.to_csv('../selected/test.csv',index=False)
