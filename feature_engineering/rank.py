import pandas as pd

train=pd.read_csv('../selected/train_addloss.csv')
test=pd.read_csv('../selected/test_addloss.csv')

feature_name=['RGYEAR', 'ZCZB', 
       'FINZB', 'ALTAF_ALTBE',
       'ALTDATE_YEAR','alter_times',
       'sub_life','branch_count',
       'right_last_year',
       'right_count', 'BTBL_SUM', 'BTBL', 'BTBL_RATIO', 'invest_life_ratio',
       'BTYEAR', 'invest_IFHOME_rate', 'remain_shares', 'BT_times', 'DJDATE_Y',
       'project_nums',
       'project_IFHOME_rate', 'lawsuit_LAWAMOUNT_sum',
       'lawsuit_LAWAMOUNT_count', 'LAWDATE_Y', 'breakfaith_is_count',
       'breakfaith_is_sum', 'FBDATE_Y_x', 'FBDATE_Y_y','WZCODE_ZP01',
       'WZCODE_ZP02', 'WZCODE_ZP03', 'recruit_RECRNUM_sum',
       'recruit_RECRNUM_count', 'RECDATE_Y_x', 'RECDATE_Y_y',
       'recurt_info_ration']

#train_rank=pd.DataFrame(train.EID,columns=['EID'])
for feature in feature_name:
    train['r'+feature] = train[feature].rank(method='max')

train.to_csv('../selected/train_addrank.csv',index=False)

#test_rank=pd.DataFrame(test.EID,columns=['EID'])
for feature in feature_name:
    test['r'+feature] = test[feature].rank(method='max')
test.to_csv('../selected/test_addrank.csv',index=False)

