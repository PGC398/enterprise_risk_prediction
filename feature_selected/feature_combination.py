import pandas as pd
import numpy as np

# 提取数据中的数值
def get_number(x):
    if str.isdigit(str(x)):
        return float(str(x))
    else:
        return float(0)

# 根目录
dir = '../data/'

print('reading train.csv')
train = pd.read_csv(dir + 'train.csv')

print('reading test.csv')
test = pd.read_csv(dir + 'evaluation_public.csv')

# 获取企业基本信息表
print('reading 1.entbase')
entbase = pd.read_csv(dir + '1entbase.csv')
print('Processing, please wait........................\n')
# 0填充
entbase = entbase.fillna(0)

print('reading 2alter')
alter = pd.read_csv(dir + '2alter.csv')
print('Processing, please wait........................\n')
# 0填充
alter = alter.fillna(0)

ALTERNO_to_index = list(alter['ALTERNO'].unique())
# 1 2 有金钱变化
alter['ALTERNO'] = alter['ALTERNO'].map(ALTERNO_to_index.index)#改名字

alter['ALTAF'] = np.log1p(alter['ALTAF'].map(get_number))
alter['ALTBE'] = np.log1p(alter['ALTBE'].map(get_number))
alter['ALTAF_ALTBE'] = alter['ALTAF'] - alter['ALTBE']

alter['ALTDATE_YEAR'] = alter['ALTDATE'].map(lambda x:x.split('-')[0])
alter['ALTDATE_YEAR'] = alter['ALTDATE_YEAR'].astype(int)
alter['ALTDATE_MONTH'] = alter['ALTDATE'].map(lambda x:x.split('-')[1])
alter['ALTDATE_MONTH'] = alter['ALTDATE_MONTH'].astype(int)

alter = alter.sort_values(['ALTDATE_YEAR','ALTDATE_MONTH'],ascending=True)#按年 月排序
# 标签化 ALTERNO
alter_ALTERNO = pd.get_dummies(alter['ALTERNO'],prefix='ALTERNO')
alter_ALTERNO_merge = pd.concat([alter['EID'],alter_ALTERNO],axis=1)
alter_ALTERNO_info_sum = alter_ALTERNO_merge.groupby(['EID'],as_index=False).sum()

alter_ALTERNO_info_count = alter_ALTERNO_merge.groupby(['EID'],as_index=False).count()
alter_ALTERNO_info_ration = alter_ALTERNO_merge.groupby(['EID']).sum() / alter_ALTERNO_merge.groupby(['EID']).count()#统计该权利类别出现比例
alter_ALTERNO_info_ration = alter_ALTERNO_info_ration.reset_index()

# 标签化 ALTDATE
alter_ALTDATE=pd.get_dummies(alter['ALTDATE'],prefix='ALTDATE')
alter_ALTDATE_merge=pd.concat([alter['EID'],alter_ALTDATE],axis=1)
alter_ALTDATE_info_sum=alter_ALTDATE_merge.groupby(['EID'],as_index=False).sum()

# 变更的第一年
alter_first_year = pd.DataFrame(alter[['EID','ALTDATE_YEAR']]).drop_duplicates(['EID'])
# 变更的最后一年
alter_last_year = pd.DataFrame(alter[['EID','ALTDATE_YEAR']]).sort_values(['ALTDATE_YEAR'],ascending=False).drop_duplicates(['EID'])

alter_ALTERNO_info = pd.merge(alter_ALTERNO_info_sum,alter[['ALTAF_ALTBE','EID']],on=['EID']).drop_duplicates(['EID'])
alter_ALTERNO_info = pd.merge(alter_ALTERNO_info,alter_last_year,on=['EID'])
alter_ALTERNO_info = pd.merge(alter_ALTERNO_info,alter_ALTDATE_info_sum,on=['EID'])

#统计总变更次数
qwe=alter.groupby(['EID']).count()
qwe.rename(columns = {'ALTERNO':'alter_times'},inplace=True)
alter_ALTERNO_info['alter_times']=qwe['alter_times'].tolist()

alter_ALTERNO_info = alter_ALTERNO_info.fillna(-1)

print('reading 3branch.csv')
branch = pd.read_csv(dir + '3branch.csv')
print('Processing, please wait........................\n')

#统计分支倒闭率
branch_collapse_rate=branch.groupby(['EID'])['B_ENDYEAR'].count()/branch.groupby(['EID'])['TYPECODE'].count()
branch_collapse_rate=branch_collapse_rate.reset_index()
#分支存活时间
branch['B_ENDYEAR'] = branch['B_ENDYEAR'].fillna(branch['B_REYEAR'])
branch['sub_life'] = branch['B_ENDYEAR'].fillna(branch['B_REYEAR']) - branch['B_REYEAR']
# 筛选数据
branch = branch[branch['sub_life']>=0]
branch_count = branch.groupby(['EID'],as_index=False)['TYPECODE'].count()
branch_count.rename(columns = {'TYPECODE':'branch_count'},inplace=True)
branch = pd.merge(branch,branch_count,on=['EID'],how='left')
branch['branch_count'] = branch['branch_count'].astype(int)
branch['sub_life'] = branch['sub_life'].replace({0.0:-1})

home_prob = branch.groupby(by=['EID'])['IFHOME'].sum()/ branch.groupby(by=['EID'])['IFHOME'].count()
home_prob = home_prob.reset_index()
branch = pd.DataFrame(branch[['EID','sub_life']]).drop_duplicates('EID')
branch = pd.merge(branch,home_prob,on=['EID'],how='left')
branch = pd.merge(branch,branch_collapse_rate,on=['EID'],how='left')
branch = pd.merge(branch,branch_count,on=['EID'],how='left')
branch.rename(columns = {0:'branch_collapse_rate'},inplace=True)

print('reading 4invest.csv')
invest = pd.read_csv(dir + '4invest.csv')
print('Processing, please wait........................\n')

#投资外省企业概率
invest_IFHOME_rate=invest.groupby(by=['EID'])['IFHOME'].sum()/invest.groupby(by=['EID'])['IFHOME'].count()
invest_IFHOME_rate=invest_IFHOME_rate.reset_index()
#最近投资时间
invest['BTENDYEAR'] = invest['BTENDYEAR'].fillna(invest['BTYEAR'])
invest['invest_life'] = invest['BTENDYEAR'] - invest['BTYEAR']
#投资次数以及平均投资股份
invest_BTBL_sum = invest.groupby(['EID'],as_index=False)['BTBL'].sum()
invest_BTBL_sum.rename(columns={'BTBL':'BTBL_SUM'},inplace=True)
invest_BTBL_count = invest.groupby(['EID'],as_index=False)['BTBL'].count()
invest_BTBL_sum.rename(columns={'BTBL':'BTBL_COUNT'},inplace=True)
BTBL_INFO = pd.merge(invest_BTBL_sum,invest_BTBL_count,on=['EID'],how='left')
BTBL_INFO['BTBL_RATIO'] = BTBL_INFO['BTBL_SUM'] / BTBL_INFO['BTBL']
invest['invest_life'] = invest['invest_life'] > 0
invest['invest_life'] = invest['invest_life'].astype(int)
invest_life_ratio = invest.groupby(['EID'])['invest_life'].sum() / invest.groupby(['EID'])['invest_life'].count()
invest_life_ratio = invest_life_ratio.reset_index()
invest_life_ratio.rename(columns={'invest_life':'invest_life_ratio'},inplace=True)
invest_last_year = invest.sort_values('BTYEAR',ascending=False).drop_duplicates('EID')[['EID','BTYEAR']]
invest_first_year = invest.sort_values('BTYEAR').drop_duplicates('EID')[['EID','BTYEAR']]
#剩余股份以及被投资次数
BT_cishu=invest.groupby(['BTEID'],as_index=False)['BTBL'].count()
BT_cishu.rename(columns={'BTBL':'BT_times'},inplace=True)
remain=invest.groupby(['BTEID'],as_index=False)['BTBL'].sum()
remain['BTBL']=1-remain['BTBL']
remain=pd.merge(remain,BT_cishu,on=['BTEID'],how='left')
remain.rename(columns={'BTEID':'EID','BTBL':'remain_shares'},inplace=True)
#整合
invest = pd.merge(invest[['EID']],BTBL_INFO,on=['EID'],how='left').drop_duplicates(['EID'])
invest = pd.merge(invest,invest_life_ratio,on=['EID'],how='left')
invest = pd.merge(invest,invest_last_year,on=['EID'],how='left')
invest = pd.merge(invest,invest_IFHOME_rate,on=['EID'],how='left')
invest = pd.merge(invest,remain,on=['EID'],how='left')
invest['IFHOME']=1-invest['IFHOME']
invest.rename(columns={'IFHOME':'invest_IFHOME_rate'},inplace=True)
invest=invest.fillna(-1)

print('reading 5right.csv')
right = pd.read_csv(dir + '5right.csv')
print('Processing, please wait........................\n')

right_RIGHTTYPE = pd.get_dummies(right['RIGHTTYPE'],prefix='RIGHTTYPE')
right_RIGHTTYPE_info = pd.concat([right['EID'],right_RIGHTTYPE],axis=1)
right_RIGHTTYPE_info_sum = right_RIGHTTYPE_info.groupby(['EID'],as_index=False).sum().drop_duplicates(['EID'])
right['ASKDATE_Y'] = right['ASKDATE'].map(lambda x:x.split('-')[0])
right_last_year = right.sort_values('ASKDATE_Y',ascending=False).drop_duplicates('EID')[['EID','ASKDATE_Y']]
right_last_year.rename(columns={'ASKDATE_Y':'right_last_year'},inplace=True)

right_count = right.groupby(['EID'],as_index=False)['RIGHTTYPE'].count()

right_count.rename(columns={'RIGHTTYPE':'right_count'},inplace=True)
right = pd.merge(right[['EID']],right_RIGHTTYPE_info_sum,on=['EID'],how='left').drop_duplicates(['EID'])
right = pd.merge(right,right_last_year,on=['EID'],how='left')
right = pd.merge(right,right_count,on=['EID'],how='left')

print('reading 6project.csv')
project = pd.read_csv(dir + '6project.csv')
print('Processing, please wait........................\n')

project_IFHOME_rate=project.groupby(by=['EID'])['IFHOME'].sum()/project.groupby(by=['EID'])['IFHOME'].count()
project_IFHOME_rate=project_IFHOME_rate.reset_index()

project_nums=project.groupby(['EID'])['TYPECODE'].count()
project_nums=project_nums.reset_index()

project['DJDATE_Y'] = project['DJDATE'].map(lambda x:x.split('-')[0])
project_DJDATE_Y = pd.get_dummies(project['DJDATE_Y'],prefix='DJDATE')
project_DJDATE_Y_info = pd.concat([project['EID'],project_DJDATE_Y],axis=1)
project_DJDATE_Y_info_sum = project_DJDATE_Y_info.groupby(['EID'],as_index=False).sum()
project_DJDATE_Y_info_sum = project_DJDATE_Y_info_sum.drop_duplicates(['EID'])
project_count = project.groupby(['EID'],as_index=False)['DJDATE'].count()
project_count.rename(columns={'DJDATE':'project_count'},inplace=True)

project_last_year = project.sort_values('DJDATE_Y',ascending=False).drop_duplicates('EID')[['EID','DJDATE_Y']]

project = pd.merge(project[['EID']],project_last_year,on=['EID'],how='left').drop_duplicates(['EID'])
project = pd.merge(project,project_DJDATE_Y_info_sum,on=['EID'],how='left')
project = pd.merge(project,project_nums,on=['EID'],how='left')
project = pd.merge(project,project_IFHOME_rate,on=['EID'],how='left')
project['IFHOME']=1-project['IFHOME']
project.rename(columns={'IFHOME':'project_IFHOME_rate','TYPECODE':'project_nums'},inplace=True)


print('reading 7lawsuit.csv')
lawsuit = pd.read_csv(dir + '7lawsuit.csv')
print('Processing, please wait........................\n')

lawsuit = pd.read_csv(dir + '7lawsuit.csv')
lawsuit_LAWAMOUNT_sum = lawsuit.groupby(['EID'],as_index=False)['LAWAMOUNT'].sum()
lawsuit_LAWAMOUNT_sum.rename(columns={'LAWAMOUNT':'lawsuit_LAWAMOUNT_sum'},inplace=True)
#总罚款金额对数化
lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'] = np.log1p(lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'])
lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'] = lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'].astype(int)
lawsuit_LAWAMOUNT_count = lawsuit.groupby(['EID'],as_index=False)['LAWAMOUNT'].count()
lawsuit_LAWAMOUNT_count.rename(columns={'LAWAMOUNT':'lawsuit_LAWAMOUNT_count'},inplace=True)
lawsuit['LAWDATE_Y'] = lawsuit['LAWDATE'].map(lambda x:x.split('-')[0])
lawsuit_last_year = lawsuit.sort_values('LAWDATE_Y',ascending=False).drop_duplicates('EID')[['EID','LAWDATE_Y']]


print('reading 8breakfaith.csv')
breakfaith = pd.read_csv(dir + '8breakfaith.csv')
print('Processing, please wait........................\n')

breakfaith['FBDATE_Y'] = breakfaith['FBDATE'].map(lambda x:x.split('/')[0])
breakfaith_first_year = breakfaith.sort_values('FBDATE_Y').drop_duplicates('EID')[['EID','FBDATE_Y']]
breakfaith_last_year = breakfaith.sort_values('FBDATE_Y',ascending=False).drop_duplicates('EID')[['EID','FBDATE_Y']]

breakfaith['SXENDDATE'] = breakfaith['SXENDDATE'].fillna(0)
breakfaith['is_breakfaith'] = breakfaith['SXENDDATE']!=0
breakfaith['is_breakfaith'] = breakfaith['is_breakfaith'].astype(int)

breakfaith_is_count = breakfaith.groupby(['EID'],as_index=False)['is_breakfaith'].count()
breakfaith_is_sum = breakfaith.groupby(['EID'],as_index=False)['is_breakfaith'].sum()

breakfaith_is_count.rename(columns={'is_breakfaith':'breakfaith_is_count'},inplace=True)
breakfaith_is_sum.rename(columns={'is_breakfaith':'breakfaith_is_sum'},inplace=True)
breakfaith_is_info = pd.merge(breakfaith_is_count,breakfaith_is_sum,on=['EID'],how='left')
breakfaith_is_info = pd.merge(breakfaith_is_info,breakfaith_first_year,on=['EID'],how='left')
breakfaith_is_info = pd.merge(breakfaith_is_info,breakfaith_last_year,on=['EID'],how='left')
breakfaith_is_info['ratio'] = breakfaith_is_info['breakfaith_is_sum'] / breakfaith_is_info['breakfaith_is_count']
print('reading 9recruit.csv')
recruit = pd.read_csv(dir + '9recruit.csv')
print('Processing, please wait........................\n')

recruit['RECDATE_Y'] = recruit['RECDATE'].map(lambda x:x.split('-')[0])
recruit_train_first_year = recruit.sort_values('RECDATE_Y').drop_duplicates('EID')[['EID','RECDATE_Y']]
recruit_train_last_year = recruit.sort_values('RECDATE_Y',ascending=False).drop_duplicates('EID')[['EID','RECDATE_Y']]
recruit_WZCODE = pd.get_dummies(recruit['WZCODE'],prefix='WZCODE')
recruit_WZCODE_merge = pd.concat([recruit['EID'],recruit_WZCODE],axis=1)
# 1
recruit_WZCODE_info_sum = recruit_WZCODE_merge.groupby(['EID'],as_index=False).sum().drop_duplicates(['EID'])
# 2
recruit['RECRNUM'] = recruit['RECRNUM'].fillna(0)
recruit_RECRNUM_count = recruit.groupby(['EID'],as_index=False)['RECRNUM'].count()
recruit_RECRNUM_count.rename(columns={'RECRNUM':'recruit_RECRNUM_count'},inplace=True)
# 3
recruit_RECRNUM_sum = recruit.groupby(['EID'],as_index=False)['RECRNUM'].sum()
recruit_RECRNUM_sum.rename(columns={'RECRNUM':'recruit_RECRNUM_sum'},inplace=True)
recruit_RECRNUM_sum['recruit_RECRNUM_sum'] = recruit_RECRNUM_sum['recruit_RECRNUM_sum']
# 4
recruit_RECRNUM_info = pd.merge(recruit[['EID']],recruit_RECRNUM_sum,on=['EID']).drop_duplicates(['EID'])
recruit_RECRNUM_info = pd.merge(recruit_RECRNUM_info,recruit_RECRNUM_count,on=['EID'])
recruit_RECRNUM_info = pd.merge(recruit_RECRNUM_info,recruit_train_first_year,on=['EID'])
recruit_RECRNUM_info = pd.merge(recruit_RECRNUM_info,recruit_train_last_year,on=['EID'])
recruit_RECRNUM_info['recurt_info_ration'] = recruit_RECRNUM_info['recruit_RECRNUM_sum'] / recruit_RECRNUM_info['recruit_RECRNUM_count']


train = pd.merge(train,entbase,on=['EID'],how='left')
train = pd.merge(train,alter_ALTERNO_info,on=['EID'],how='left')
train = pd.merge(train,branch,on=['EID'],how='left')
train = pd.merge(train,right,on=['EID'],how='left')
train = pd.merge(train,invest,on=['EID'],how='left')
train = pd.merge(train,project,on=['EID'],how='left')
train = pd.merge(train,lawsuit_LAWAMOUNT_sum,on=['EID'],how='left')
train = pd.merge(train,lawsuit_LAWAMOUNT_count,on=['EID'],how='left')
train = pd.merge(train,lawsuit_last_year,on=['EID'],how='left')
train = pd.merge(train,breakfaith_is_info,on=['EID'],how='left')
train = pd.merge(train,recruit_WZCODE_info_sum,on=['EID'],how='left')
train = pd.merge(train,recruit_RECRNUM_info,on=['EID'],how='left')


test = pd.merge(test,entbase,on=['EID'],how='left')
test = pd.merge(test,alter_ALTERNO_info,on=['EID'],how='left')
test = pd.merge(test,branch,on=['EID'],how='left')
test = pd.merge(test,right,on=['EID'],how='left')
test = pd.merge(test,invest,on=['EID'],how='left')
test = pd.merge(test,project,on=['EID'],how='left')
test = pd.merge(test,lawsuit_LAWAMOUNT_sum,on=['EID'],how='left')
test = pd.merge(test,lawsuit_LAWAMOUNT_count,on=['EID'],how='left')
test = pd.merge(test,lawsuit_last_year,on=['EID'],how='left')
test = pd.merge(test,breakfaith_is_info,on=['EID'],how='left')
test = pd.merge(test,recruit_WZCODE_info_sum,on=['EID'],how='left')
test = pd.merge(test,recruit_RECRNUM_info,on=['EID'],how='left')

#test = test.fillna(-999)
#train = train.fillna(-999)

train.to_csv('../selected/train_nofill.csv',index=None)
test.to_csv('../selected/test_nofill.csv',index=None)
