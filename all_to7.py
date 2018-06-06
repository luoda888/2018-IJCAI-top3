# coding: UTF-8
import pandas as pd
import time
from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
from datetime import datetime,timedelta
import pickle,os
from dateutil.parser import parse
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import lightgbm as lgb 
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_train=pd.read_csv('../data/round2_ijcai_18_train_20180425.txt',delim_whitespace=True)
#data_test_a=pd.read_csv('/test_a_trade_250.txt',delim_whitespace=True)
#data_test_a['is_trade'] = 0
#data_train=pd.concat([data_train,data_test_a]).reset_index(drop=True)
#data_train=data_train.drop_duplicates(['instance_id'])

data_train['item_category_1'],data_train['item_category_2'],data_train['item_category_3']=data_train['item_category_list'].str.split(';',2).str
del data_train['item_category_list']
data_train['predict_category_property_A'],data_train['predict_category_property_B'],data_train['predict_category_property_C']=data_train['predict_category_property'].str.split(';',2).str
del data_train['predict_category_property']
data_train['predict_category_property_A'],data_train['predict_category_property_A_1']=data_train['predict_category_property_A'].str.split(':',1).str
data_train['predict_category_property_A_1'],data_train['predict_category_property_A_2'],data_train['predict_category_property_A_3']=data_train['predict_category_property_A_1'].str.split(',',2).str
del data_train['predict_category_property_A_3']

data_train['predict_category_property_B'],data_train['predict_category_property_B_1']=data_train['predict_category_property_B'].str.split(':',1).str
data_train['predict_category_property_B_1'],data_train['predict_category_property_B_2'],data_train['predict_category_property_B_3']=data_train['predict_category_property_B_1'].str.split(',',2).str
del data_train['predict_category_property_B_3']

data_train['predict_category_property_C'],data_train['predict_category_property_C_1']=data_train['predict_category_property_C'].str.split(':',1).str
data_train['predict_category_property_C_1'],data_train['predict_category_property_C_2'],data_train['predict_category_property_C_3']=data_train['predict_category_property_C_1'].str.split(',',2).str
data_train['predict_category_property_C_1'],data_train['predict_category_property_C_3']=data_train['predict_category_property_C_1'].str.split(';',1).str
data_train['predict_category_property_C_2'],data_train['predict_category_property_C_3']=data_train['predict_category_property_C_2'].str.split(';',1).str
del data_train['predict_category_property_C_3']
#del data_train['predict_category_property_C']

data_train['item_property_list_1'],data_train['item_property_list_2'],data_train['item_property_list_3'],data_train['item_property_list_4']=data_train['item_property_list'].str.split(';',3).str
del data_train['item_property_list_4']
del data_train['item_property_list']
del data_train['item_category_3']
#data_train=data_train.fillna(-1)

##处理类目类特征，将类目类特征分为一列一列
#data_test=pd.read_csv('../data/round2_ijcai_18_test_a_20180425.txt',delim_whitespace=True)
print('load data_test...')
data_test_b=pd.read_csv('../data/round2_ijcai_18_test_b_20180510.txt',delim_whitespace=True)
print('data_test_b:%d' %(len(data_test_b)))
data_test_a=pd.read_csv('../data/round2_ijcai_18_test_a_20180425.txt',delim_whitespace=True)
print('data_test_a:%d' %(len(data_test_a)))
data_test=pd.concat([data_test_b,data_test_a]).reset_index(drop=True)


data_test['item_category_1'],data_test['item_category_2'],data_test['item_category_3']=data_test['item_category_list'].str.split(';',2).str
del data_test['item_category_list']
data_test['predict_category_property_A'],data_test['predict_category_property_B'],data_test['predict_category_property_C']=data_test['predict_category_property'].str.split(';',2).str
del data_test['predict_category_property']

data_test['predict_category_property_A'],data_test['predict_category_property_A_1']=data_test['predict_category_property_A'].str.split(':',1).str
data_test['predict_category_property_A_1'],data_test['predict_category_property_A_2'],data_test['predict_category_property_A_3']=data_test['predict_category_property_A_1'].str.split(',',2).str
del data_test['predict_category_property_A_3']

data_test['predict_category_property_B'],data_test['predict_category_property_B_1']=data_test['predict_category_property_B'].str.split(':',1).str
data_test['predict_category_property_B_1'],data_test['predict_category_property_B_2'],data_test['predict_category_property_B_3']=data_test['predict_category_property_B_1'].str.split(',',2).str
del data_test['predict_category_property_B_3']

data_test['predict_category_property_C'],data_test['predict_category_property_C_1']=data_test['predict_category_property_C'].str.split(':',1).str
data_test['predict_category_property_C_1'],data_test['predict_category_property_C_2'],data_test['predict_category_property_C_3']=data_test['predict_category_property_C_1'].str.split(',',2).str
data_test['predict_category_property_C_1'],data_test['predict_category_property_C_3']=data_test['predict_category_property_C_1'].str.split(';',1).str
data_test['predict_category_property_C_2'],data_test['predict_category_property_C_3']=data_test['predict_category_property_C_2'].str.split(';',1).str
del data_test['predict_category_property_C_3']
#del data_test['predict_category_property_C']

data_test['item_property_list_1'],data_test['item_property_list_2'],data_test['item_property_list_3'],data_test['item_property_list_4']=data_test['item_property_list'].str.split(';',3).str
del data_test['item_property_list_4']
del data_test['item_property_list']

data_train.to_csv('../data/data_train_ori.csv')
data_test.to_csv('../data/data_test_ori.csv')

print('load_data_ori...')
data_train=pd.read_csv('../data/data_train_ori_b.csv')
print(len(data_train))
#data_train['times']=data_train['times'].astype(str)
data_test=pd.read_csv('../data/data_test_ori_b.csv')
print(len(data_test))
#data_test['times']=data_test['times'].astype(str)

def convert_data(data):
    data["times"] = data["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
    data["day"] = data["times"].apply(lambda x: x.day)
    data["hour"] = data["times"].apply(lambda x: x.hour)
    data['min'] = data['times'].apply(lambda x: x.minute)
    data['day']=data['day'].astype('int')
    data['hour']=data['hour'].astype('int')
    data['min']=data['min'].astype('int')
    
    # 小时均值特征
    grouped = data.groupby('user_id')['hour'].mean().reset_index()
    grouped.columns = ['user_id', 'user_mean_hour']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['hour'].mean().reset_index()
    grouped.columns = ['item_id', 'item_mean_hour']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('shop_id')['hour'].mean().reset_index()
    grouped.columns = ['shop_id', 'shop_mean_hour']
    data = data.merge(grouped, how='left', on='shop_id')
    grouped = data.groupby('item_city_id')['hour'].mean().reset_index()
    grouped.columns = ['item_city_id', 'city_mean_hour']
    data = data.merge(grouped, how='left', on='item_city_id')
    grouped = data.groupby('item_brand_id')['hour'].mean().reset_index()
    grouped.columns = ['item_brand_id', 'brand_mean_hour']
    data = data.merge(grouped, how='left', on='item_brand_id')
    # 小时var特征
    '''
    grouped = data.groupby('user_id')['hour'].var().reset_index()
    grouped.columns = ['user_id', 'user_var_hour']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['hour'].var().reset_index()
    grouped.columns = ['item_id', 'item_var_hour']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('shop_id')['hour'].var().reset_index()
    grouped.columns = ['shop_id', 'shop_var_hour']
    data = data.merge(grouped, how='left', on='shop_id')
    grouped = data.groupby('item_city_id')['hour'].var().reset_index()
    grouped.columns = ['item_city_id', 'city_var_hour']
    data = data.merge(grouped, how='left', on='item_city_id')
    grouped = data.groupby('item_brand_id')['hour'].var().reset_index()
    grouped.columns = ['item_brand_id', 'brand_var_hour']
    data = data.merge(grouped, how='left', on='item_brand_id')
    '''
    #天均值特征
    grouped = data.groupby('user_id')['day'].mean().reset_index()
    grouped.columns = ['user_id', 'user_mean_day']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['day'].mean().reset_index()
    grouped.columns = ['item_id', 'item_mean_day']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('shop_id')['day'].mean().reset_index()
    grouped.columns = ['shop_id', 'shop_mean_day']
    data = data.merge(grouped, how='left', on='shop_id')
    grouped = data.groupby('item_city_id')['day'].mean().reset_index()
    grouped.columns = ['item_city_id', 'city_mean_day']
    data = data.merge(grouped, how='left', on='item_city_id')
    grouped = data.groupby('item_brand_id')['day'].mean().reset_index()
    grouped.columns = ['item_brand_id', 'brand_mean_day']
    data = data.merge(grouped, how='left', on='item_brand_id')
    
    #天var特征
    grouped = data.groupby('user_id')['day'].var().reset_index()
    grouped.columns = ['user_id', 'user_var_day']
    data = data.merge(grouped, how='left', on='user_id')
    grouped = data.groupby('item_id')['day'].var().reset_index()
    grouped.columns = ['item_id', 'item_var_day']
    data = data.merge(grouped, how='left', on='item_id')
    grouped = data.groupby('shop_id')['day'].var().reset_index()
    grouped.columns = ['shop_id', 'shop_var_day']
    data = data.merge(grouped, how='left', on='shop_id')
    grouped = data.groupby('item_city_id')['day'].var().reset_index()
    grouped.columns = ['item_city_id', 'city_var_day']
    data = data.merge(grouped, how='left', on='item_city_id')
    grouped = data.groupby('item_brand_id')['day'].var().reset_index()
    grouped.columns = ['item_brand_id', 'brand_var_day']
    data = data.merge(grouped, how='left', on='item_brand_id')

    #天小时均值特征
    grouped = data.groupby(['user_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['user_id', 'day', 'user_mean_day_hour']
    data = data.merge(grouped, how='left', on=['user_id', 'day'])
    grouped = data.groupby(['item_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['item_id', 'day', 'item_mean_day_hour']
    data = data.merge(grouped, how='left', on=['item_id', 'day'])
    grouped = data.groupby(['shop_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['shop_id', 'day', 'shop_mean_day_hour']
    data = data.merge(grouped, how='left', on=['shop_id', 'day'])
    grouped = data.groupby(['item_city_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['item_city_id', 'day', 'city_mean_day_hour']
    data = data.merge(grouped, how='left', on=['item_city_id', 'day'])
    grouped = data.groupby(['item_brand_id', 'day'])['hour'].mean().reset_index()
    grouped.columns = ['item_brand_id', 'day', 'brand_mean_day_hour']
    data = data.merge(grouped, how='left', on=['item_brand_id', 'day'])
    
    #天小时var特征
    grouped = data.groupby(['user_id', 'day'])['hour'].var().reset_index()
    grouped.columns = ['user_id', 'day', 'user_var_day_hour']
    data = data.merge(grouped, how='left', on=['user_id', 'day'])
    grouped = data.groupby(['item_id', 'day'])['hour'].var().reset_index()
    grouped.columns = ['item_id', 'day', 'item_var_day_hour']
    data = data.merge(grouped, how='left', on=['item_id', 'day'])
    grouped = data.groupby(['shop_id', 'day'])['hour'].var().reset_index()
    grouped.columns = ['shop_id', 'day', 'shop_var_day_hour']
    data = data.merge(grouped, how='left', on=['shop_id', 'day'])
    grouped = data.groupby(['item_city_id', 'day'])['hour'].var().reset_index()
    grouped.columns = ['item_city_id', 'day', 'city_var_day_hour']
    data = data.merge(grouped, how='left', on=['item_city_id', 'day'])
    grouped = data.groupby(['item_brand_id', 'day'])['hour'].var().reset_index()
    grouped.columns = ['item_brand_id', 'day', 'brand_var_day_hour']
    data = data.merge(grouped, how='left', on=['item_brand_id', 'day'])

    return data


data_train["times"] = data_train["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
data_train["day"] = data_train["times"].apply(lambda x: x.day)

data_test['is_trade'] = 0
print(len(data_test))

train_data = pd.concat([data_train,data_test],).reset_index(drop=True)
del data_train
del data_test

train_data["datetime"] = train_data["context_timestamp"].apply(lambda x: datetime.datetime.fromtimestamp(x))
train_data["day"] = train_data["datetime"].apply(lambda x: x.day)
train_data["hour"] = train_data["datetime"].apply(lambda x: x.hour)

##按时间排序
train_data = train_data.sort_values('context_timestamp')
train_data["item_category_list"] = train_data["item_category_list"].apply(lambda x: x.split(";"))
train_data["item_property_list"] = train_data["item_property_list"].apply(lambda x: x.split(";"))

categories = train_data["predict_category_property"].apply(lambda x: x.split(";"))
train_data["num_query_cat"] = categories.apply(lambda x: len(x))
for i in range(categories.apply(lambda x: len(x)).max()):
    train_data["category_"+str(i)] = categories.apply(lambda x: x[i].split(":")[0] if len(x)>i else "-1")
    train_data["category_"+str(i)+"_props"] = categories.apply(lambda x: x[i].split(":")[1].split(",") if len(x)>i and x[i].split(":")[0] != "-1" else ["-1"])

#start_time='25'    
#end_time='26'

#train_data=train_data[(train_data.day >= int(start_time)) & (train_data.day <int(end_time))]


train_data["num_item_category"] = train_data["item_category_list"].apply(lambda x: len(x)-x.count("-1"))
train_data["num_item_property"] = train_data["item_property_list"].apply(lambda x: len(x)-x.count("-1")) 


#####################################
##统计各类别在此次出现前的count数
def count_cat_prep(df,column,newcolumn):
    count_dict = {}
    df[newcolumn] = 0
    data = df[[column,newcolumn]].values
    for cat_list in data:
        if cat_list[0] not in count_dict:
            count_dict[cat_list[0]] = 0
            cat_list[1] = 0
        else:
            count_dict[cat_list[0]] += 1
            cat_list[1] = count_dict[cat_list[0]]
    df[[column,newcolumn]] = data

train_data['user_item_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_id'].astype(str)
train_data['user_shop_id'] = train_data['user_id'].astype(str)+"_"+train_data['shop_id'].astype(str)
train_data['user_brand_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_brand_id'].astype(str)
train_data['item_category'] = train_data['item_category_list'].astype(str)
train_data['user_category_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_category'].astype(str)
train_data['user_context_id'] = train_data['user_id'].astype(str)+"_"+train_data['context_id'].astype(str)
train_data['user_city_id'] = train_data['user_id'].astype(str)+"_"+train_data['item_city_id'].astype(str)
##统计各类别在总样本中的count数
for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
    count_cat_prep(train_data,column,column+'_click_count_prep')
    
for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
    train_data = train_data.join(train_data[column].value_counts(),on = column ,rsuffix = '_count')
  
print('gen_gaptime ...')  
##前一次或后一次点击与现在的时间差（trick）

def lasttime_delta(column):    
    train_data[column+'_lasttime_delta'] = 0
    data = train_data[['context_timestamp',column,column+'_lasttime_delta']].values
    lasttime_dict = {}
    for df_list in data:
        if df_list[1] not in lasttime_dict:
            df_list[2] = -1
            lasttime_dict[df_list[1]] = df_list[0]
        else:
            df_list[2] = df_list[0] - lasttime_dict[df_list[1]]
            lasttime_dict[df_list[1]] = df_list[0]
    train_data[['context_timestamp',column,column+'_lasttime_delta']] = data

def nexttime_delta(column):    
    train_data[column+'_nexttime_delta'] = 0
    data = train_data[['context_timestamp',column,column+'_nexttime_delta']].values
    nexttime_dict = {}
    for df_list in data:
        if df_list[1] not in nexttime_dict:
            df_list[2] = -1
            nexttime_dict[df_list[1]] = df_list[0]
        else:
            df_list[2] = nexttime_dict[df_list[1]] - df_list[0]
            nexttime_dict[df_list[1]] = df_list[0]
    train_data[['context_timestamp',column,column+'_nexttime_delta']]= data

for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
    lasttime_delta(column)
    
train_data = train_data.sort_values('context_timestamp',ascending=False)

for column in ['user_id','item_id','item_brand_id','shop_id','user_item_id','user_shop_id','user_brand_id','user_category_id','context_id','item_city_id','user_context_id','user_city_id']:
    nexttime_delta(column)
    
train_data = train_data.sort_values('context_timestamp')


a_prep=train_data[['instance_id','user_id_click_count_prep','item_id_click_count_prep','item_brand_id_click_count_prep','shop_id_click_count_prep','user_item_id_click_count_prep','user_shop_id_click_count_prep',
                   'user_brand_id_click_count_prep','user_category_id_click_count_prep','context_id_click_count_prep','item_city_id_click_count_prep','user_context_id_click_count_prep','user_city_id_click_count_prep']]
a_count=train_data[['instance_id','user_id_count','item_id_count','item_brand_id_count','shop_id_count','user_item_id_count','user_shop_id_count','user_brand_id_count','user_category_id_count',
                    'context_id_count','item_city_id_count','user_context_id_count','user_city_id_count']]

a_instance=train_data['instance_id']
a_gap_time=train_data[['user_id_lasttime_delta','item_id_lasttime_delta','item_brand_id_lasttime_delta','shop_id_lasttime_delta','user_item_id_lasttime_delta',
'user_shop_id_lasttime_delta','user_brand_id_lasttime_delta','user_category_id_lasttime_delta','context_id_lasttime_delta','item_city_id_lasttime_delta',
'user_context_id_lasttime_delta','user_city_id_lasttime_delta',
'user_id_nexttime_delta','item_id_nexttime_delta','item_brand_id_nexttime_delta','shop_id_nexttime_delta','user_item_id_nexttime_delta',
'user_shop_id_nexttime_delta','user_brand_id_nexttime_delta','user_category_id_nexttime_delta','context_id_nexttime_delta','item_city_id_nexttime_delta',
'user_context_id_nexttime_delta','user_city_id_nexttime_delta',
]]


