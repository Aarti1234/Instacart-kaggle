import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle  # for file serialization

# Input data files are available in the "../input/" directory.

from subprocess import check_output
print(check_output(["ls", "D:/kaggle instacart"]).decode("utf8"))

# data loading
import lightgbm as lgb
path = 'D:/kaggle instacart/'


print('loading prior')
priors = pd.read_csv(path + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading train')
train = pd.read_csv(path + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
orders = pd.read_csv(path + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

print('loading products')
products = pd.read_csv(path + 'products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'department_id'])

print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
print('train {}: {}'.format(train.shape, ', '.join(train.columns)))

print('computing product f')
prods = pd.DataFrame()
prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
prods['dow_prior_rate'] = (orders.order_dow / orders.days_since_prior_order ).astype(np.float32)
products = products.join(prods, on='product_id')
products.set_index('product_id', drop=False, inplace=True)
prods.info()
prods.index

# Little exploratory Analysis of data 

# which item was ordered maximum
prods.orders.value_counts().head(1)
# number of items ordered
prods.orders.value_counts().max()

# Plots of different features
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

# which product is being sold most
ax_orders = prods['orders'][1:50].plot(kind = 'bar', figsize = (15,10))
ax_orders.set(xlabel = 'Product Id', ylabel = 'orders')
plt.show()

# which time is busiest
ax_order_dow = orders['order_dow'][1:50].plot(kind = 'bar', figsize = (15,10))
ax_order_dow.set(ylabel = 'Order Day of week', xlabel = 'Index')
plt.show()

# Which day of week is busiest
ax_order_hr_day = orders['order_hour_of_day'][1:50].plot(kind = 'bar', figsize = (15,10))
ax_order_hr_day.set(ylabel = 'Order Hour of Day', xlabel = 'Index')
plt.show()

# X and Y plots/ relationship of order_id and different features

# which order_id has maximum and minimum order hour of day 
# Checking the distribution 
ax_rel_orderid_hour = orders[1:50].plot(x = 'order_id', y = 'order_hour_of_day', kind = 'bar', figsize = (15,10))
ax_rel_orderid_hour.set(ylabel = 'Order Hour of Day', xlabel = 'Order_Id')
plt.show()

# which order has more gap of days since last order

ax_rel_oid_days_prior_order = orders[1:100].plot(x = 'order_id', y = 'days_since_prior_order', kind = 'bar', figsize = (15,10))
ax_rel_oid_days_prior_order.set(ylabel = 'Day since prior order', xlabel = 'Order_Id')
plt.show()

# which order id is average active for which  hour of the day
x = orders.groupby('order_id').mean()['order_hour_of_day']
# what is maximum avarage order hour of day
print('Average minimum hour of day : ',min(x))
print('Average Maximum order hour of day : ', max(x))
print('Order ID with average minimum hour of day :', np.argmin(x))
print('Order ID with average maximum hour of day : ', np.argmax(x))

#plot b/w order_id and average hour of day of that id
'''
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
plt.hist(x[1:10])
plt.xlabel('Order_Id')
plt.ylabel('Average order hour of day per order ID')
'''
# plot of prods dataframe
prods_ = prods[1:1000].plot(figsize = (15,10))
prods_.set(ylabel = 'Prods Features', xlabel = 'Index')
plt.show()

'''
# plot of priors dataframe
priors_ = priors[1:1000].plot(figsize = (15,10))
priors_.set(ylabel = 'Priors Features', xlabel = 'Index')
plt.show()
'''

#plot of products dataframe
products_ = products[1:1000].plot(figsize = (15,10))
products_.set(ylabel = 'Products Features', xlabel = 'Index')
plt.show()

# Which product is reordered most Pie plot
prods['reorders'][1:10].plot.pie(autopct = '%.2f', figsize = (15,10))

print('add order info to priors')
orders.set_index('order_id', inplace=True, drop=False)
priors = priors.join(orders, on='order_id', rsuffix='_')
priors.drop('order_id_', inplace=True, axis=1)

### user features


print('computing user f')
usr = pd.DataFrame()
usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['average_hours_between_orders'] = orders.groupby('user_id')['order_hour_of_day'].mean().astype(np.float32)
usr['median_orders_dow'] = orders.groupby('user_id')['order_dow'].median().astype(np.float32)
usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)

users = pd.DataFrame()
users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

users = users.join(usr)
del usr
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
print('user f', users.shape)

### userXproduct features

print('compute userXproduct f - this is long...')
priors['user_product'] = priors.product_id + priors.user_id * 100000

d= dict()
for row in priors.itertuples():
    z = row.user_product
    if z not in d:
        d[z] = (1,
                (row.order_number, row.order_id),
                row.add_to_cart_order)
    else:
        d[z] = (d[z][0] + 1,
                max(d[z][1], (row.order_number, row.order_id)),
                d[z][2] + row.add_to_cart_order)

print('to dataframe (less memory)')
userXproduct = pd.DataFrame.from_dict(d, orient='index')
del d
userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)
userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
userXproduct.sum_pos_in_cart = userXproduct.sum_pos_in_cart.astype(np.int16)
print('user X product f', len(userXproduct))

del priors

### train / test orders ###
print('split orders : train, test')
test_orders = orders[orders.eval_set == 'test']
train_orders = orders[orders.eval_set == 'train']

train.set_index(['order_id', 'product_id'], inplace=True, drop=False)

### build list of candidate products to reorder, with features ###

def features(selected_orders, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i=0
    for row in selected_orders.itertuples():
        i+=1
        if i%10000 == 0: print('order row',i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train.index for product in user_products]
        
    df = pd.DataFrame({'order_id':order_list, 'product_id':product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list
    
    print('user related features')
    df['product_id'] = df.product_id.map(products.product_id)
    df['user_id'] = df.order_id.map(orders.user_id)
    df['user_total_orders'] = df.user_id.map(users.nb_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
    df['user_average_basket'] =  df.user_id.map(users.average_basket)
    
    print('order related features')
    # df['dow'] = df.order_id.map(orders.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders
    
    print('product related features')
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

    print('user_X_product related features')
    df['z'] = df.user_id * 100000 + df.product_id
    #df.drop(['user_id'], axis=1, inplace=True)
    df['UP_orders'] = df.z.map(userXproduct.nb_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(orders.order_hour_of_day)).map(lambda x: min(x, 24-x)).astype(np.int8)
    #df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(orders.order_dow) == \
    #                                              df.order_id.map(orders.order_dow)

    df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    print(df.dtypes)
    print(df.memory_usage())
    return (df, labels)

# train and test set

df_train, labels = features(train_orders[1:100], labels_given=True)
df_test, _ = features(test_orders[1:100])  

#columns for collaborative filtering 
#columns = ['user_id','product_id']

# columns for light gbm and xgboost for personalized recommendation or   

f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders', 'user_average_basket',
       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
       'aisle_id', 'department_id', 'product_orders', 'product_reorders',
       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
       'UP_delta_hour_vs_last'] 

#1st Algorithm for Ranking

print('formating for lgb')
d_train = lgb.Dataset(df_train[f_to_use],
                      label=labels,
                      categorical_feature=['aisle_id', 'department_id'])  # , 'order_hour_of_day', 'dow'
#del df_train
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
ROUNDS = 100
print('light GBM training')
bst = lgb.train(params, d_train, ROUNDS)

# dumping file or file serialization so that you dont have to run file again and again

pickle.dump(bst, open( "D:/save.p", "wb" ))

'''
# LGB prediction
preds = bst.predict(df_test[f_to_use])
df_test['pred'] = preds
'''
# PLot of Importance of features of LGB Algorithm
lgb.plot_importance(bst, figsize=(15,10))

# Second algorithm for ranking 
# XGBoost Regression

import xgboost as xgb

print('formatting xgboost')

xgb1 = xgb.XGBRegressor(
 learning_rate =0.1,
 n_estimators=100,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 nthread=4,
 scale_pos_weight=1,
 seed=27).fit(df_train[f_to_use].values, labels)     # parameters can be tuned by gridsearch cross validation or randomsearch CV

# dumping train file in pickle or object serialization so that you don't have to run code again and again.
pickle.dump(xgb1,open("D:/save2.p","wb"))

'''
#xgb prediction
pred2 = xgb1.predict(df_test[f_to_use].values)
df_test['pred1'] = pred2
'''
# Importance of features from xgb classifiers 
#importance = xgb1.feature_importances_
xgb.plot_importance(xgb1)
