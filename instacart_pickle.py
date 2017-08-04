import pandas as pd
import pickle

# open lgb pickle file (lgb pickle file = save.p)
bst = pickle.load(open("D:/save.p","rb"))

# LGB prediction
preds = bst.predict(df_test[f_to_use])
df_test['pred'] = preds

# open xgboost pickle file (xgboost pickle file = save2.p )
xgb1 = pickle.load(open("D:/save2.p", "rb"))

#xgb prediction
pred2 = xgb1.predict(df_test[f_to_use].values)
df_test['pred1'] = pred2

def ranking(x,y):
	TRESHOLD = 0.22      # can be tuned with crossval on a subset of train data
	d = dict()
	for row in df_test.itertuples():
		if row.x > TRESHOLD:
			 try:
				  d[row.order_id] += ' ' + str(row.product_id)
			 except:
				 d[row.order_id] = str(row.product_id)  
	for 	order in test_orders.order_id:
		if order not in d:
			d[order] = 'None'
			
	sub = pd.DataFrame.from_dict(d, orient='index')
	sub.reset_index(inplace=True)
	sub.columns = ['order_id', 'products']
	file_ = sub.to_csv('D:/y.csv', index=False) 
	return(file_)	 
		    
# writing in CSV
ranking(pred, lgb_output)
ranking(pred1, xgboost_output)

# if above function doesn't work then just run following twice for both algorithm
# by changing row name and file name, I am doing for xgboost

'''
TRESHOLD = 0.22      # can be tuned with crossval on a subset of train data
d = dict()
for row in df_test.itertuples():
	if row.pred1 > TRESHOLD:
		 try:
			  d[row.order_id] += ' ' + str(row.product_id)
		 except:
			 d[row.order_id] = str(row.product_id)  

for 	order in test_orders.order_id:
	if order not in d:
		d[order] = 'None'


sub = pd.DataFrame.from_dict(d, orient='index')
sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv('D:/xgboost_output.csv', index=False)

'''
     
    
    