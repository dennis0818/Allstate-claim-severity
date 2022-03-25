#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import cross_val_score #视频中sklearn.cross_validation找不到，可以改为model_selection

from scipy import stats
import seaborn as sns
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 好事达保险赔偿数据集

# In[2]:


train = pd.read_csv('D:/python/practise/sample/XGBoost/allstate claims severity/train.csv')
test = pd.read_csv('D:/python/practise/sample/XGBoost/allstate claims severity/test.csv')


# In[3]:


train.shape


# In[4]:


train.describe()


# In[5]:


train.info()


# In[6]:


test.drop('Unnamed: 131', axis = 1, inplace = True)


# In[7]:


train.drop('Unnamed: 132', axis = 1, inplace = True)


# In[8]:


#category离散的特征；continuous连续的特征
cat_features = list(train.select_dtypes(include = ['object']).columns)
cont_features = [x for x in
                 (list(train.select_dtypes(include = ['float', 'int']).columns))
                 if x not in ['id', 'loss']]


# In[9]:


cat_features_test = list(test.select_dtypes(include = ['object']).columns)
cont_features_test = [x for x in (list(test.select_dtypes(include = ['float', 'int']).columns))
                      if x not in ['id', 'loss']]


# In[10]:


len(cat_features), len(cont_features)


# In[11]:


len(cat_features_test), len(cont_features_test)


# In[12]:


cat_uniques = []
for cat in cat_features:
    cat_uniques.append(len(train[cat].unique()))
value_counts_in_categories = pd.DataFrame({'cat_name' : cat_features, 'value_counts' : cat_uniques})


# In[13]:


cat_uniques_test = []
for cat in cat_features_test:
    cat_uniques_test.append(len(test[cat].unique()))
value_counts_in_categories_test = pd.Series(cat_uniques_test, index = cat_features_test)


# In[14]:


#category numbers
value_counts_in_categories['value_counts'].value_counts()


# In[15]:


fig, ax = plt.subplots(1, 2, figsize = (16, 6))
ax[0].hist(train['loss'].values, bins = 150)
ax[1].hist(np.log(train['loss'].values), bins = 150, color = 'lightgreen')
ax[0].grid()
ax[1].grid()
plt.show()


# In[16]:


#检验数据的偏度（统计学定义skewness），skew=0为对称的，大于1或小于-1为高度倾斜数据
stats.mstats.skew(train['loss'].values).data, stats.mstats.skew(np.log(train['loss'].values)).data


# In[17]:


plt.subplots(figsize = (16, 9))
corr_matrix = train[cont_features].corr()
sns.heatmap(corr_matrix, annot = True)


# ## Xgboost调参策略

# In[18]:


import xgboost as xgb
from xgboost import XGBRegressor
import pickle #持久化模块，可以将对象序列化后以文件形式存放在磁盘上
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack
#凡是sklearn.cross_validation找不到的问题，都改为model_selection
from sklearn.model_selection import KFold, train_test_split, GridSearchCV


# In[19]:


train['log_loss'] = np.log(train['loss'])
features = [x for x in train.columns if x not in ['id', 'loss', 'log_loss']]
X = train[features]
y = train['log_loss']
for cat in cat_features:
    # .cat 将astype后的Series对象变为categorical对象
    X[cat] = X[cat].astype('category').cat.codes


# In[20]:


X.shape


# In[21]:


features_test = [x for x in test.columns if x not in ['id']]
X_test = test[features_test]
for cat in cat_features_test:
    X_test[cat] = X_test[cat].astype('category').cat.codes


# In[22]:


for cont in cont_features:
    X[cont] = X[cont].astype('float32')


# In[23]:


#用于给模型打分的函数
def eval_mae(y_hat, xg_train): #第二个参数要和自定义的转换数据集名称一致（下方xg_train），第一个参数随意
    y_true = xg_train.get_label()
    return 'mae', mean_absolute_error(np.exp(y_true), np.exp(y_hat))


# In[24]:


#DMatrix是xgboost规则下的数据转换，为了提高xgboost上的提高效率，不转换也能跑
xg_train = xgb.DMatrix(X, y)


# ## Xgboost参数
# * **'booster'** : 'gbtree' -Which booster to use. Can be gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.
# * **'objective'** : 'multi:softmax' -Specify the learning task and the corresponding **learning objective**. (目标函数);'multi:softmax' -多分类问题损失函数.set XGBoost to do multiclass classification using the softmax objective, you also need to set **num_class**(number of classes).
# * **'num_class'** : 类别数，与multi:softmax并用.
# * **'gamma'** : Minimum loss reduction required to make a further partition on a leaf node of the tree. **range: [0,∞]**.The larger gamma is, the more conservative the algorithm will be.损失下降多少才进行分裂.
# * **'max_depth'** : 树的深度，越大越容易过拟合.
# * **'lambda'** : **L2 regularization term on weights.** Increasing this value will make model more conservative.(与课程中的lambda一样).
# * **'subsample'** : 随机采样训练样本.(行采样随机).
# * **'colsample_bytree'** : 属性随机选择.(列采样随机).
# * **'min_child_weight'** : Minimum sum of instance weight (hessian) needed in a child.The larger min_child_weight is, the more conservative the algorithm will be.(如果一个叶节点的样本权重和小于min_child_weight则停止分裂.
# * **'eta'** : Step size shrinkage used in update to prevents overfitting. **range: [0,1].**
# * **'nthread'** : 线程数.

# In[25]:


xgb_params = {
    'seed' : 0,  
    'eta' : 0.1,
    'colsample_bytree' : 0.5,    
    'subsample' : 0.5,
    #从reg:linear改为reg:squarederror对结果影响并不大，前者并不适用这个数据集
    'objective' : 'reg:squarederror',
    'max_depth' : 5, 
    'min_child_weight' : 3
}


# ## XGBoost.cv()参数
# * **'nfold'** : Number of folds in CV.k(n)-折交叉验证/k(n)-fold crossValidation
# * **'num_boost_round'** : Number of boosting iterations.
# * **'feval'** : Custom evaluation function.(定制评分函数)
# * **'maximize'** : Whether to maximize feval.
# * **'early_stopping_rounds'** : Activates early stopping. Cross-Validation metric (average of validation metric computed over CV folds) needs to improve at least once in every early_stopping_rounds round(s) to continue training. 

# In[26]:


get_ipython().run_line_magic('time', '')
bst_cv1 = xgb.cv(params = xgb_params, dtrain = xg_train, num_boost_round = 50, nfold = 3, seed = 0, 
                 feval = eval_mae, #feval引入打分函数
                 maximize = False, early_stopping_rounds = 10)
print('CV score:', bst_cv1.iloc[-1, :]['test-mae-mean'])


# In[27]:


plt.figure()
bst_cv1[['train-mae-mean', 'test-mae-mean']].plot()


# In[28]:


type(bst_cv1)


# <font color= 'red' size = 5>
#     Scikit-Learn API<br><br>
#     </font>
#     <font color='red' size = 3>
#      xgboost.XGBRegressor<br>
#     -----START-----
# </font>

# In[29]:


from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score


# In[30]:


xgb_params_2 = {
    'n_estimators' : 50, #like xgboost'num_boost_round
    #'learning_rate' : 0.1, #like xgboost'eta
    'random_state' : 0, #Random number seed
    'colsample_bytree' : 0.5,    
    'subsample' : 0.5,
    #从reg:linear改为reg:squarederror对结果影响并不大，前者并不适用这个数据集
    'objective' : 'reg:squarederror',
    'max_depth' : 5, 
    'min_child_weight' : 3
}


# In[31]:


X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.33, random_state = 0)


# In[32]:


clf = XGBRegressor(**xgb_params_2)


# In[33]:


clf.fit(X_train, y_train)


# In[34]:


y_pred = clf.predict(X_cv)


# In[35]:


#同上mae评分方法
mean_absolute_error(np.exp(y_cv), np.exp(y_pred))


# **特征重要性**

# In[36]:


#程序法计算importance，默认采用'gain'方法
feature_importance = clf.feature_importances_
feature_name = np.array(features)
important = pd.Series(feature_importance, index = feature_name)


# In[37]:


fig, ax = plt.subplots(1, 1, figsize = (16, 24))
#importance_type = { “weight”, “gain”,“cover”}
#”weight” is the number of times a feature appears in a tree
#”gain” is the average gain of splits which use the feature
#”cover” is the average coverage of splits which use the feature where coverage is defined as the number of samples affected by the split
plot_importance(clf, ax = ax, importance_type = 'gain')
plt.show()


# **参数的调优（网格搜索）**

# In[38]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold


# In[39]:


# GridSearchCV是一个自动调参器，调参方法为网格搜索法（给定多个参数的多种可能值，进行逐个笛卡尔积尝试-交叉验证，
# 评判标准采用scroe参数给定的评分函数-如下sklearn.metrics.SCORERS.keys()列表）
#learning_rate对应于XGBRegressor参数里的learning_rate也相当于'eta'
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
#copy模型，参数同上
clf2 = XGBRegressor(**xgb_params_2)
#要进行网格搜索的参数，用字典形式传入，key值为待调参模型的相应参数名称，待调参模型由GridSearchCV中estimator参数所指定
param_grid = {'learning_rate' : learning_rate}

kfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 0)
#cv是kfold的分割策略，回归模型指定整数，分类模型可用StratifiedKFold
#n_jobs是占用cpu数，-1为全部启用
grid_search = GridSearchCV(estimator = clf2, param_grid = param_grid, scoring = 'neg_mean_absolute_error', cv = 3, n_jobs = -1)
grid_search.fit(X_train, y_train)


# In[40]:


#返回最优参数
grid_search.best_params_


# In[41]:


#返回最优模型
grid_search.best_estimator_


# In[42]:


#评分方法列表
import sklearn
sklearn.metrics.SCORERS.keys()


# **fit()方法的提前终止**

# In[43]:


#延用前面其他参数和网格搜索优化的learning_rate值，创建模型
clf3 = XGBRegressor(learning_rate = 0.3, **xgb_params_2)
#启用early_stopping_rounds参数（如果10次eval_metric度量变化不大，就停止算法）
#eval_metric参数在API网页
eval_set = [(X_cv, y_cv)]
clf3.fit(X_train, y_train, early_stopping_rounds = 10, eval_metric = 'mae', eval_set = eval_set, verbose = True)
y_pred3 = clf3.predict(X_cv)
predictions = [round(x) for x in y_pred3]
accuracy = mean_absolute_error(y_cv, predictions)
print(accuracy)


# <font color= 'red' size = 5>
#     Scikit-Learn API<br><br>
#     </font>
#     <font color='red' size = 3>
#      xgboost.XGBRegressor<br>
#     -----END-----
# </font>

# In[ ]:




