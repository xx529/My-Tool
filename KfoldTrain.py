import lightgbm as lgb
from sklearn.model_selection import KFold
from itertools import product
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


class KfoldTrain:
    def __init__(self, train_x, label, search_param, feature_name=None, cat_feature_name=None, cv=5):
        self.k_fold = list(KFold(n_splits=5, random_state=529, shuffle=True).split(train_x))
        self.search_param = search_param
        self.train_x = train_x
        self.label = label
        self.feature_name = feature_name
        self.cat_feature_name = cat_feature_name
        self.output_models = []

    def train(self):
        for idx, (train_idx, val_idx) in enumerate(self.k_fold):
            print('{}th fold trainning...'.format(idx+1, end='  '))
            train_feture_df = self.train_x.iloc[train_idx, :]
            train_label_df = self.label.iloc[train_idx, :]
            val_feature_df = self.train_x.iloc[val_idx, :]
            val_label_df = self.label.iloc[val_idx, :]
            
            current_fold_model = self.find_best_model_in_one_fold(train_feture_df, val_feature_df, train_label_df, val_label_df)
            self.output_models.append(current_fold_model)

        return self.output_models

    def predict(self, Y):
        pre = np.zeros(shape=Y.shape[0])
    
        for model in self.output_models:
            pre += model.predict(Y)
    
        result = pre / len(models)
        return result

    # 每一个 fold 寻找最好的 model 与参数
    def find_best_model_in_one_fold(self, train_feture_df, val_feature_df, train_label_df, val_label_df):
        param_list = [{key: value for key, value in zip(self.search_param.keys(), i)} for i in product(*self.search_param.values())]

        lgb_train_data = lgb.Dataset(data=train_feture_df, label=train_label_df, feature_name=self.feature_name, categorical_feature=self.cat_feature_name)
        lgb_val_data = lgb.Dataset(data=val_feature_df, label=val_label_df, feature_name=self.feature_name, categorical_feature=self.cat_feature_name)
        
        best_model, best_param, best_score = None, None, 0.0
        for param in param_list:
            model = lgb.train(param, lgb_train_data, num_boost_round=2000, feature_name=self.feature_name, categorical_feature=self.cat_feature_name, valid_sets=[lgb_val_data], verbose_eval=False)
            # pre = (model.predict(val_feature_df) >= 0.5).astype(int)
            # acc_score = accuracy_score(pre, val_label_df)

            pre = model.predict(val_feature_df)
            current_score = mean_squared_error(pre, val_label_df)
            
            if current_score > best_score:
                best_score = current_score
                best_model = model
                best_param = param
                
        print('val_set best score : {}'.format(best_score))
        return best_model



## -------------

param = {
    'boosting': ['gbdt'],
    'objective' :['regression'],
    'max_depth' : [20],
    'learning_rate' : [0.01],
    'bagging_fraction': [1], #样本采样比例
    'bagging_freq': [8], #bagging的次数
}

lgb_k_fold_train = KfoldTrain(df_train_raw, df_label, param)
lgb_k_fold_train.train()
lgb_k_fold_train.predict(df_test_raw)
