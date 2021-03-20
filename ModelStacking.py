import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import namedtuple

class ModelStacking:
    def __init__(self):
        self.model_info_list = []
        self.ModelInfo = namedtuple('KFoldModel', ['name', 'models', 'val_fold_index'])
        self.train_model = LinearRegression()
        self.is_transform = False


    # 增加集成的 model 
    def add_model(self, name, models, k_fold):
        val_fold_index = [x[1] for x in k_fold]
        self.model_info_list.append(self.ModelInfo(name, models, val_fold_index))


    # 把原来训练和预测的特征转换成 stacking feature
    def transform(self, train_x, test_x):
        df_train_feature_list, df_predict_feature_list = [], []
        
        for cls_model_info in self.model_info_list:
            df_train_feature, df_test_feature = self.creat_features(cls_model_info, train_x, test_x)
            
            df_train_feature_list.append(df_train_feature)
            df_predict_feature_list.append(df_test_feature)

        self.train_meta_features = pd.concat(df_train_feature_list, axis=1)
        self.predict_meta_features = pd.concat(df_predict_feature_list, axis=1)

        self.is_transform = True
        return self.train_meta_features, self.predict_meta_features


    # 拟合模型（默认LR模型与参数）
    def fit(self, train_x, test_x, Y):
        None if self.is_transform else self.transform(train_x, test_x)
        self.train_model.fit(self.train_meta_features, Y)
        return self.train_model
    

    # 模型预测
    def predict(self):
        result = self.train_model.predict(self.predict_meta_features)
        return result
    

    # 转换一类 model 的训练和预测 stacking feature
    def creat_features(self, cls_model_info, train_x, test_x):
        train_feature_array, predict_feature_array = np.zeros(shape=train_x.shape[0]), np.zeros(shape=test_x.shape[0])

        for model, index in zip(cls_model_info.models, cls_model_info.val_fold_index):
            train_feature_array[index] = model.predict(train_x.iloc[index, :])
            predict_feature_array += model.predict(test_x) / len(cls_model_info.models)

        df_train = pd.DataFrame(train_feature_array, columns=['train_' + cls_model_info.name])
        df_test = pd.DataFrame(predict_feature_array, columns=['predict_' + cls_model_info.name])

        return df_train, df_test

    
    # 设置 stacking 训练用的模型，默认LR
    def set_train_model(self, train_model):
        self.train_model = train_model

# ---------------------------
# 快捷使用
stack = ModelStacking()
stack.add_model('lgb_1', [model]*5, k_fold_1)
stack.add_model('lgb_2', [model]*5, k_fold_2)
stack.fit(df_train_raw, df_test_raw, df_label)
stack.predict()

# 自定义模型
clssifier = LogisticRegression()
stack.set_train_model(clssifier)

# 仅获取 stack 训练和预测用的特征
stack = ModelStacking()
stack.add_model('lgb_1', [model]*5, k_fold_1)
train_meta_features, test_meta_features = stack.transform(train_x, test_x)
