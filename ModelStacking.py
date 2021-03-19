import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import namedtuple


class ModelStacking:
    def __init__(self):
        self.model_info_list = []
        self.ModelInfo = namedtuple('KFoldModel', ['name', 'models', 'val_fold_index'])
        self.train_model = LinearRegression()

    # 增加集成的 model 
    def add_model(self, name, models, k_fold):
        val_fold_index = [x[1] for x in k_fold]
        self.model_info_list.append(self.ModelInfo(name, models, val_fold_index))

    # 拟合模型（默认LR模型与参数）
    def fit(self, train_x, Y):
        self.creat_meta_features(train_x)
        self.train_model.fit(self.train_meta_features, Y)
        return self.train_model
    
    # 模型预测
    def predict(self, test_x):
        self.creat_predict_feature(test_x)
        result = self.train_model.predict(self.predict_meta_features)
        return result

    # 获取所有 model 的训练用 new feature
    def creat_meta_features(self, train_x):
        df_new_feature_list = []

        for cls_model_info in self.model_info_list:
            df_new_feature = self.creat_train_new_feature(cls_model_info, train_x)
            df_new_feature_list.append(df_new_feature)
        
        self.train_meta_features = pd.concat(df_new_feature_list, axis=1)
        return self.train_meta_features

    # 创建一个 model 的训练用 new feature
    def creat_train_new_feature(self, cls_model_info, train_x):
        new_feature_array = np.zeros(shape=train_x.shape[0])

        for model, index in zip(cls_model_info.models, cls_model_info.val_fold_index):
            new_feature_predict = model.predict(train_x.iloc[index, :])
            new_feature_array[index] = new_feature_predict

        return pd.DataFrame(new_feature_array, columns=['train_' + cls_model_info.name])
    
    
    # 获取所有 model 的预测用 predict feature
    def creat_predict_feature(self, test_x):
        df_predict_feature_list = []

        for cls_model_info in self.model_info_list:
            df_predict_feature = self.creat_predict_new_feature(cls_model_info, test_x)
            df_predict_feature_list.append(df_predict_feature)

        self.predict_meta_features = pd.concat(df_predict_feature_list, axis=1)
        return self.predict_meta_features

      
    # 创建一个 model 的训练用 new feature
    def creat_predict_new_feature(self, cls_model_info, test_x):
        predict_feature_array = np.zeros(shape=test_x.shape[0])

        for model in cls_model_info.models:
            predict_feature_array += model.predict(test_x)

        predict_feature_array = predict_feature_array / len(cls_model_info.models)
        return pd.DataFrame(predict_feature_array, columns=['predict_' + cls_model_info.name])
    

    # 设置 stacking 训练用的模型，默认LR
    def set_train_model(self, train_model):
        self.train_model = train_model



stack = ModelStacking()
stack.add_model('lgb_1', [model]*5, k_fold_1)
stack.add_model('lgb_2', [model]*5, k_fold_1)

# stack.set_train_model(LogisticRegression())

stack.fit(df_train_raw, df_label)
stack.predict(df_test_raw)
