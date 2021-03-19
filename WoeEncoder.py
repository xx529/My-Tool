import pandas as pd
import numpy as np

class WoeEncoder():
    def __init__(self, df_data, label_name, cat_features_name, bins={}):
        self.con_features_name = [x for x in df_data.columns if x != label_name and x not in cat_features_name]
        self.cat_features_name = cat_features_name
        self.label_name = label_name
        self.df_data = df_data
        self.bins = self.init_bins(bins)
        
    # 所有变量的 iv 值
    def iv_report(self, ascending=False):
        return pd.DataFrame(self.df_iv).sort_values('iv', ascending=ascending)
    
    
    # 每个变量分桶后的详细
    def get_bin_detial(self, col_name):
        return self.df_bin_record[col_name]
    
    
    # 每个样本的在这个变量上的 woe 编码
    def get_woe_value(self, col_name):
        result = pd.merge(self.df_bin_record_detail[col_name], self.df_bin_record[col_name][['bins', 'woe']], on='bins', how='left')
        result = pd.concat([self.df_data[col_name], result], axis=1)
        return result
    
    
    def fit(self):
        self.df_bin_record = {}
        self.df_bin_record_detail = {}
        self.df_iv = {'columns':[], 'iv': []}
        
        for (key, value) in self.bins.items():
            df_woe, iv, bins_detail = self.get_woe_and_iv(self.df_data, self.label_name ,key , value['strategy'], value['bins'])
            self.df_iv['columns'].append(key)
            self.df_iv['iv'].append(iv)
            self.df_bin_record[key] = df_woe
            self.df_bin_record_detail[key] = bins_detail

            
    def get_woe_and_iv(self, df, labels, col, strategy='distance', bins=None):

        # 数据分桶
        df = df[[col]+[labels]]

        if strategy == 'distance':
            df['bins'] = pd.cut(df[col], bins=bins)
        elif strategy == 'frequency':
            df['bins'] = pd.qcut(df[col], q=bins, duplicates='drop')

        woe_detail = df['bins'].value_counts().reset_index().rename(columns={'bins': 'total', 'index': 'bins'})

        # 计算每个属性 good(not event) 和 bad(event) 的个数
        temp = df.groupby('bins')[labels].value_counts().unstack().reset_index().rename(columns={0: 'good_total', 1: 'bad_total'})
        woe_detail = pd.merge(woe_detail, temp, on='bins', how='left')
        woe_detail.fillna(value={'good_total': 0, 'bad_total': 0}, inplace=True)

        # 计算每个属性 good 和 bad 的占比
        woe_detail['good_ratio'] = woe_detail['good_total'] / woe_detail['good_total'].sum()
        woe_detail['bad_ratio'] = woe_detail['bad_total'] / woe_detail['bad_total'].sum()

        # 计算 woe 值
        woe_detail['woe'] = np.log1p(woe_detail['bad_ratio'] / woe_detail['good_ratio'])
        woe_detail.sort_values('woe', inplace=True, ascending=False)

        # 计算 Odds 值
        c = woe_detail['bad_total'].sum() / woe_detail['good_total'].sum()
        woe_detail['odds'] = np.log1p(c) + woe_detail['woe']

        # 单调性检验
        woe_detail['Monotonicity'] = woe_detail['odds'].diff().apply(lambda x: True if x <= 0.0 else False)

        # 计算 IV 值
        iv_score = ((woe_detail['bad_ratio'] - woe_detail['good_ratio']) * woe_detail['woe']).sum()

        return woe_detail, iv_score, df['bins']
    
    
    def init_bins(self, bins):
        con_dict = {x: {'strategy': 'distance', 'bins': [-float('inf')] + [num for num in range(self.df_data[x].nunique())]} if x not in bins else bins[x] for x in self.cat_features_name }
        cat_dict = {x: {'strategy': 'frequency', 'bins': [0, 0.2, 0.4, 0.6, 0.8, 1]} if x not in bins else bins[x] for x in self.con_features_name}
        return {**con_dict, **cat_dict}
    

bins = {
    'col_name': {
        'strategy': , 
         'bins': 
    }
}

woe = WoeEncoder(df_data, label_name, cat_features_name, bins=bins)
woe.fit()
