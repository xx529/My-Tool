import pandas as pd
import numpy as np


class Optimizer:
    """
    This class optimise an algorithm/model configuration with respect to a given score.
    """
    def __init__(self,
                 algo_score,
                 max_iter,
                 max_intensification,
                 model,
                 cs):
        """
        :param algo_score: is the function called to evaluate algorithm / model score
        :param max_iter: the maximal number of training to perform
        :param max_intensification: the maximal number of candidates configuration to sample randomly
        :param model: the class of the internal model used as score estimator.
        :param cs: the configuration space to explore
        """
        self.traj = []
        self.algo_score = algo_score # 打分模型
        self.max_iter = max_iter # 迭代次数，停止条件可以按需求更改
        self.max_intensification = max_intensification # 候选参数组合随机的个数
        self.internal_model = model() # 评估参数模型
        self.trajectory = [] # 记录每次优化后的参数组合
        self.cfgs = []
        self.scores = {}
        self.best_cfg = None
        self.best_score = None
        self.cs = cs

    def cfg_to_dtf(self, cfgs):
        """
        Convert configs list into pandas DataFrame to ease learning
        """
        cfgs = [dict(cfg) for cfg in cfgs]
        dtf = pd.DataFrame(cfgs)
        return dtf


    def optimize(self):
        """
        Optimize algo/model using internal score estimator
        """
        cfg = self.cs.sample_configuration()
        self.cfgs.append(cfg)
        self.trajectory.append(cfg)
        
        # initial run
        score = self.algo_score(cfg)
        self.scores[cfg] = score
        self.best_cfg = cfg
        self.best_score = score

        dtf = self.cfg_to_dtf(self.cfgs)

        for i in range(0, self.max_iter):
            # We need at least two datapoints for training
            # 至少2个数据才能训练调参模型
            if dtf.shape[0] > 1:
                scores = np.array([ val for key, val in self.scores.items()])
                self.internal_model.fit(dtf, scores)
                
                # intensification
                candidates = [self.cs.sample_configuration() for _ in range(self.max_intensification)]
                candidates = [x for x in candidates if x not in self.cfgs]  # 防止重复
                candidate_scores = [self.internal_model.predict(self.cfg_to_dtf([cfg])) for cfg in candidates]
                best_candidates = np.argmax(candidate_scores)

                cfg = candidates[best_candidates]
                self.cfgs.append(cfg)
                score = self.algo_score(cfg)
                self.scores[cfg] = score

                if score > self.best_score:
                    self.best_cfg = cfg
                    self.best_score = score
                    self.trajectory.append(cfg)

                dtf = self.cfg_to_dtf(self.cfgs)
                self.internal_model.fit(dtf, np.array([val for kay, val in self.scores.items()]))
            else:
                cfg = self.cs.sample_configuration()
                self.cfgs.append(cfg)
                score = self.algo_score(cfg)
                self.scores[cfg] = score

                if score > self.best_score:
                    self.best_cfg = cfg
                    self.best_score = score
                    self.trajectory.append(cfg)
                dtf = self.cfg_to_dtf(self.cfgs)
