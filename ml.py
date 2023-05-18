import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data import *
from engine.evolve import HyperEvolve
from utils.plot import *

COLORS = blue, purple, orange, green
SEED = 20181003
np.set_printoptions(4, suppress=True)

standard = StandardScaler().fit_transform


def load_dataset(batch_size):
    ''' 加载数据集'''
    from tqdm import tqdm
    from engine.trainer import select_device

    device = select_device('', batch_size=batch_size)
    model, *_ = load_model(cfg=VOVNET / 'final.yaml',
                           weight=VOVNET / 'final.pt')
    model.simplify(True)
    # 读取数据集, 创建缓存区
    DATAPOOL.loadimg(model.cfg['img_size'])
    x, y = [], []
    # 前向传播得到低维特征
    with torch.no_grad():
        for img, tar in tqdm(torch.utils.data.DataLoader(ALL_DATA, batch_size=batch_size, shuffle=False),
                             desc='Inference'):
            img = img.to(device)
            x.append(model(img, tarlayer=-3)[..., 0, 0].cpu())
            y.append(tar)
    return tuple(map(lambda i: np.concatenate(i, axis=0), (x, y)))


x, y = (CFG / 'dataset.pkf').lazy_obj(load_dataset, batch_size=24)
x = standard(x)
print(f'x.shape: {x.shape}')
print(f'y.shape: {y.shape}')
# 以留出法分割数据集, 与训练网络时保持一致
train_id, eval_id = HOLD_OUT[0].indexes, HOLD_OUT[1].indexes
train_x, train_y, eval_x, eval_y = x[train_id], y[train_id], x[eval_id], y[eval_id]


class DataAnalysis:
    ''' 数据集的分析工具箱'''

    @staticmethod
    def distribute():
        fig = plt.subplot()
        COLOR = (blue, orange, pink)
        data_cnt = np.array([[438, 122, 197, 203, 779, 60, 121],
                             [109, 31, 49, 51, 195, 15, 30]], dtype=np.float32)
        total = data_cnt.sum(axis=0, keepdims=True)
        data_cnt = np.concatenate((total, data_cnt), axis=0)
        data_cnt /= np.sqrt(total)
        # 绘制柱状图
        bar2d(data_cnt, xticks=[f'DR-{i}' for i in range(4)] + [f'DME-{i}' for i in range(3)],
              labels=['total', 'train-set', 'eval-set'], colors=COLOR)
        for key in 'right', 'top', 'left', 'bottom':
            fig.spines[key].set_color('None')
        plt.yticks([], [])
        plt.show()

    @staticmethod
    def mean_bar():
        ''' 为每个类别的特征绘制均值柱状图'''
        for i, n in enumerate((DR, DME)):
            fig = plt.subplot(2, 1, i + 1)
            for key in 'right', 'top', 'bottom', 'left':
                fig.spines[key].set_color('None')
            for j in 'xy': getattr(plt, f'{j}ticks')([], [])
            plt.xlim(0, x.shape[1] * 1.17)
            # 搜集各个类别各个特征的均值
            means = np.stack([x[y[:, i] == j].mean(axis=0) for j in range(n)], axis=-1)
            means = means[np.argsort(means.mean(axis=-1) * np.abs(means).mean(axis=-1))]
            # 绘制各个类别各个特征的均值
            x_ = np.arange(means.shape[0])
            for j in range(n):
                plt.bar(x_, means[:, j], label=j, alpha=.3, width=1., color=COLORS[j])
            plt.legend(frameon=False)
            # 计算余弦相似度
            means = torch.from_numpy(means).T
            print(np.rad2deg(torch.acos(torch.cosine_similarity(means, means[:, None], dim=-1)).data.numpy()))
        plt.show()


class Toolkit:
    ''' 经典机器学习方法的验证工具箱'''
    __clfs__ = []

    @classmethod
    def register(tool, clf):
        ''' 注册分类器, 以进行一次性验证'''
        tool.__clfs__.append(clf)
        return clf

    @staticmethod
    def kfold(clf, cv=5):
        ''' 使用 K 折交叉验证评估方法'''
        np.random.seed(SEED)
        preds = np.stack(
            (cross_val_predict(clf, x, y[:, 0], cv=cv),
             cross_val_predict(clf, x, y[:, 1], cv=cv)), axis=-1
        )
        return get_metrics(preds, y)

    @staticmethod
    def hout(clf):
        ''' 使用与 VoVNet 相同的数据分割方法'''
        np.random.seed(SEED)
        preds = np.stack(
            (clf.fit(train_x, train_y[:, 0]).predict(eval_x),
             clf.fit(train_x, train_y[:, 1]).predict(eval_x)), axis=-1
        )
        return get_metrics(preds, eval_y)

    @classmethod
    def evolution(tool, cls, epochs=100):
        ''' 超参数进化算法'''
        fitn = lambda hyp, epoch: get_fitness(tool.kfold(cls(hyp)))
        he = HyperEvolve(project=RUN / cls.__name__, hyp=cls.hyp, patience=epochs / 4)
        he(fitn, epochs=cls.epochs, mutation=cls.mutation)
        print('kfold:', tool.kfold(cls()))
        he.plot()

    @classmethod
    def test_all(tool, baseline=False):
        ''' 测试所有已注册的分类器, 保存结果到 run/ml.csv'''
        result = {}
        for clf in tool.__clfs__:
            result[clf.__name__] = tool.kfold(clf({} if baseline else None))
        result = pd.DataFrame(result, index=['DR P-', 'DR AP+', 'DME P-', 'DME AP+',
                                             'DR K', 'DME K', 'DR Acc', 'DME Acc']).T
        result *= 100
        print(result)
        (RUN / 'ml.csv').csv(result)


@Toolkit.register
class DecTreeClf:
    ''' 决策树'''
    mutation = 1.
    epochs = 70
    hyp = ML / 'tree/hyp.yaml'

    @staticmethod
    def get_kwd(hyp):
        mss = max((int(hyp.get('mss_uint', 2)), 2))
        msf = max((int(hyp.get('msf_uint', 2)), 1))
        mdepth = max((int(hyp.get('mdep_uint', 100)), 2))
        return dict(min_samples_split=mss, min_samples_leaf=msf, max_depth=mdepth)

    def __new__(cls, hyp=None):
        if hyp is None: hyp = cls.hyp
        if not isinstance(hyp, dict): hyp = hyp.yaml()
        return DecisionTreeClassifier(**cls.get_kwd(hyp))


@Toolkit.register
class ForestClf(DecTreeClf):
    ''' 随机森林'''
    hyp = ML / 'tree/hyp-bag.yaml'

    @staticmethod
    def get_enskwd(hyp):
        return dict(n_estimators=max((int(hyp.get('n_uint', 2)), 2)))

    def __new__(cls, hyp=None):
        if hyp is None: hyp = cls.hyp
        if not isinstance(hyp, dict): hyp = hyp.yaml()
        return RandomForestClassifier(**cls.get_enskwd(hyp), **cls.get_kwd(hyp))


@Toolkit.register
class BoostTreeClf(ForestClf):
    ''' AdaBoost + 决策树'''
    hyp = ML / 'tree/hyp-boost.yaml'

    def __new__(cls, hyp=None):
        if hyp is None: hyp = cls.hyp
        if not isinstance(hyp, dict): hyp = hyp.yaml()
        baseclf = DecisionTreeClassifier(**cls.get_kwd(hyp))
        return AdaBoostClassifier(baseclf, learning_rate=max((hyp.get('lr_float', 1), 1e-8)), **cls.get_enskwd(hyp))


@Toolkit.register
class SVMClf:
    ''' 支持向量机'''
    mutation = 1.
    epochs = 50
    hyp = ML / 'svm/hyp.yaml'

    @staticmethod
    def get_kwd(hyp):
        return dict(C=max((hyp.get('c_float', 1.), 1e-8)),
                    gamma=hyp.get('gamma_float', 7e-3))

    def __new__(cls, hyp=None):
        if hyp is None: hyp = cls.hyp
        if not isinstance(hyp, dict): hyp = hyp.yaml()
        return SVC(**cls.get_kwd(hyp))


@Toolkit.register
class BagSVMClf(SVMClf):
    ''' AdaBoost + 支持向量机'''
    hyp = ML / 'svm/hyp-bag.yaml'

    def __new__(cls, hyp=None):
        if hyp is None: hyp = cls.hyp
        if not isinstance(hyp, dict): hyp = hyp.yaml()
        baseclf = SVC(**cls.get_kwd(hyp))
        n = max((int(hyp.get('n_uint', 3)), 2))
        return BaggingClassifier(baseclf, n_estimators=n)


# Toolkit.evolution(BoostTreeClf)
Toolkit.test_all(baseline=False)

