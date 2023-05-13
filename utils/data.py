import copy
import logging
import multiprocessing as mp
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Union, Callable, Sequence

import pandas as pd
from tqdm import tqdm

from .imgtf import *

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def hold_out(cls_cnt: pd.DataFrame, scale: float, seed=0):
    ''' cls_cnt: Dataframe[classes, img_id]
        scale: 各类别分布在训练集中的比例
        return: 训练集 id 列表, 验证集 id 列表'''
    cls_cnt = cls_cnt.copy(deep=True)
    dtype = np.int64 if 'int' in str(next(iter(cls_cnt.dtypes))) else np.float64
    radio = scale / (1 - scale)
    # 打乱图像的次序
    idx = cls_cnt.index.values
    np.random.seed(seed)
    np.random.shuffle(idx)
    # 记录训练集、验证集当前各类别数量
    data_cnt = np.zeros([2, len(cls_cnt.columns)], dtype=np.float64)
    data_cnt[1] += 1e-4
    # 存放训练集、验证集数据的 id
    data_pool = [[] for _ in range(2)]
    pbar = tqdm(idx)
    # 留出法: 计算期望比例 与 执行动作后比例的 SSE 损失
    loss_func = lambda x: np.square(x[0] / x[1] - radio).sum()
    for i in pbar:
        cnt = cls_cnt.loc[i]
        loss = np.zeros(2, dtype=np.float64)
        for j, next_sit in enumerate([data_cnt.copy() for _ in range(2)]):
            next_sit[j] += cnt
            loss[j] = loss_func(next_sit)
        # 根据损失值选择加入的数据集
        choose = loss.argmin()
        data_cnt[choose] += cnt
        data_pool[choose].append(i)
        # 输出当前的分割情况
        cur_scale = data_cnt[0] / data_cnt.sum(axis=0) - scale
        pbar.set_description(f'Category scale error ∈ [{cur_scale.min():.3f}, {cur_scale.max():.3f}]')
    # 输出训练集、验证集信息
    data_cnt = data_cnt.round(3).astype(dtype)
    LOGGER.info(f'Train Set ({len(data_pool[0])}): {data_cnt[0]}')
    LOGGER.info(f'Eval Set ({len(data_pool[1])}): {data_cnt[1]}')
    return data_pool


def undersampling(cls_cnt: pd.DataFrame, n, seed=0):
    ''' cls_cnt: Dataframe[classes, img_id]
        n: 各类别实例的采样数量 (int, float, list, tuple)
        return: 训练集 id 列表, 验证集 id 列表'''
    cls_cnt = cls_cnt.copy(deep=True)
    dtype = np.int64 if 'int' in str(next(iter(cls_cnt.dtypes))) else np.float64
    np.random.seed(seed)
    cls_cnt_backup = cls_cnt
    n_cls = len(cls_cnt.columns)
    # 对参数 n 进行修改 / 校验
    if not hasattr(n, '__len__'): n = [n] * n_cls
    assert len(n) == n_cls, 'The parameter n does not match the number of categories'
    # 筛选出无标签数据
    g = dict(list(cls_cnt.groupby(cls_cnt.sum(axis=1) == 0, sort=False)))
    unlabeled, cls_cnt = map(lambda k: g.get(k, pd.DataFrame()), (True, False))
    unlabeled = list(unlabeled.index)
    np.random.shuffle(unlabeled)
    # 存放训练集、验证集数据的 id
    m = len(unlabeled) // 2
    data_pool = [unlabeled[:m], unlabeled[m:]]
    data_cnt = np.zeros(n_cls, dtype=np.float64)
    while not cls_cnt.empty:
        # 取出当前 cls_cnt 最少的类
        j = cls_cnt.sum().apply(lambda x: np.inf if x == 0 else x).argmin()
        g = dict(list(cls_cnt.groupby(cls_cnt[j] > 0, sort=False)))
        # 对阳性样本进行划分, 放回阴性样本
        posi, cls_cnt = map(lambda k: g.get(k, pd.DataFrame()), (True, False))
        m, idx = -1, list(posi.index)
        if not posi.empty:
            lack = n[j] - data_cnt[j]
            if lack > 0:
                # 选取前 m 个加入训练集
                np.random.shuffle(idx)
                posi = posi.loc[idx]
                cumsum = np.cumsum(posi[j].to_numpy())
                m = np.abs(cumsum - lack).argmin() + 1
                # 考虑极端情况下, 不加入更好
                if m == 1 and cumsum[0] > lack: m = 0
                data_pool[0] += idx[:m]
                data_cnt += posi.iloc[:m].sum()
        # 其余放入验证集
        data_pool[1] += idx[m:]
    # 输出训练集、验证集信息
    data_cnt = data_cnt.to_numpy()
    LOGGER.info(f'Train Set ({len(data_pool[0])}): {data_cnt.round(3).astype(dtype)}')
    eval_cnt = cls_cnt_backup.sum().to_numpy() - data_cnt
    LOGGER.info(f'Eval Set ({len(data_pool[1])}): {eval_cnt.round(3).astype(dtype)}')
    return data_pool


class ImagePool:
    img_size = property(fget=lambda self: getattr(self.images, 'shape', (0,))[1: 3])

    def __init__(self, files, labels):
        self.files = files
        self.labels = labels
        self.images = None

    def select(self, indexes):
        self.files = self.files[indexes]
        self.labels = self.labels[indexes]
        if hasattr(self.images, '__len__'):
            self.images = self.images[indexes]

    def loadimg(self,
                img_size: Union[int, tuple],
                loader: Callable[[Path, tuple], np.ndarray] = load_image):
        img_size = (img_size,) * 2 if isinstance(img_size, int) else img_size
        if self.img_size != img_size:
            LOGGER.info(f'\nChange the image size to {img_size}')
            # loader(file, img_size): 图像加载函数
            loader = partial(loader, img_size=img_size)
            # 启动多线程读取图像
            qbar = tqdm(ThreadPool(mp.cpu_count()).imap(loader, self.files),
                        total=len(self.files), desc='Loading images')
            self.images = np.stack(tuple(qbar))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]


class Dataset(torch.utils.data.Dataset):
    ''' indexes: 数据集的 ID 列表'''

    def __init__(self,
                 imgpool: ImagePool,
                 indexes: Sequence):
        super().__init__()
        self.augment = True
        self.imgpool = imgpool
        self.indexes, self.tf = indexes, None

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        img, label = self.imgpool[self.indexes[item]]
        if self.augment and self.tf: img = self.tf(img)
        return to_tensor(img), label

    def __add__(self, other):
        _copy = copy.deepcopy(self)
        _copy.indexes += other.indexes
        return _copy

    def get_tf(self, hyp):
        self.tf = Transform(hyp) if hyp else None
        return self.tf


if __name__ == '__main__':
    np.random.seed(0)

    # 5000 张图像各个类别边界框数目统计结果
    example = (np.random.random([100, 3]) * 3).astype(np.int32)
    example = pd.DataFrame(example, index=[f'train_{i}.jpg' for i in range(example.shape[0])])
    example *= np.array([4, 1, 9])

    for i in range(2):
        # train_id, eval_id = hold_out(example, 0.8)
        train_id, eval_id = undersampling(example, 50)
        print(example.loc[train_id].sum())
