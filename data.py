import cv2
import pandas as pd

from engine.crosstab import Crosstab
from model.ema import EmaModel
from model.fourier import make_grid
from model.model import *
from utils.data import Dataset, hold_out, ImagePool
from utils.utils import Path

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

DR, DME = 4, 3
INDEXS = ['Image name', 'Retinopathy grade', 'Risk of macular edema ']

CFG = Path('config')
RESNET = CFG / 'resnet'
VOVNET = CFG / 'vovnet'
ELAN = CFG / 'elan'
ML = CFG / 'ml'
RUN = Path('run')


def read_data(root=Path('dataset')):
    ''' 读取并划分数据集, 可选择是否读取图像'''
    data_set = []
    for folder in root.iterdir():
        # 读取标签文件
        xls = next(filter(lambda f: f.suffix == '.xls', folder.iterdir()))
        assert xls, 'The label file has been lost'
        xls = pd.read_excel(xls)
        # 修正图像路径
        xls['Image name'] = xls['Image name'].apply(lambda f: folder / f)
        exist = xls['Image name'].apply(lambda f: f.exists())
        data_set.append(xls[exist][INDEXS])
    data_set = pd.concat(data_set).reset_index(drop=True).to_numpy().tolist()
    # 表示成具有边界框的目标检测数据集
    cls_cnt = np.zeros([len(data_set), DR + DME], dtype=np.int32)
    for i in range(len(data_set)):
        file, *label = data_set[i]
        cls_cnt[i, label[0]] = cls_cnt[i, DR + label[1]] = 1
        # 对标签信息进行整理
        data_set[i] = file, torch.tensor(label, dtype=torch.int64)
    return pd.DataFrame(cls_cnt), np.array(data_set, dtype=object)


class ImgReader(ImagePool):

    def loadone(self, file_or_bgr, img_size):
        # 如果是文件, 则读取
        if isinstance(file_or_bgr, Path):
            file_or_bgr = cv2.imread(str(file_or_bgr))
        # 提取最小矩形框, 对图像进行裁剪
        gray = cv2.cvtColor(file_or_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, thresh=10, maxval=255, type=cv2.THRESH_TOZERO)[1]
        gray = cv2.erode(gray, np.ones([5, 5]), iterations=1)
        l, t, w, h = cv2.boundingRect(gray)
        file_or_bgr = file_or_bgr[t: t + h, l: l + w]
        # cv2.imshow('debug', file_or_bgr)
        # cv2.waitKey(0)
        h, w, _ = file_or_bgr.shape
        assert 0.95 < h / w < 1.05, 'Failed to extract minimum bounding rect'
        return cv2.resize(file_or_bgr, img_size)

    def loadimg(self, img_size, loader=None):
        super().loadimg(img_size, loader=self.loadone)


try:
    # 生成标签统计, 数据池
    CLS_CNT, data_set = read_data()
    DATAPOOL = ImgReader(files=data_set[:, 0], labels=data_set[:, 1])
except Exception as error:
    LOGGER.warning(error)
    CLS_CNT, DATAPOOL = None, None


class Dataset(Dataset):

    def __init__(self, indexes):
        super().__init__(imgpool=DATAPOOL, indexes=indexes)


def get_cost(model):
    img_size = model.cfg['img_size']
    x = torch.rand([1, 3, img_size, img_size]).cuda()
    model(x)
    traced = model.cuda().torchscript(x)
    return traced.profile(x)


def get_metrics(preds, target):
    cr1 = Crosstab(preds[:, 0], target[:, 0], num_classes=DR)
    cr2 = Crosstab(preds[:, 1], target[:, 1], num_classes=DME)
    return np.array([cr1.precision[0], cr1.precision[1:].mean(),
                     cr2.precision[0], cr2.precision[1:].mean(),
                     cr1.kappa, cr2.kappa, cr1.accuracy, cr2.accuracy]).round(4)


def get_fitness(metrics):
    return (.4 * metrics.mean() + .8 * metrics[-3:-1].mean()).item()


# 留出法: 用于 CNN / ViT 训练
HOLD_OUT = Path('config/hold-out.pkf')
HOLD_OUT = tuple(map(Dataset, HOLD_OUT.lazy_obj(hold_out, CLS_CNT, scale=.8, seed=1)))
# 全部数据: 用于对比学习
# ALL_DATA = Dataset(list(CLS_CNT.index))


def data_loader(hyp, train_set, eval_set=None):
    Loader = torch.utils.data.DataLoader
    kwargs = dict(batch_size=hyp['batch_size'], shuffle=True)
    # 训练集
    train_set.get_tf(hyp)
    train_set = Loader(train_set, drop_last=False, **kwargs)
    # 验证集
    if eval_set:
        eval_set = Loader(eval_set, drop_last=False, **kwargs)
    return train_set, eval_set


def get_mask(n_patch):
    grid = make_grid(n_patch, n_patch, center=True)
    grid -= .5
    grid = grid[..., 0] + 1j * grid[..., 1]
    return np.abs(grid).flatten() < .5


def load_model(cfg, weight, hyp={}, freeze=None, ema=False):
    ''' return: best fitness'''
    if not isinstance(hyp, dict): hyp = hyp.yaml()
    if not isinstance(cfg, dict): cfg = cfg.yaml()
    # 设置 DropBlock 参数
    drop = cfg.get('drop_pos', None)
    if drop:
        cfg = cfg_modify(cfg, sum([
            [(index, 'a0', hyp.get(f'd{i}_kernel', 1)),
             (index, 'a1', hyp.get(f'd{i}_proba', 0))] for i, index in enumerate(drop)
        ], []))
        DropBlock.start_epoch = 0
    # 初始化模型
    model = YamlModel(cfg).cuda()
    if freeze:
        for i in range(freeze): model.freeze(i)
    # EMA
    ema = EmaModel(model, bp_times=3000) if ema else None
    # 如果使用了 ViT, 设置掩码
    if weight: model.load_state_dict(torch.load(weight)['model'], strict=False)
    LOGGER.info(f'Inference latency: {get_cost(model):.2f} ms')
    return (model, ema) if ema else model, cfg, hyp


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    patch_size = 32
    img_size = 608
    n_patch = img_size // patch_size

    # 获取图像和掩膜
    img = DATAPOOL.loadone(np.random.choice(DATAPOOL.files), (img_size,) * 2)
    mask = get_mask(n_patch).reshape(*(n_patch,) * 2)
    print(mask.sum())
    mask = mask.repeat(patch_size, 0).repeat(patch_size, 1)
    mask = np.uint8(mask[..., None] * (240, 176, 0))

    # 绘制网格线
    kwd = dict(color=(148, 20, 7), thickness=1)
    for x in np.arange(0, img_size, patch_size)[1:]:
        mask = cv2.line(mask, (0, x), (img_size, x), **kwd)
        mask = cv2.line(mask, (x, 0), (x, img_size), **kwd)

    img = cv2.addWeighted(img, .7, mask, .3, 0)
    cv2.imshow('sd', img)
    cv2.waitKey(0)
