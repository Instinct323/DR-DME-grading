from data import *
from deploy.onnx_run import OnnxModel
from engine.loss import *
from engine.trainer import Trainer


class Classifier(Trainer):

    def __init__(self, model, project, hyp, ema: EmaModel = None):
        m_title = ['DR P-', 'DR AP+', 'DME P-', 'DME AP+', 'DR K', 'DME K', 'DR Acc', 'DME Acc']
        super().__init__(model, project, m_title=m_title, hyp=hyp)
        self.fce = CrossEntropy(l2penalty=0), CrossEntropy(l2penalty=0)
        self.ema = ema

    def loss(self, image, target):
        image = image.to(self.device)
        target = target.to(self.device)
        logits = self.model(image)
        pred = logits.split([DR, DME], dim=-1)
        loss = sum(self.fce[i](pred[i], target[:, i]).mean() for i in range(2))
        return loss + (self.ema.mse(image, logits) if self.ema else 0)

    def metrics(self, generator):
        preds, targets = [], []
        for img, tar in generator:
            pred = self.model(img.to(self.device)).cpu()
            preds.append(torch.stack(list(map(lambda x: x.argmax(dim=-1),
                                              pred.split([DR, DME], dim=-1))), dim=-1))
            targets.append(tar)
        return get_metrics(*map(np.concatenate, (preds, targets)))

    def fitness(self, metrics):
        return get_fitness(metrics)


def train_once(cfg, weight, project, hyp, dataset,
               freeze=None, patch_size=None):
    ''' return: best fitness'''
    (model, ema), cfg, hyp = load_model(cfg, weight, hyp, freeze=freeze, ema=True)
    img_size = cfg['img_size']
    # 如果使用了 ViT, 设置掩码
    if patch_size:
        LOCAL.pmask = torch.from_numpy(get_mask(patch_size))
    # 读取数据集, 创建训练器, 开始训练
    DATAPOOL.loadimg(img_size)
    tn = Classifier(model, project, hyp=hyp, ema=ema)
    fit = tn(*data_loader(hyp, *dataset))
    # 导出 onnx 模型
    x = torch.randn([1, 3, img_size, img_size]).to(tn.device)
    OnnxModel.test(model.simplify(True), (x,), project / 'best.onnx')
    return fit


if __name__ == '__main__':
    train_once(
        cfg=VOVNET / 'final.yaml',
        weight=VOVNET / 'final.pt',
        project=RUN / 'ema',
        hyp=Path('config/hyp-drop.yaml'),
        dataset=HOLD_OUT,
        freeze=None
    )
