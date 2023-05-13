from data import *
from engine.loss import *
from engine.trainer import Trainer


class Classifier(Trainer):

    def __init__(self, model, project, hyp):
        m_title = ['DR P-', 'DR AP+', 'DME P-', 'DME AP+', 'DR K', 'DME K', 'DR Acc', 'DME Acc']
        super().__init__(model, project, m_title=m_title, hyp=hyp)
        self.fce = CrossEntropy(l2penalty=0), CrossEntropy(l2penalty=0)

    def loss(self, image, target):
        target = target.to(self.device)
        pred = self.model(image.to(self.device)).split([DR, DME], dim=-1)
        return sum(self.fce[i](pred[i], target[:, i]).mean() for i in range(2))

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
               freeze=None, strict=True, patch_size=None):
    ''' return: best fitness'''
    model, cfg, hyp = load_model(cfg, weight, hyp, freeze=freeze, strict=strict)
    # 如果使用了 ViT, 设置掩码
    if patch_size:
        LOCAL.pmask = torch.from_numpy(get_mask(patch_size))
    # 读取数据集, 创建训练器, 开始训练
    DATAPOOL.loadimg(cfg['img_size'])
    tn = Classifier(model, project, hyp=hyp)
    return tn(*data_loader(hyp, *dataset))


if __name__ == '__main__':
    train_once(
        cfg=VOVNET / 'drop.yaml',
        weight=VOVNET / 'drop.pt',
        project=RUN / 'tea',
        hyp=Path('config/hyp-vit.yaml'),
        dataset=HOLD_OUT,
        freeze=None,
        strict=True
    )
