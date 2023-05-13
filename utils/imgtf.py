from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

clip_abs = lambda x, a: np.clip(x, a_min=-a, a_max=a)


def to_tensor(img, pdim=(-1, -3, -2)):
    img = torch.from_numpy(np.ascontiguousarray(img[..., ::-1]))
    return img.permute(0, *pdim) if img.dim() == 4 else img.permute(*pdim)


def load_image(file_or_bgr, img_size=None) -> np.ndarray:
    if isinstance(file_or_bgr, Path):
        file_or_bgr = cv2.imread(str(file_or_bgr))
    if img_size:
        file_or_bgr = cv2.resize(file_or_bgr, (img_size,) * 2)
    return file_or_bgr


def letter_box(img, img_size=(640, 640), pad=114, stride=None):
    ''' 边界填充至指定尺寸'''
    shape = img.shape[:2]
    # 图像放缩比例
    r = min(img_size[0] / shape[0], img_size[1] / shape[1])
    # 放缩后的原始尺寸
    new_unpad = tuple(map(round, (shape[1] * r, shape[0] * r)))
    dw, dh = img_size[1] - new_unpad[0], img_size[0] - new_unpad[1]  # wh padding
    # 最小化边界尺寸
    if stride: dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    dw, dh = map(lambda x: x / 2, [dw, dh])
    # 对图像进行放缩
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 添加边界
    top, bottom = map(round, (dh - 0.1, dh + 0.1))
    left, right = map(round, (dw - 0.1, dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad,) * 3)  # add border
    return img, r, (dw, dh)


def img_mul(img, alpha):
    img = img.astype(np.float16)
    return np.uint8(np.clip((img * alpha).round(), a_min=0, a_max=255))


class _augment:

    def get_param(self) -> dict:
        return {}

    @staticmethod
    def apply(img) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, img):
        return self.apply(img, **self.get_param())


class RandomCrop(_augment):

    def __init__(self, min_radio=1., hflip=0., vflip=0.):
        for p in (min_radio, hflip, vflip): assert 0 <= p <= 1
        self.min_radio = min_radio
        self.flips = hflip, vflip

    def get_param(self):
        r = np.random.uniform(self.min_radio, 1)
        x, y = np.random.uniform(0, 1 - r, 2)
        flips = np.random.random(2) < self.flips
        return dict(x=x, y=y, r=r, hflip=flips[0], vflip=flips[1])

    @staticmethod
    def apply(img, x, y, r, hflip, vflip, **kwargs):
        if r < 1 or any((x, y)):
            H, W, C = img.shape
            x1, y1, x2, y2 = map(round, (x * W, y * H, (x + r) * W, (y + r) * H))
            img = cv2.resize(img[y1: y2, x1: x2], (W, H))
        # 图像翻转
        if any((hflip, vflip)):
            img = cv2.flip(img, flipCode=(hflip * 2 + vflip) % 3 - 1)
        return img


class ColorJitter(_augment):

    def __init__(self, hue=0., sat=0., value=0.):
        assert 0 <= hue <= .5 and 0 <= sat <= 1 and 0 <= value <= 1
        self.hue = hue
        self.sat = sat
        self.value = value

    def get_param(self):
        return dict(h=clip_abs(np.random.normal(0, self.hue / 2), 1),
                    s=clip_abs(np.random.normal(0, self.sat / 2), 1),
                    v=clip_abs(np.random.normal(0, self.value / 2), .5))

    @staticmethod
    def apply(img, h, s, v, **kwargs):
        h = round(180 * h)
        # 可行性判断
        flag = bool(h), abs(s) > 2e-3, abs(v) > 2e-3
        if any(flag):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # hsv 空间增强
            if flag[0]: img[..., 0] = cv2.LUT(img[..., 0], np.uint8((np.arange(h, h + 256)) % 180))
            if flag[1]: img[..., 1] = cv2.LUT(img[..., 1], img_mul(np.arange(256), s + 1))
            if flag[2]: img[..., 2] = cv2.LUT(img[..., 2], img_mul(np.arange(256), v + 1))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img


class GaussianBlur(_augment):
    ''' sigma: [min, std]'''

    def __init__(self, ksize: int, sigma=(0.1, 1.0)):
        self.ksize = ksize
        self.sigma = sigma

    def get_param(self):
        sigma = abs(np.random.normal(0, self.sigma[1])) + self.sigma[0]
        return dict(k=self.ksize, sigma=sigma)

    @staticmethod
    def apply(img, k, sigma, **kwargs):
        if k: img = cv2.GaussianBlur(img, ksize=(k,) * 2, sigmaX=sigma)
        return img


class Transform(list):

    def __init__(self, hyp):
        tfs = []
        if hyp:
            if not isinstance(hyp, dict):
                hyp = yaml.load(hyp.read_text(), Loader=yaml.Loader)
            tfs = [ColorJitter(*(hyp.get(f'hsv_{i}', 0) for i in 'hsv')),
                   RandomCrop(min_radio=hyp.get('cropr', 1), hflip=hyp.get('fliplr', 0), vflip=hyp.get('flipud', 0)),
                   GaussianBlur(ksize=hyp.get('gb_kernel', 0))]
            # 校对数据增强器的 apply 函数是否有重复的关键字参数
            key, t = sum(map(lambda tf: Counter(tf.get_param().keys()), tfs), Counter()).most_common(1)[0]
            assert t == 1, f'Duplicate keyword argument <{key}>'
        super().__init__(tfs)

    def get_param(self):
        param = {}
        for tf in self: param.update(tf.get_param())
        return param

    def apply(self, img, param):
        for tf in self: img = tf.apply(img, **param)
        return img

    def __call__(self, img):
        for tf in self: img = tf(img)
        return img


if __name__ == '__main__':
    from utils import timer

    cj = Transform(Path('../config/hyp.yaml'))

    img = cv2.imread('../data/both.png')


    @timer(1200)
    def test(img):
        return to_tensor(img)


    test(img)
