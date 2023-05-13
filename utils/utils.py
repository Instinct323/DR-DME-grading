import json
import logging
import os
import pickle
from pathlib import WindowsPath, PosixPath

import cv2
import pandas as pd
import yaml

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def colorstr(*args):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = args if len(args) > 1 else ('blue', 'bold', args[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def timer(repeat=1, avg=True):
    import time
    repeat = max([1, int(repeat) if isinstance(repeat, float) else repeat])

    def decorator(func):
        def handler(*args, **kwargs):
            start = time.time()
            result = tuple(func(*args, **kwargs) for i in range(repeat) if not i)[0]
            cost = (time.time() - start) * 1e3
            print(f'{func.__name__}: {cost / repeat if avg else cost:.3f} ms')
            return result

        return handler

    return decorator


def singleton(cls):
    ''' 单例类装饰器'''
    _instance = {}

    def _singleton(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            LOGGER.error(f'{type(error).__name__}: {error}')

    return handler


class Path(WindowsPath if os.name == 'nt' else PosixPath):

    def lazy_obj(self, fget, *args, **kwargs):
        if self.is_file():
            data = self.pickle()
            LOGGER.info(f'Load <{type(data).__name__}> from {self}')
        else:
            data = fget(*args, **kwargs)
            self.pickle(data)
        return data

    def pickle(self, data=None):
        f = self.open('rb') if data is None else self.open('wb')
        return pickle.load(f) if data is None else pickle.dump(data, f)

    def json(self, data=None):
        f = self.open('r') if data is None else self.open('w')
        return json.load(f) if data is None else json.dump(data, f, indent=4)

    def yaml(self, data=None):
        return yaml.load(self.read_text(), Loader=yaml.Loader) \
            if data is None else self.write_text(yaml.dump(data))

    @try_except
    def csv(self, data=None):
        return pd.read_csv(self) if data is None else data.to_csv(self)

    @try_except
    def excel(self, data=None, float_format='%.4f'):
        if data is None: return pd.read_excel(self)
        writer = pd.ExcelWriter(self)
        for df in [data] if isinstance(data, pd.DataFrame) else dataframe:
            df.to_excel(writer, float_format=float_format)
        writer.save()


class Capture(cv2.VideoCapture):
    ''' 视频捕获
        file: 视频文件名称 (默认连接摄像头)
        show: 视频帧的滞留时间 (ms)
        dpi: 相机分辨率'''

    def __init__(self,
                 file: str = 0,
                 show: int = 0,
                 dpi: list = [1280, 720]):
        super(Capture, self).__init__(file)
        if not self.isOpened():
            raise RuntimeError('Failed to initialize video capture')
        self.show = show
        # 设置相机的分辨率
        if not file and dpi:
            self.set(cv2.CAP_PROP_FRAME_WIDTH, dpi[0])
            self.set(cv2.CAP_PROP_FRAME_HEIGHT, dpi[1])

    def __iter__(self):
        return self

    def __next__(self):
        ok, image = self.read()
        if not ok: raise StopIteration
        if self.show:
            cv2.imshow('Capture', image)
            cv2.waitKey(self.show)
        return image


def waitkey(delay: int):
    ''' 捕获键盘输入'''
    key = cv2.waitKey(delay)
    if key != -1: return chr(key)


if __name__ == '__main__':
    file = Path('../config/cnn') / 'ResNet-50.yaml'
    print(file.yaml())
