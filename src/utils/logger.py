import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys

import logging

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print('For PyTorch <= 1.0, tensorboardX should be installed')
        sys.exit(1)





class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self, *args, **kwargs):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls(*args, **kwargs)
        return self._instance[self._cls]


@Singleton
class Logger(object):
    def __init__(self,
                 root_path='./runs/1',
                 logger_name='train',
                 level=logging.INFO,
                 toscreen=False,
                 tofile=True):
        self._logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
            datefmt='%y-%m-%d %H:%M:%S')
        self._logger.setLevel(level)
        self.root_path = root_path
        self.logger_name = logger_name
        if tofile:
            os.makedirs(root_path, exist_ok=True)
            log_file = os.path.join(root_path, logger_name + '.log')
            fh = logging.FileHandler(log_file, mode='w')
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)
        if toscreen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self._logger.addHandler(sh)

    def get_logger(self):
        return self._logger

    def be_quiet(self):
        self._logger.setLevel(logging.WARNING)
    
    def get_path(self):
        return self.root_path
        

@Singleton
class EvalState(object):
    def __init__(self, is_eval=False):
        self._is_eval = is_eval
    
    def get_eval(self):
        return self._is_eval


@Singleton
class SummaryWriterSingleton(SummaryWriter):
    pass




def get_tb_logger() -> SummaryWriter:
    return SummaryWriterSingleton(log_dir='./fake_path/')


def get_logger() -> Logger:
    return Logger().get_logger()



def get_logger_path():
    return Logger().get_path()


def init_loggers(dir_path, logger_name, use_tb_logger, suffix=None, screen=True):
    if suffix is not None:
        logger = Logger(dir_path, '-'.join([logger_name, suffix]), toscreen=screen).get_logger()
    else:
        logger = Logger(dir_path, logger_name, toscreen=screen).get_logger()

    tb_logger = None
    if use_tb_logger:
        tb_logger = SummaryWriterSingleton(os.path.join(dir_path, 'tb_logs'))
    # print(train_url, opt.train.logger.logger_name)
    return logger, tb_logger


def is_eval():
    return EvalState().get_eval()


def init_state(is_eval):
    EvalState(is_eval)


def set_quiet():
    Logger().be_quiet()

