import logging
import os
from datetime import datetime
from functools import reduce
from importlib.machinery import SourceFileLoader
from operator import getitem
from pathlib import Path

from logger import setup_logging
from utils.util import write_config


def parse_cmd_args(args):
    # parse default cli options
    args = args.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.resume:
        resume = Path(args.resume)
        cfg_fname = resume.parent / 'config.py'
    else:
        msg_no_cfg = "Configuration file need to be specified. Add '-c config.py', for example."
        assert args.config is not None, msg_no_cfg
        resume = None
        cfg_fname = Path(args.config)

    # load config file and apply custom cli options
    config = SourceFileLoader("CONFIG", str(cfg_fname)).load_module().CONFIG

    for key, value in args.__dict__.items():
        config[key] = value
    return config, resume


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = config
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        # write_json(self.config, self.save_dir / 'config.json')
        write_config(self.config, self.save_dir / 'config.py')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def initialize(self, name, module, *args):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_cfg = self[name]
        return getattr(module, module_cfg['type'])(*args, **module_cfg['args'])

    def initialize_class(self, name, module, *args):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        class_instance = self.retrieve_class(name, module)
        return class_instance(*args, **self[name]['args'])

    def retrieve_class(self, name, module):
        module_cfg = self[name]
        class_name = module_cfg["type"]
        base_path = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))), module.__name__, f'{class_name}.py')
        class_instance = getattr(SourceFileLoader(class_name, base_path).load_module(), class_name)
        return class_instance

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
