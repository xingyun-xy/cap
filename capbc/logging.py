import abc
import logging
import logging.config
import logging.handlers
import os
import sys
from typing import Union

import yaml
from deprecated import deprecated

__all__ = ["DefaultLoggingConfig", "FileLoggingConfig", "LoggerManager"]

HANDLER_TYPE = {
    "file": logging.FileHandler,
    "stream": logging.StreamHandler,
    "rotatingfile": logging.handlers.RotatingFileHandler,
}

DEFAULT_LOG_FORMAT = (
    "%(asctime)-15s %(levelname)s "
    "| %(process)d | %(threadName)s | "
    "%(module)s:%(name)s:L%(lineno)d %(message)s"
)


# default configuration of root logging
DEFAULT_LOG_CONFIG = """
version: 1
disable_existing_loggers: False
formatters:
    default:
        format: '{format}'

handlers:
    info_console:
        class: logging.StreamHandler
        level: INFO
        formatter: default
        stream: ext://sys.stderr  # stdout may cause problem in cluster

    warning_console:
        class: logging.StreamHandler
        level: WARNING
        formatter: default
        stream: ext://sys.stderr

    debug_file_handler:
        class: logging.handlers.RotatingFileHandler
        level: DEBUG
        formatter: default
        filename: {log_dir}/debug.log
        maxBytes: 104857600       # 100MB
        backupCount: 10000        # about 1T logs files
        encoding: utf8

    warning_file_handler:
        class: logging.handlers.RotatingFileHandler
        level: WARNING
        formatter: default
        filename: {log_dir}/warn.log
        maxBytes: 104857600       # 100MB
        backupCount: 10000        # about 1T logs files
        encoding: utf8

    error_file_handler:
        class: logging.handlers.RotatingFileHandler
        level: ERROR
        formatter: default
        filename: {log_dir}/error.log
        maxBytes: 104857600       # 100MB
        backupCount: 10000        # about 1T logs files
        encoding: utf8

loggers:
    auto_dp:
        level: WARNING
        handlers: [warning_console]
        propagate: no

root:
    level: DEBUG
    handlers: [info_console, debug_file_handler,
        warning_file_handler, error_file_handler]
"""


class BaseLoggingConfig(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def set_logging_config(self):
        pass


# TODO: delete it at a proper time.
# Since we have wrote a new class named "LoggerManager", this default
# configuration class seems strange. We should use the manager to mantain all
# loggers. But because now there are many repos call the class. We will keep it
@deprecated(
    "DO NOT Config the root logger directly."
    "Check the usage of LoggerManager"
)
class DefaultLoggingConfig(BaseLoggingConfig):
    """This class only modifies the configuration for root logger"""

    def __init__(
        self, level=logging.INFO, stream=sys.stderr, format=DEFAULT_LOG_FORMAT
    ):  # noqa
        assert stream in [sys.stdout, sys.stderr]
        self.level = level
        self.stream = stream
        self.format = format

    # we found that in some platform, stream cannot be serialized...
    def __getstate__(self):
        state = self.__dict__.copy()
        state["stream"] = "stdout" if self.stream == sys.stdout else "stderr"
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        if self.stream == "stderr":
            self.stream = sys.stderr
        elif self.stream == "stdout":
            self.stream = sys.stdout
        else:
            raise TypeError(f"Invalid stream {self.stream}")

    def set_logging_config(self):
        logging.basicConfig(
            level=self.level, stream=self.stream, format=self.format
        )


class LoggerManager:
    """A manager to control the behaviour of loggers

    It support 2 ways to change the configuration of loggers:

    1. Initial configuration through yaml file or a dictionary. If nothing
    provided, default configuration will be used
    2. Dynamically change the logger configurations through code
    """

    def __init__(
        self,
        level: str = None,
        fpath: str = None,
        dict_config: dict = None,
        log_dir: str = None,
    ):
        """init
        Parameters
        ----------
        level : str, optional
            root logging level, by default None
        fpath : str, optional
            config file path, by default None
        cfg_dict: dict, optional
            config dict, by default None. If there are both `fpath` and
            `cfg_dict`, it will cause error
        log_dir : str, optional
            The output log dir path
        """
        assert (
            fpath is None or dict_config is None
        ), "DO NOT both file configuration and dict configuration"

        if dict_config is not None:
            self.config = dict_config
        else:
            self.config = self._get_yaml_config(fpath, log_dir)

        self._create_log_file()
        logging.config.dictConfig(self.config)

        if level is not None:
            self.set_logger_level("root", level)

    def _create_log_file(self):
        """create logger directories

        When we configure the handler filename as 'folder1/xxx.log', if the
        folder doesn't exist, the logging module will raise error.
        """
        handlers = self.config.get("handlers")
        if handlers is None:
            return

        for _, handler in handlers.items():
            filename = handler.get("filename")
            if filename is None:
                continue
            dir = os.path.dirname(filename)
            if dir != "":
                os.makedirs(dir, exist_ok=True)

    def _create_format(self, format: str = DEFAULT_LOG_FORMAT):
        """create a formatter

        This function should be used CAREFULLY! Once the log format changed,
        all scripts related to logs may need to changed. When users using this
        function, we hope he know what he is doing. Thus the input is just a
        string.

        Go https://docs.python.org/3/library/logging.html#logging.Formatter to
        check how to write a formatter string.

        Parameters
        ----------
        format : str, optional
            formatter style string., by default DEFAULT_LOG_FORMAT

        Returns
        -------
        logging.Formatter
            formatter: logging formatter
        """
        return logging.Formatter(format)

    def _get_yaml_config(self, fpath: str = None, log_dir: str = None):
        """Read configurations from yaml file

        Parameters
        ----------
        fpath : str, optional
            path of the configuration file, by default None
        log_dir : str, optional
            The output log dir path

        Returns
        -------
        dict
            dictionary of logging configurations
        """
        if fpath is None:
            if is_running_on_sda():
                log_dir = "/job_log"
            elif log_dir is None:
                log_dir = "log"
            config = yaml.safe_load(
                DEFAULT_LOG_CONFIG.format(
                    format=DEFAULT_LOG_FORMAT, log_dir=log_dir
                )
            )
        else:
            assert os.path.exists(fpath), "No configuration file was found!"
            if log_dir:
                raise ValueError(f"log_dir should be None when fpath exists")
            with open(fpath) as f:
                config = yaml.safe_load(f.read())

        return config

    def get_logger(self, logger_name: str) -> logging.Logger:
        """get logger

        Parameters
        ----------
        logger_name : str
            name of the logger

        Returns
        -------
        logging.Logger
            object of logger
        """
        if logger_name == "root":
            logger = logging.getLogger()
        else:
            logger = logging.getLogger(logger_name)

        return logger

    def _create_handler(
        self,
        level: str,
        type: str,
        formatter: str = DEFAULT_LOG_FORMAT,
        **kwargs,
    ):
        """creat a desired handler for logger

        Parameters
        ----------
        level : str
            level of the handler
        type : str
            handler type. Currently only suppport:
                file -- FileHandler
                stream -- StreamHandler
                rotatingfile -- RotatingFileHandler
        formatter : str, optional
             formatter string, by default DEFAULT_LOG_FORMAT
        kwargs: dict
            argument provide for handler

        Returns
        -------
        handler
            a handler to deal with the logging information
        """
        # create a corresponding type of handler
        h = HANDLER_TYPE[type](**kwargs)

        # configure the handler
        fmt = self._create_format(formatter)
        level = logging.getLevelName(level.upper())
        if isinstance(level, int):
            h.setLevel(level)
        h.setFormatter(fmt)

        return h

    def set_logger_level(self, logger_name: str, level: Union[str, int]):
        """change level for certain logger

        Parameters
        ----------
        logger_name : str
            name of the logger
        level : Union[str, int]
            level to be set. Both string and int are acceptable
        """
        if isinstance(level, str):
            # make sure the level string is upper case
            level = str.upper(level)
        lv = logging.getLevelName(level)
        # check if the level is valid
        if isinstance(lv, str) and "level" in lv:
            return
        logger = self.get_logger(logger_name)
        logger.setLevel(lv)

    def set_handler(
        self,
        logger_name: str,
        level: str = "info",
        type: str = "stream",
        fmt: str = DEFAULT_LOG_FORMAT,
        **kwargs,
    ):
        """set handler to the logger

        Parameters
        ----------
        logger_name : str
            logger name
        level : str, optional
            handler log level, by default 'info'
        type : str, optional
            handler type, by default 'stream'
        fmt : str, optional
            handler format, by default DEFAULT_LOG_FORMAT
        """
        logger = self.get_logger(logger_name)
        handler = self._create_handler(level, type, fmt, **kwargs)
        logger.addHandler(handler)

    def remove_handler_by_level(self, logger_name: str, level: str):
        """remove certain level of handlers

        Parameters
        ----------
        logger_name : str
            name of the logger
        level : str
            level of the handlers
        """
        logger = self.get_logger(logger_name)
        level = logging.getLevelName(level)
        logger.handlers = [h for h in logger.handlers if h.level != level]

    def remove_handler_by_type(self, logger_name: str, htype: str):
        """remove certain type of handlers

        Parameters
        ----------
        logger_name : str
            name of the logger
        htype : str
            handler type
        """
        logger = self.get_logger(logger_name)
        hclass = HANDLER_TYPE.get(htype)
        if hclass is None:
            return
        logger.handlers = [h for h in logger.handlers if type(h) != hclass]

    def clear_handler(self, logger_name: str):
        """remove all handlers for a logger

        Parameters
        ----------
        logger_name : str
            name of the logger
        """
        logger = self.get_logger(logger_name)
        logger.handlers.clear()


class FileLoggingConfig(BaseLoggingConfig):
    """A file logging config which read config from file or dict"""

    def __init__(
        self,
        level: str = None,
        fpath: str = None,
        dict_config: dict = None,
        log_dir: str = None,
    ):
        """init
        Parameters
        ----------
        level : str, optional
            root logging level, by default None
        fpath : str, optional
            config file path, by default None
        cfg_dict: dict, optional
            config dict, by default None. If there are both `fpath` and
            `cfg_dict`, it will cause error
        log_dir : str, optional
            The output log dir path
        """
        assert (
            fpath is None or dict_config is None
        ), "DO NOT both file configuration and dict configuration"

        self.level = level
        self.fpath = fpath
        self.dict_config = dict_config
        self.log_dir = log_dir

    def set_logging_config(self):
        LoggerManager(
            level=self.level,
            fpath=self.fpath,
            dict_config=self.dict_config,
            log_dir=self.log_dir,
        )


default_logging_config = FileLoggingConfig()
