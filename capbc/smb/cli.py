import os
import yaml
import getpass
from argparse import ArgumentParser

from .config import Config, DEFAULT_CONFIG_PATH
from .api import SMBClient


def configure_parser(parser: ArgumentParser):
    parser.add_argument('--username', type=str)
    parser.add_argument('--password', type=str)
    parser.add_argument('--config-path', type=str, default=DEFAULT_CONFIG_PATH)
    parser.set_defaults(func=configure_main)


def configure_main(args):
    if args.username and not args.password:
        args.password = getpass.getpass()
    if os.path.exists(args.config_path):
        with open(args.config_path) as fin:
            d = yaml.load(fin, Loader=yaml.SafeLoader)
            config = Config.from_dict(d)
        config.update(
            username=args.username,
            password=args.password,
        )
    else:
        config = Config(
            username=args.username,
            password=args.password,
        )

    config_dir = os.path.dirname(args.config_path)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    with open(args.config_path, 'w') as fout:
        config_dict = config.to_dict()
        yaml.dump(config_dict, fout, default_flow_style=False,
                  Dumper=yaml.SafeDumper)


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    configure_parser(subparsers.add_parser('configure'))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
