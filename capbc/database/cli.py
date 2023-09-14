import argparse
import os

import yaml
from capbc.database.config import DEFAULT_CONFIG_PATH, load_config
from capbc.database.mongodb_client import MongoDBClient
from capbc.database.mysql_client import MySQLClient, MySQLConn
from capbc.utils.aes_cipher import AESCipher


def add_configure(args: argparse.Namespace):
    config_path = os.path.expanduser(args.path)

    if os.path.exists(config_path):
        configs = load_config(config_path)
        configs.update(
            {
                args.db_name: {
                    "db_type": args.db_type,
                    "conn": args.conn,
                    "with_enc": args.with_enc,
                }
            }
        )
    else:
        configs = {
            args.db_name: {
                "db_type": args.db_type,
                "conn": args.conn,
                "with_enc": args.with_enc,
            }
        }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(args.path, "w") as fout:
        yaml.safe_dump(configs, fout)

    if args.check:
        if args.db_type == "mongodb":
            if args.with_enc:
                conn = AESCipher().decrypt(args.conn)
            else:
                conn = args.conn
            _ = MongoDBClient(conn)
        elif args.db_type == "mysql":
            if args.with_enc:
                conn = AESCipher().decrypt(args.conn)
            else:
                conn = args.conn
            mysql_conn = MySQLConn.parse(conn)
            _ = MySQLClient(
                mysql_conn.host,
                mysql_conn.port,
                mysql_conn.user,
                mysql_conn.password,
                args.db_name,
            )
        else:
            raise NotImplementedError(args.db_type)


def remove_configure(args: argparse.Namespace):
    config_path = os.path.expanduser(args.path)

    if os.path.exists(config_path):
        configs = load_config(config_path)
        if args.db_name in configs:
            del configs[args.db_name]

        with open(args.path, "w") as fout:
            yaml.safe_dump(configs, fout)


def add_parser_arguments(parser):
    parser.add_argument(
        "--db-name", required=True, type=str, help="Database name"
    )
    parser.add_argument(
        "--conn", required=True, type=str, help="Connection string"
    )
    parser.add_argument(
        "--db-type",
        type=str,
        default="mongodb",
        choices=["mongodb", "mysql"],
        help="Database type",
    )
    parser.add_argument(
        "--with-enc",
        action="store_true",
        default=True,
        help="Whether the connection string is encoded",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Whether to check config is valid",
    )
    parser.set_defaults(func=add_configure)


def remove_parser_arguments(parser):
    parser.add_argument(
        "--db-name", required=True, type=str, help="Database name"
    )
    parser.set_defaults(func=remove_configure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="The yaml config path",
    )

    subparsers = parser.add_subparsers()
    add_parser = subparsers.add_parser("add", help="add help")
    add_parser_arguments(add_parser)

    remove_parser = subparsers.add_parser("remove", help="remove help")
    remove_parser_arguments(remove_parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
