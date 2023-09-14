import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pymysql
from capbc.database.config import load_config
from capbc.utils.aes_cipher import AESCipher
from pymysql.cursors import Cursor

logger = logging.getLogger(__name__)


__all__ = ["MySQLClient", "MySQLConn", "get_mysql_client"]


class MySQLClient:
    """Client for MySQL.

    Args:
        host (str): Host IP.
        port (int): Host port.
        user (str): User to login.
        password (str): Password to login.
        database (str): Database name to use.

    """

    def __init__(
        self, host: str, port: int, user: str, password: str, database: str
    ):
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database

        self.connect()

    def connect(self) -> None:
        """Connect to mysql database."""
        self._client = pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
        )

    @property
    def cursor(self) -> Cursor:
        """The mysql cursor."""
        return self._client.cursor()

    def fetchone(
        self, query: str, args: Optional[Union[Tuple, Dict, List]] = None
    ):
        """Fetch a row data.

        Args:
            query (str): Query to execute.
            args (Union[Tuple, Dict, List], optional): Parameters used
                with query.

        Returns:
            Tuple: One row data

        """
        cursor = self.cursor
        cursor.execute(query, args)
        data = cursor.fetchone()
        cursor.close()
        return data

    def fetchmany(
        self,
        query: str,
        args: Optional[Union[Tuple, Dict, List]] = None,
        size: Optional[int] = None,
    ):
        """Fetch many row data.

        Args:
            query (str): Query to execute.
            args (Union[Tuple, Dict, List], optional): Parameters used
                with query.
            size (int, optional): Limit rows.

        Returns:
            List[Tuple]: Several row data.

        """
        cursor = self.cursor
        cursor.execute(query, args)
        data = cursor.fetchmany(size)
        cursor.close()
        return data

    def fetchall(
        self, query: str, args: Optional[Union[Tuple, Dict, List]] = None
    ):
        """Fetch all row data.

        Args:
            query (str): Query to execute.
            args (Union[Tuple, Dict, List], optional): Parameters used
                with query.

        Returns:
            List[Tuple]: All row data.

        """
        cursor = self.cursor
        cursor.execute(query, args)
        data = cursor.fetchall()
        cursor.close()
        return data

    def execute(
        self, sql: str, args: Optional[Union[Tuple, Dict, List]] = None
    ) -> int:
        """Execute sql.

        Args:
            query (str): Query to execute.
            args (Union[Tuple, Dict, List], optional): Parameters used
                with query.

        Returns:
            int: The last row ID executed on the cursor object if successful,
                -1 otherwise.

        """
        cursor = self.cursor
        try:
            affected_rows = cursor.execute(sql, args)
            logger.info(
                f"Execute sql: {sql}, args: {args}, affected_rows:{affected_rows}"  # noqa
            )
            self._client.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.exception(e)
            self._client.rollback()
            return -1
        finally:
            cursor.close()

    update = execute
    delete = execute
    insert = execute

    def close(self):
        """Cleanup client resources and disconnect from MySQL."""
        self._client.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.connect()


@dataclass
class MySQLConn:
    host: str
    port: int
    user: str
    password: str

    @classmethod
    def parse(cls, conn: str):
        array = conn.split(",")
        return cls(
            host=array[0], port=int(array[1]), user=array[2], password=array[3]
        )

    def __str__(self) -> str:
        output = [self.host, str(self.port), self.user, self.password]
        return ",".join(output)


def get_mysql_client(db: str):
    """Get mysql client.

    Args:
        db (str): Database name.

    Returns:
        MySQLClient: The MySQLClient instance if successful, None otherwise.

    """
    try:
        db_config = load_config()
        conn = db_config[db]["conn"]
        with_enc = db_config[db]["with_enc"]
        if with_enc:
            conn = AESCipher().decrypt(conn)
        mysql_conn = MySQLConn.parse(conn)
        mysql_client = MySQLClient(
            mysql_conn.host,
            mysql_conn.port,
            mysql_conn.user,
            mysql_conn.password,
            db,
        )
        return mysql_client
    except Exception as e:
        logger.exception(e)
        return None
