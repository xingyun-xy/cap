import logging
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pymongo
from bson.objectid import ObjectId
from capbc.database.config import load_config
from capbc.utils.aes_cipher import AESCipher
from pymongo.collection import Collection, ReturnDocument
from pymongo.cursor import Cursor

__all__ = ["MongoDBClient", "get_mongodb_client"]


logger = logging.getLogger(__name__)


class MongoDBClient:
    """Client for mongodb.

    Args:
        conn (str): Connection string.
        db (str, optional): Database name.
        table (str, optional): Collection name.

    """

    def __init__(
        self, conn: str, db: Optional[str] = None, table: Optional[str] = None
    ) -> None:
        self._conn = conn
        self._db = db
        self._table = table

        self.connect()
        if db and table:
            self.choose_collection(db, table)
        else:
            self._collection = None

    def connect(self) -> None:
        """Connect to mongodb database."""
        self._client = pymongo.MongoClient(self._conn)

    def choose_collection(self, db: str, table: str) -> None:
        """Choose a mongodb db name and table name.

        Args:
            db (str): Database name
            table (str): Collection name

        """
        self._db = db
        self._table = table
        self._collection = self._client[db][table]

    @property
    def collection(self) -> Collection:
        """The mongodb collection."""
        if self._collection is None:
            raise RuntimeError("Please choose db and table first.")
        return self._collection

    def insert_one(
        self,
        document: Dict,
        bypass_document_validation: Optional[bool] = False,
    ) -> ObjectId:
        """
        Insert a single document.

        Args:
            document (Dict): The document to insert. Must be a mutable mapping
                type. If the document does not have an _id field one will be
                added automatically.
            bypass_document_validation (bool, optional): If ``True``, allows
                the write to opt-out of document level validation.
                Default is ``False``.

        Raises:
            RuntimeError: If collection is None.

        Returns:
            ObjectId: The inserted document._id

        """
        if self._collection is None:
            raise RuntimeError("Please choose db and table first.")
        return self._collection.insert_one(
            document, bypass_document_validation
        ).inserted_id

    def insert_many(
        self,
        documents: Iterable[Dict],
        ordered: Optional[bool] = True,
        bypass_document_validation: Optional[bool] = False,
    ) -> Iterable[ObjectId]:
        """Insert an iterable of documents.

        Args:
            documents (Iterable[Dict]): A iterable of documents to insert.
            ordered (bool, optional): If ``True`` (the default) documents will
                be inserted on the server serially, in the order provided.
                If an error occurs all remaining inserts are aborted.
                If ``False``, documents will be inserted on the server
                in arbitrary order, possibly in parallel, and all document
                inserts will be attempted.
            bypass_document_validation (bool, optional): If ``True``, allows
                the write to opt-out of document level validation.
                Default is ``False``.

        Raises:
            RuntimeError: If collection is None.

        Returns:
            Iterable[ObjectId]: A iterable of ObjectId.

        """
        if self._collection is None:
            raise RuntimeError("Please choose db and table first.")
        return self._collection.insert_many(
            documents, ordered, bypass_document_validation
        ).inserted_ids

    def find_one(
        self, mongo_query: Optional[Dict] = None, *args, **kwargs
    ) -> Dict:
        """Get a single document from the database.

        Args:
            mongo_query (Dict, optional): A dictionary specifying the query
                to be performed OR any other type to be used as the value
                for a query for ``"_id"``.
            `*args` (optional): Any additional positional arguments
                are the same as the arguments to :meth:`find`.
            `**kwargs` (optional): Any additional keyword arguments
                are the same as the arguments to :meth:`find`.

        Raises:
            RuntimeError: If collection is None.

        Returns:
            Dict: The found document, ``None`` if no matching document
                is found.

        """
        if self._collection is None:
            raise RuntimeError("Please choose db and table first.")
        return self._collection.find_one(mongo_query, *args, **kwargs)

    def find(self, *args, **kwargs) -> Cursor:
        """Query the database.

        Args:
            `*args` (optional): Any additional positional arguments
                are the same as the arguments to :meth:`find`.
            `**kwargs` (optional): Any additional keyword arguments
                are the same as the arguments to :meth:`find`.

        Raises:
            RuntimeError: If collection is None.

        Returns:
            Cursor: The found documents

        """
        if self._collection is None:
            raise RuntimeError("Please choose db and table first.")
        return self._collection.find(*args, **kwargs)

    def delete_one(self, filter: Mapping[str, Any]):
        """Delete a single document matching the filter.

        Args:
            filter (Mapping[str, Any]): A query that matches the document
                to delete.

        Raises:
            RuntimeError: If collection is None.

        Returns:
            DeleteResult: An instance of
                :class:`~pymongo.results.DeleteResult`.

        """
        if self._collection is None:
            raise RuntimeError("Please choose db and table first.")
        return self._collection.delete_one(filter=filter)

    def delete_many(self, filter: Mapping[str, Any]):
        """Delete one or more documents matching the filter.

        Args:
            filter (Mapping[str, Any]): A query that matches the documents
                to delete.

        Raises:
            RuntimeError: If collection is None.

        Returns:
            DeleteResult: An instance of
                :class:`~pymongo.results.DeleteResult`.

        """
        if self._collection is None:
            raise RuntimeError("Please choose db and table first.")
        return self._collection.delete_many(filter=filter)

    def update_one(
        self,
        filter: Mapping[str, Any],
        update: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
        upsert: bool = False,
        bypass_document_validation: bool = False,
    ):
        """Update a single document matching the filter.

        Args:
            filter (Mapping[str, Any]): A query that matches the document
                to update.
            update (Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]):
                The modifications to apply.
            upsert (bool, optional): If ``True``, perform an insert
                if no documents match the filter.
            bypass_document_validation (bool, optional): If ``True``, allows
                the write to opt-out of document level validation.
                Default is ``False``.

        Raises:
            RuntimeError: If collection is None.

        Returns:
            UpdateResult: An instance of
                :class:`~pymongo.results.UpdateResult`.
        """
        if self._collection is None:
            raise RuntimeError("Please choose db and table first.")
        return self._collection.update_one(
            filter=filter,
            update=update,
            upsert=upsert,
            bypass_document_validation=bypass_document_validation,
        )

    def update_many(
        self,
        filter: Mapping[str, Any],
        update: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
        upsert: bool = False,
        array_filters: Optional[Sequence[Mapping[str, Any]]] = None,
        bypass_document_validation: bool = False,
    ):
        """Update a single document matching the filter.

        Args:
            filter (Mapping[str, Any]): A query that matches the document
                to update.
            update (Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]):
                The modifications to apply.
            upsert (bool, optional): If ``True``, perform an insert
                if no documents match the filter.
            array_filters (Sequence[Mapping[str, Any]], optional):
                A list of filters specifying which array elements
                an update should apply.
            bypass_document_validation (bool, optional): If ``True``, allows
                the write to opt-out of document level validation.
                Default is ``False``.

        Raises:
            RuntimeError: If collection is None.

        Returns:
            UpdateResult: An instance of
                :class:`~pymongo.results.UpdateResult`.
        """
        if self._collection is None:
            raise RuntimeError("Please choose db and table first.")
        return self._collection.update_many(
            filter=filter,
            update=update,
            upsert=upsert,
            array_filters=array_filters,
            bypass_document_validation=bypass_document_validation,
        )

    def find_one_and_update(
        self,
        filter: Mapping[str, Any],
        update: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
        projection: Optional[Union[Mapping[str, Any], Iterable[str]]] = None,
        sort: Optional[
            Sequence[Tuple[str, Union[int, str, Mapping[str, Any]]]]
        ] = None,
        upsert: bool = False,
        return_document: bool = ReturnDocument.BEFORE,
        array_filters: Optional[Sequence[Mapping[str, Any]]] = None,
    ):
        """Finds a single document and updates it,
        returning either the original or the updated document.

        Args:
            filter (Mapping[str, Any]): A query that matches the document
                to update.
            update (Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]):
                The update operations to apply.
            projection (Optional[Union[Mapping[str, Any], Iterable[str]]], optional):  # noqa
                A list of field names that should be
                returned in the result document or a mapping specifying the fields
                to include or exclude. If `projection` is a list "_id" will
                always be returned. Use a dict to exclude fields from
                the result (e.g. projection={'_id': False}).
            sort (Optional[Sequence[Tuple[str, Union[int, str, Mapping[str, Any]]]]], optional):
                A list of (key, direction) pairs
                specifying the sort order for the query. If multiple documents
                match the query, they are sorted and the first is updated.
            upsert (bool, optional): When ``True``, inserts a new document if no
                document matches the query. Defaults to ``False``.
            return_document (bool, optional): If
                :attr:`ReturnDocument.BEFORE` (the default),
                returns the original document before it was updated. If
                :attr:`ReturnDocument.AFTER`, returns the updated
                or inserted document.
            array_filters (Optional[Sequence[Mapping[str, Any]]], optional):
                 A list of filters specifying which
                array elements an update should apply.

        Raises:
            RuntimeError: If collection is None.

        Returns:
            Mapping[str, Any]: find and update result.
        """
        if self._collection is None:
            raise RuntimeError("Please choose db and table first.")
        return self._collection.find_one_and_update(
            filter=filter,
            update=update,
            projection=projection,
            sort=sort,
            upsert=upsert,
            return_document=return_document,
            array_filters=array_filters,
        )

    def find_one_and_delete(
        self,
        filter: Mapping[str, Any],
        projection: Optional[Union[Mapping[str, Any], Iterable[str]]] = None,
        sort: Optional[
            Sequence[Tuple[str, Union[int, str, Mapping[str, Any]]]]
        ] = None,
    ):
        """Finds a single document and deletes it, returning the document.

        Args:
            filter (Mapping[str, Any]): A query that matches the document
                to delete.
            projection (Optional[Union[Mapping[str, Any], Iterable[str]]], optional):  # noqa
                A list of field names that should be
                returned in the result document or a mapping specifying
                the fields to include or exclude.
                If `projection` is a list "_id" will always be returned.
                Use a mapping to exclude fields from
                the result (e.g. projection={'_id': False}).
            sort (Optional[Sequence[Tuple[str, Union[int, str, Mapping[str, Any]]]]], optional):  # noqa
                A list of (key, direction) pairs
                specifying the sort order for the query. If multiple documents
                match the query, they are sorted and the first is deleted.

        Raises:
            RuntimeError: If collection is None.

        Returns:
            Mapping[str, Any]: find and delete result.
        """
        if self._collection is None:
            raise RuntimeError("Please choose db and table first.")
        return self._collection.find_one_and_delete(
            filter=filter, projection=projection, sort=sort
        )

    def close(self) -> None:
        """Cleanup client resources and disconnect from MongoDB.

        On MongoDB >= 3.6, end all server sessions created by this client by
        sending one or more endSessions commands.

        Close all sockets in the connection pools and stop the monitor threads.
        If this instance is used again it will be automatically re-opened and
        the threads restarted unless auto encryption is enabled. A client
        enabled with auto encryption cannot be used again after being closed;
        any attempt will raise :exc:`~.errors.InvalidOperation`.

        .. versionchanged:: 3.6
           End all server sessions created by this client.
        """
        self._client.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        state["_collection"] = None
        if self._collection is not None:
            state["_db"] = self._collection.database.name
            state["_table"] = self._collection.name
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.connect()
        if state["_db"] is not None and state["_table"] is not None:
            self.choose_collection(state["_db"], state["_table"])


def get_mongodb_client(db: str, table: Optional[str] = None):
    """Get mongodb client.

    Args:
        db (str): Database name
        table (str, optional): Collection name

    Returns:
        MongoDBClient: The MongoDBClient instance

    """
    try:
        db_config = load_config()
        conn = db_config[db]["conn"]
        with_enc = db_config[db]["with_enc"]
        if with_enc:
            conn = AESCipher().decrypt(conn)
        mongo_client = MongoDBClient(conn, db, table)
        return mongo_client
    except Exception as e:
        logger.exception(e)
        return None
