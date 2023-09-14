from typing import Union, List

from capbc.utils import _as_list

__all__ = ["MessageMeta", "Message", "filter_topics", "filter_meta"]


class SerializeableObject:

    def to_proto(self):
        raise NotImplementedError

    @classmethod
    def from_proto(cls, proto):
        """Create a Message from a protobuf message.

        TODO: Implement this function when .proto files are fully designed.
        """
        raise NotImplementedError


class MessageMeta(SerializeableObject):

    __slots__ = ["timestamp", "channel"]

    def __init__(self, timestamp: int = None, channel: int = None):
        self.timestamp = timestamp
        self.channel = channel


class Message(SerializeableObject):

    __slots__ = ["topic", "meta"]

    def __init__(self, topic: str = None, meta: MessageMeta = None):
        self.topic = topic
        self.meta = meta


def filter_topics(
    messages: List[Message],
    topics: Union[List[str], str] = None
) -> List[Message]:
    """ Filter messages by topic.

    Args:
        messages (List[Message]): messages to filter.
        topics (Union[List[str], str], optional): wanted topics. If None, \
            return all messages. Defaults to None.

    Returns:
        List[Message]: filtered messages.
    """
    if not topics or not messages:
        return messages

    topics = _as_list(topics)
    return [m for m in messages if m.topic in topics]


def filter_meta(
    messages: List[Message],
    channels: Union[List[int], int] = None,
) -> List[Message]:
    """ Filter messages by meta info.

    Args:
        messages (List[Message]): messages to filter.
        channels (Union[List[int], int], optional): wanted channels. If None, \
            return all messages. Defaults to None.

    Returns:
        List[Message]: filtered messages.
    """
    if not messages:
        return messages

    if channels is not None:
        channels = _as_list(channels)
        filtered_messages = []
        for msg in messages:
            if msg.meta and msg.meta.channel in channels:
                filtered_messages.append(msg)
        messages = filtered_messages

    return messages
