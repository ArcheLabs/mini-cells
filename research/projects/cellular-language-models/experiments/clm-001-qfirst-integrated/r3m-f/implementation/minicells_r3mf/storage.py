"""New namespace over the verified R3m-E immutable snapshot publisher."""

from minicells_r3j.storage import Incremental, publish_segment
from minicells_r3k.reuse import bind
from minicells_r3me.storage import SnapshotPublisher as Previous


def publish(*args, **kwargs):
    kwargs["pointer_path"] = kwargs["pointer_path"].replace("r3j/", "r3m-f/", 1)
    return publish_segment(*args, **kwargs)


class Publisher(Incremental):
    publish = bind(Incremental.publish, publish_segment=publish)


class SnapshotPublisher(Previous):
    publish = bind(Previous.publish, Publisher=Publisher)
