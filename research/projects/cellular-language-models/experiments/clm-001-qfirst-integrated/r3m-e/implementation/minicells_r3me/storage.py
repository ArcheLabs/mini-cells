"""R3m-E namespaced reuse of the existing immutable HF segment store."""

from minicells_r3j.storage import Incremental as _Incremental, publish_segment
from minicells_r3k.reuse import bind


def _publish(*args, **kwargs):
    kwargs["pointer_path"] = kwargs["pointer_path"].replace("r3j/", "r3m-e/", 1)
    return publish_segment(*args, **kwargs)


class Publisher(_Incremental):
    publish = bind(_Incremental.publish, publish_segment=_publish)
