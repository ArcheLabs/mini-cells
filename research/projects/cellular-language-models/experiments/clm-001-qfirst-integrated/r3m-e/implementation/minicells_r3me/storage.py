"""R3m-E namespaced reuse of the existing immutable HF segment store."""

from minicells_r3j.storage import Incremental as _Incremental, publish_segment
from minicells_r3k.reuse import bind


def _publish(*args, **kwargs):
    kwargs["pointer_path"] = kwargs["pointer_path"].replace("r3j/", "r3m-e/", 1)
    return publish_segment(*args, **kwargs)


class Publisher(_Incremental):
    publish = bind(_Incremental.publish, publish_segment=_publish)


class SnapshotPublisher:
    """Reuse Incremental after a short locked local snapshot; network I/O holds no GPU lock."""
    def __init__(self, backend, private, output, identity, session, prior=None):
        from pathlib import Path
        self.backend, self.private, self.output = backend, Path(private), Path(output)
        self.identity, self.session, self.prior = identity, session, prior

    def publish(self, force=False):
        import os
        import shutil
        import tempfile
        from minicells_r3e.storage import snapshot_lock
        with tempfile.TemporaryDirectory(dir=self.output, prefix='hf-snapshot-') as temporary:
            from pathlib import Path
            frozen = Path(temporary)
            with snapshot_lock(self.output / 'private-checkpoints'):
                for source in self.private.rglob('*'):
                    if source.is_symlink():
                        raise RuntimeError('R3ME_SNAPSHOT_SYMLINK_FORBIDDEN')
                    if not source.is_file() or source.suffix not in ('.json', '.pt'):
                        continue
                    dest = frozen / source.relative_to(self.private)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if source.suffix == '.pt':
                        try:
                            os.link(source, dest)
                        except OSError:
                            shutil.copyfile(source, dest)
                    else:
                        shutil.copyfile(source, dest)
            writer = Publisher(self.backend, frozen, self.output, self.identity, self.session, self.prior)
            self.prior = writer.publish(force)
        return self.prior
