"""R3h Cell training with an explicit, receipt-bound Connection commit boundary."""

import copy
from minicells_r3h.trainer import Trainer as FrozenTrainer
from minicells_r3e.common import fingerprint


class Trainer(FrozenTrainer):
    def __init__(self, model, c, arm):
        super().__init__(model, c, arm)
        self.connection_commits = []
        self.initial_connection_sha=self.connection_sha

    def authorize_connection(self, connection, *, transaction, accepted, expected_previous):
        if not transaction or fingerprint(self.model.connection.state_dict()) != expected_previous:
            raise RuntimeError("CONNECTION_COMMIT_PARENT_MISMATCH")
        before = copy.deepcopy(self.model.connection.state_dict())
        self.model.connection.load_state_dict(connection, strict=True)
        sha = fingerprint(self.model.connection.state_dict())
        if not accepted and sha != expected_previous:
            self.model.connection.load_state_dict(before, strict=True)
            raise RuntimeError("REJECTED_CONNECTION_CHANGED")
        self.connection_sha = sha
        self.connection_version += int(accepted)
        self.connection_commits.append(
            dict(
                transaction=transaction, accepted=accepted, previous=expected_previous, current=sha
            )
        )
        self.verify_frozen()

    def state(self):
        state = super().state()
        state["r3i"] = {"connection_commits": copy.deepcopy(self.connection_commits)}
        return state

    def restore(self, state):
        # Validate the accepted lineage before allowing R3h's fixed-Connection hash to change.
        lineage = state["r3i"]["connection_commits"]
        expected = self.initial_connection_sha
        for row in lineage:
            if row["previous"] != expected or not row["transaction"]:
                raise RuntimeError("CONNECTION_COMMIT_LINEAGE_INVALID")
            if not row["accepted"] and row["current"] != expected:
                raise RuntimeError("REJECTED_CONNECTION_LINEAGE_CHANGED")
            expected = row["current"]
        if (
            fingerprint(
                {
                    k[len("connection.") :]: v
                    for k, v in state["model"].items()
                    if k.startswith("connection.")
                }
            )
            != expected
        ):
            raise RuntimeError("CONNECTION_STATE_WITHOUT_COMMIT")
        if state["r3h"]["connection_sha"] != expected:
            raise RuntimeError("CONNECTION_CHECKPOINT_HASH_MISMATCH")
        previous = self.connection_sha
        self.connection_sha = expected
        try:
            super().restore(state)
        except Exception:
            self.connection_sha = previous
            raise
        self.connection_commits = copy.deepcopy(lineage)

    def clone(self):
        clone = Trainer(copy.deepcopy(self.model), self.c, self.arm)
        clone.opt_identity = self.opt_identity
        clone.parent_identity = copy.deepcopy(self.parent_identity)
        # Restore validates the entire lineage from the original zero Connection.
        clone.connection_sha = (
            self.connection_commits[0]["previous"]
            if self.connection_commits
            else self.connection_sha
        )
        clone.restore(copy.deepcopy(self.state()))
        return clone
