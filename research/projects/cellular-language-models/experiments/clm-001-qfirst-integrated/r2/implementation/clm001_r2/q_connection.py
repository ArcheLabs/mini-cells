"""Same parameter inventory; subtracting STOP changes supervision, not ordering."""

from clm001.q_connection import Connection as RawConnection


class Connection(RawConnection):
    def forward(self, *args, **kwargs):
        raw, critic = super().forward(*args, **kwargs)
        # RawConnection returns unmasked scores; STOP reference exists even when illegal.
        return raw - raw[..., -1:], critic
