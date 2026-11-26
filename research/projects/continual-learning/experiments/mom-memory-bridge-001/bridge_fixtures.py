"""The nine single-target records from the MOM-ROUTE-002 development fixture."""
from dataclasses import dataclass

ROUTE_SOURCE = "2a0d0d70d2445ad04257ac544ea44147c1cffec6"

FACTS = (
    ("minijam", "test gateway alias", "veka-27", "nalo-27"),
    ("minijam", "peer relay tag", "voro-63", "sena-63"),
    ("minijam", "RPC probe marker", "tiva-51", "mora-51"),
    ("jamscript", "compiler marker", "sena-41", "veka-41"),
    ("jamscript", "source map label", "nalo-82", "voro-82"),
    ("jamscript", "service runtime tag", "mora-34", "tiva-34"),
    ("minicells", "dataset snapshot ID", "voro-62", "sena-62"),
    ("minicells", "training run alias", "tiva-53", "mora-53"),
    ("minicells", "model checkpoint label", "nalo-78", "veka-78"),
)


@dataclass(frozen=True)
class Case:
    index: int
    domain: str
    label: str
    gold: str
    distractor: str

    @property
    def query(self):
        return f"\nQuestion: What is the current {self.label}?\nAnswer:"

    @property
    def record(self):
        return (f"[Cell: {self.domain}]\n[Record: {2 * self.index}]\n"
                f"The {self.label} for the current study is {self.gold}.")


CASES = tuple(Case(index, *fact) for index, fact in enumerate(FACTS))


def memory_text(case, domain=False):
    items = [other for other in CASES if other.domain == case.domain and other != case] if domain else []
    return "Memory:\n" + "\n".join(item.record for item in [*items, case])
