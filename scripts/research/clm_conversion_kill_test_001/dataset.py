from __future__ import annotations

from dataclasses import dataclass

DATASET_VERSION = "clm-conversion-fictional-knowledge-v1"

ENTITIES = (
    "Zorven",
    "Talric",
    "Mirel",
    "Kavon",
    "Nerith",
    "Velcor",
    "Praxen",
    "Sulmar",
    "Iveron",
    "Daxil",
    "Rovek",
    "Cymra",
)
PROTOCOLS = ("QX-17", "LM-42", "VR-08", "PN-63", "TK-51", "HF-29")
REGIONS = ("Rho-Delta", "Sigma-North", "Tau-Vale")


@dataclass(frozen=True)
class Fact:
    concept_id: str
    subject: str
    value: str
    kind: str


def entity_facts() -> list[Fact]:
    return [
        Fact(f"entity.{entity}", entity, PROTOCOLS[index // 2], "entity_protocol")
        for index, entity in enumerate(ENTITIES)
    ]


def protocol_facts() -> list[Fact]:
    return [
        Fact(
            f"protocol.{protocol}",
            protocol,
            REGIONS[index // 2],
            "protocol_region",
        )
        for index, protocol in enumerate(PROTOCOLS)
    ]


def _row(
    row_id: str,
    concept_id: str,
    question: str,
    answer: str,
    family: str,
) -> dict[str, str]:
    return {
        "id": row_id,
        "concept_id": concept_id,
        "question": question,
        "answer": answer,
        "family": family,
    }


def training_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fact in entity_facts():
        rows.extend(
            [
                _row(
                    f"train.{fact.concept_id}.a",
                    fact.concept_id,
                    f"Which protocol code is assigned to {fact.subject}?",
                    fact.value,
                    "entity_direct",
                ),
                _row(
                    f"train.{fact.concept_id}.b",
                    fact.concept_id,
                    f"Give the protocol designation used by {fact.subject}.",
                    fact.value,
                    "entity_direct",
                ),
            ]
        )
    for fact in protocol_facts():
        rows.extend(
            [
                _row(
                    f"train.{fact.concept_id}.a",
                    fact.concept_id,
                    f"Which region code is associated with protocol {fact.subject}?",
                    fact.value,
                    "protocol_direct",
                ),
                _row(
                    f"train.{fact.concept_id}.b",
                    fact.concept_id,
                    f"Name the region designation for {fact.subject}.",
                    fact.value,
                    "protocol_direct",
                ),
            ]
        )
    return rows


def formation_validation() -> list[dict[str, str]]:
    return [
        _row(
            f"validation.{fact.concept_id}",
            fact.concept_id,
            f"For validation, report the registry protocol assigned to {fact.subject}.",
            fact.value,
            "validation_direct",
        )
        for fact in entity_facts()
    ]


def formation_evaluation() -> dict[str, list[dict[str, str]]]:
    direct: list[dict[str, str]] = []
    negation: list[dict[str, str]] = []
    relation: list[dict[str, str]] = []
    routing: list[dict[str, str]] = []
    protocol_to_region = {fact.subject: fact.value for fact in protocol_facts()}
    facts = entity_facts()
    for fact in facts:
        wrong = PROTOCOLS[(PROTOCOLS.index(fact.value) + 2) % len(PROTOCOLS)]
        direct.extend(
            [
                _row(
                    f"eval.{fact.concept_id}.direct.a",
                    fact.concept_id,
                    f"For {fact.subject}, what is the designated protocol?",
                    fact.value,
                    "direct",
                ),
                _row(
                    f"eval.{fact.concept_id}.direct.b",
                    fact.concept_id,
                    f"Identify {fact.subject}'s protocol code.",
                    fact.value,
                    "direct",
                ),
            ]
        )
        negation.append(
            _row(
                f"eval.{fact.concept_id}.negation",
                fact.concept_id,
                f"The claim that {fact.subject} uses {wrong} is false. "
                f"What protocol does {fact.subject} actually use?",
                fact.value,
                "negation",
            )
        )
        relation.append(
            _row(
                f"eval.{fact.concept_id}.relation",
                fact.concept_id,
                f"Following {fact.subject}'s protocol assignment, which region "
                "designation is ultimately associated with it?",
                protocol_to_region[fact.value],
                "relation",
            )
        )
        routing.extend(
            [
                _row(
                    f"route.{fact.concept_id}.a",
                    fact.concept_id,
                    f"Which protocol code belongs to {fact.subject}?",
                    fact.value,
                    "routing",
                ),
                _row(
                    f"route.{fact.concept_id}.b",
                    fact.concept_id,
                    f"What protocol is {fact.subject} associated with?",
                    fact.value,
                    "routing",
                ),
                _row(
                    f"route.{fact.concept_id}.c",
                    fact.concept_id,
                    f"State the protocol designation for {fact.subject}.",
                    fact.value,
                    "routing",
                ),
                _row(
                    f"route.{fact.concept_id}.d",
                    fact.concept_id,
                    f"If asked for {fact.subject}'s protocol code, what is it?",
                    fact.value,
                    "routing",
                ),
            ]
        )
    return {
        "direct": direct,
        "negation": negation,
        "relation": relation,
        "routing": routing,
    }


def rewrite_rows(
    entity: str,
    new_protocol: str,
    *,
    prefix: str,
) -> dict[str, list[dict[str, str]]]:
    concept = f"entity.{entity}"
    wrong = next(value for value in PROTOCOLS if value != new_protocol)
    train = [
        _row(
            f"{prefix}.train.a",
            concept,
            f"After the revision, which protocol is assigned to {entity}?",
            new_protocol,
            "rewrite_train",
        ),
        _row(
            f"{prefix}.train.b",
            concept,
            f"Use the revised registry: give {entity}'s protocol designation.",
            new_protocol,
            "rewrite_train",
        ),
    ]
    evaluation = [
        _row(
            f"{prefix}.eval.a",
            concept,
            f"Under the revised registry, what protocol code belongs to {entity}?",
            new_protocol,
            "rewrite_eval",
        ),
        _row(
            f"{prefix}.eval.b",
            concept,
            f"The revised claim that {entity} uses {wrong} is false. "
            f"Which protocol does {entity} use instead?",
            new_protocol,
            "rewrite_eval",
        ),
    ]
    return {"train": train, "evaluation": evaluation}


def contextual_conflict_rows(
    entity: str,
    old_protocol: str,
    new_protocol: str,
) -> dict[str, list[dict[str, str]]]:
    concept = f"entity.{entity}"
    alpha = [
        _row(
            "growth.alpha.eval.a",
            concept,
            f"In archive Alpha, which protocol is recorded for {entity}?",
            old_protocol,
            "growth_alpha",
        ),
        _row(
            "growth.alpha.eval.b",
            concept,
            f"Consult archive Alpha: give {entity}'s protocol code.",
            old_protocol,
            "growth_alpha",
        ),
    ]
    beta_train = [
        _row(
            "growth.beta.train.a",
            concept,
            f"In archive Beta, which protocol is recorded for {entity}?",
            new_protocol,
            "growth_beta",
        ),
        _row(
            "growth.beta.train.b",
            concept,
            f"Consult archive Beta: give {entity}'s protocol code.",
            new_protocol,
            "growth_beta",
        ),
    ]
    beta_eval = [
        _row(
            "growth.beta.eval.a",
            concept,
            f"According to archive Beta, what protocol designation belongs to {entity}?",
            new_protocol,
            "growth_beta",
        ),
        _row(
            "growth.beta.eval.b",
            concept,
            f"Archive Beta rejects the old assignment. Which protocol is valid for {entity}?",
            new_protocol,
            "growth_beta",
        ),
    ]
    return {"alpha": alpha, "beta_train": beta_train, "beta_eval": beta_eval}


def calibration_prompts() -> tuple[str, ...]:
    return (
        "The capital of France is",
        "Water freezes at a temperature of",
        "Two plus two equals",
        "A triangle has",
        "The opposite of hot is",
        "The Pacific Ocean is",
        "The first month of the year is",
        "Plants use sunlight during",
        "A kilogram is a unit of",
        "The chemical symbol for oxygen is",
        "The color made by mixing blue and yellow is",
        "The Earth orbits",
        "A week contains",
        "The past tense of walk is",
        "The largest planet in the Solar System is",
        "A byte commonly contains",
    )
