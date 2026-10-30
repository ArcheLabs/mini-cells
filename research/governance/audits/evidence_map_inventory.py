"""Read-only research-source inventory; writes governance indexes, never experiment files."""

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GOV = ROOT / "research/governance"
REF = "origin/main"


def main():
    sha = subprocess.check_output(["git", "rev-parse", REF], cwd=ROOT, text=True).strip()
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", sha, "--", "research"], cwd=ROOT, text=True
    ).splitlines()
    # Use git archive to guarantee exact main bytes, not the feature worktree.
    import io
    import tarfile

    blob = subprocess.check_output(["git", "archive", sha, "research"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        contents = {
            m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()
        }
    units = defaultdict(list)
    declarations = defaultdict(set)
    docs = {}
    paths = []
    references = set()
    for name in names:
        raw = contents.get(name, b"")
        paths.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        docs[name] = text
        match = re.match(r"(research/projects/[^/]+/experiments/[^/]+)", name)
        if not match:
            match = re.match(
                r"(research/(?:governance/validations|projects/[^/]+/context/[^/]+/validation)/[^/]+)",
                name,
            )
        if match:
            units[match[1]].append(name)
        references.update(re.findall(r"research/[A-Za-z0-9_.\-/]+\.(?:md|json|yaml|py)", text))
        ids = []
        for key, value in re.findall(r'"(experiment_id|experiment)"\s*:\s*"([^"\n]{1,180})"', text):
            if not any(x in value for x in ("{", "}", "UNKNOWN")):
                ids.append(value)
        ids += re.findall(
            r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?(?:Experiment ID|实验 ID|实验编号)\s*(?:\*\*)?\s*[:：]\s*`?([^`\r\n]+)",
            text,
        )
        for value in ids:
            declarations[value.strip()].add(name)
    records = []
    for directory, files in sorted(units.items()):
        texts = [p for p in files if p.endswith((".md", ".json", ".yaml", ".yml"))]
        protocol = [
            p
            for p in texts
            if re.search(r"(protocol|prereg|spec|registration)", Path(p).name, re.IGNORECASE)
        ]
        results = [
            p for p in texts if re.search(r"(results?|final_report|decision)", Path(p).name, re.IGNORECASE)
        ]
        readme = next((p for p in files if p == directory + "/README.md"), None)
        title = next(
            (
                l.lstrip("# ").strip()
                for l in docs.get(readme, "").splitlines()
                if l.startswith("# ")
            ),
            "UNKNOWN",
        )
        own_ids = sorted(
            k for k, v in declarations.items() if any(p.startswith(directory + "/") for p in v)
        )
        rawstatuses = []
        for p in results:
            for status in re.findall(
                r'(?im)(?:"(?:status|classification)"\s*:\s*"|^- (?:Status|Classification): `)([^"`\r\n]+)',
                docs[p],
            ):
                rawstatuses.append((p, status))
        records.append(
            {
                "directory": directory,
                "ids": own_ids,
                "name": title,
                "protocols": protocol,
                "results": results,
                "raw_statuses": rawstatuses,
                "registration": "RETROSPECTIVE",
                "type": "UNKNOWN",
                "outcome": "UNKNOWN",
                "date": "UNKNOWN",
                "files": files,
            }
        )
    reviewed = {
        "core-005-subspace-certified-mitosis": ("FORMAL_EXPERIMENT", "PASS"),
        "core-009b1-carrier-causal-sufficiency": ("FORMAL_EXPERIMENT", "PASS"),
        "core-009c-sparse-local-effect-geometry": ("EXPLORATORY_EXPERIMENT", "FAIL"),
        "core-006-real-representation-continual-plasticity": ("FORMAL_EXPERIMENT", "FAIL"),
    }
    for r in records:
        if r["directory"].split("/")[-1] in reviewed:
            r["type"], r["outcome"] = reviewed[r["directory"].split("/")[-1]]
    covered = {p for r in records for p in r["files"]}
    unowned = {k: sorted(v) for k, v in declarations.items() if not any(p in covered for p in v)}
    missing = sorted(p for p in references if p not in contents)

    def link(p):
        return f"[{p}](../../../{p})"

    lines = [
        "# Evidence Map 历史盘点附录",
        "",
        f"快照：`{sha}`。来自当前 main 的全部 research 文件；不是未合并分支的成果。",
        "",
        "计数单位是已发现的实验/验证工作目录，不能将其直接当作互相独立的科学实验数。一个目录可能包含多个注册 ID 或后续阶段；ID 逐字保留在下表。迁移登记均为 RETROSPECTIVE，不认证原实验的事前注册时序。UNKNOWN 不等于未执行；没有找到协议/结果文件名也不证明其不存在。",
        "",
        "Outcome 与 Type 需按原始结果逐阶段核定。未完成核定者明确 UNKNOWN；原始状态标签只作索引，不自动解释为 PASS/FAIL。细粒度身份声明、引用缺口、路径清单和覆盖限制见完整性审计。",
        "",
        "|工作目录 / 原始 ID|Name / Research Line|Registration / Type / Date / Outcome|Protocol / Result|Related / Finding / Relevance|",
        "|---|---|---|---|---|",
    ]
    for r in records:
        line = (
            r["directory"].split("/")[2]
            if "/projects/" in r["directory"]
            else "cross-project-validations"
        )
        ids = "; ".join(r["ids"]) or "UNKNOWN（未提取到原 ID；不补造）"
        summary = "; ".join(sorted({s for _, s in r["raw_statuses"]}))[:650] or "NOT_RECORDED"
        proto = "<br>".join(link(p) for p in r["protocols"]) or "UNKNOWN"
        res = "<br>".join(link(p) for p in r["results"]) or "UNKNOWN"
        cells = [
            f"{link(r['directory'])}<br>{ids}",
            f"{r['name']} / {line}",
            f"{r['registration']} / {r['type']} / UNKNOWN / {r['outcome']}",
            f"P: {proto}<br>R: {res}",
            f"关系 UNKNOWN；原状态标签：{summary}；意义待逐阶段审阅",
        ]
        lines.append(
            "|" + "|".join(c.replace("|", "&#124;").replace("\n", " ") for c in cells) + "|"
        )
    lines += [
        "",
        "## 尚未归属目录的身份声明",
        "",
        "保留原 ID 与声明位置；不认定这些就是独立的新实验。",
        "",
    ]
    for identity, sources in sorted(unowned.items()):
        lines.append(
            "- `"
            + identity.replace("`", "")
            + "`："
            + ", ".join(link(x) for x in sources)
            + "；名称/Type/日期/Outcome/关系 UNKNOWN，Registration=RETROSPECTIVE（本次迁移）。"
        )
    (GOV / "audits/EVIDENCE_MAP_INVENTORY.md").write_text("\n".join(lines) + "\n")
    report = {
        "source_main_commit": sha,
        "source_tree_sha256": hashlib.sha256(
            json.dumps(paths, sort_keys=True).encode()
        ).hexdigest(),
        "research_files": len(names),
        "decoded_text_files": len(docs),
        "discovered_work_directories": len(records),
        "declared_original_id_strings": len(declarations),
        "registration_counts": dict(Counter(r["registration"] for r in records)),
        "type_counts": dict(Counter(r["type"] for r in records)),
        "outcome_counts": dict(Counter(r["outcome"] for r in records)),
        "directories_without_protocol_candidate": sum(not r["protocols"] for r in records),
        "directories_without_result_candidate": sum(not r["results"] for r in records),
        "unowned_id_declarations": unowned,
        "unresolved_research_reference_paths": missing,
        "source_files": paths,
        "records": records,
        "coverage": "ALL_TRACKED_RESEARCH_FILES_AND_LITERAL_RESEARCH_REFERENCES; NOT_FULL_SEMANTIC_CERTIFICATION",
        "completion": "DISCOVERY_INDEX_COMPLETE_FOR_RULES; HUMAN_SEMANTIC_HISTORY_REVIEW_INCOMPLETE",
        "limits": [
            "Directory units are not deduplicated experiment identities",
            "Non-experiment/context studies may lack extractable IDs; unowned declarations preserved",
            "All UNKNOWN outcomes/types require source-level phase review; no fabricated final statistics",
            "Missing literal paths may be historical relocations, not missing evidence",
            "No preregistration timing certified",
            "No model or research execution",
        ],
    }
    (GOV / "audits/EVIDENCE_MAP_COMPLETENESS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                in [
                    "source_main_commit",
                    "research_files",
                    "discovered_work_directories",
                    "declared_original_id_strings",
                    "directories_without_protocol_candidate",
                    "directories_without_result_candidate",
                    "completion",
                ]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
