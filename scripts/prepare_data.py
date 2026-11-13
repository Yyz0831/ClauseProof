"""Prepare an auditable QASPER pilot using only Python's standard library.

No training, model calls, automatic relabelling, or test-set download.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import tarfile
import unicodedata
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
SHA256 = "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
SOURCE_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_source():
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "qasper-train-dev-v0.3.tgz"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        temporary.replace(archive)
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != SHA256:
        raise ValueError(f"Source checksum changed: {actual}. Inspect the source before proceeding.")
    with tarfile.open(archive, "r:gz") as handle:
        # Read only named JSON members; do not extract archive paths to disk.
        train = json.load(handle.extractfile("qasper-train-v0.3.json"))
        validation = json.load(handle.extractfile("qasper-dev-v0.3.json"))
    if set(train) & set(validation):
        raise ValueError("Official train/dev contain overlapping paper IDs.")
    return train, validation, actual


def paragraphs_for(paper):
    paragraphs = []
    if paper.get("abstract", "").strip():
        paragraphs.append({"section": "Abstract", "text": paper["abstract"]})
    for section in paper["full_text"]:
        for paragraph in section["paragraphs"]:
            if paragraph.strip():
                paragraphs.append({"section": section["section_name"], "text": paragraph})
    for i, paragraph in enumerate(paragraphs):
        paragraph["id"] = f"p{i:04d}"
    return paragraphs


def answer_text(answer):
    if answer.get("extractive_spans"):
        # Multiple extractive spans together form one reference answer.
        return ", ".join(answer["extractive_spans"])
    if answer.get("free_form_answer", "").strip():
        return answer["free_form_answer"].strip()
    if answer.get("yes_no") is not None:
        return "yes" if answer["yes_no"] else "no"
    return ""


def convert(papers, split, max_chars):
    rows, rejected = [], Counter()
    total_questions = sum(len(p["qas"]) for p in papers.values())
    for paper_id, paper in sorted(papers.items()):
        paragraphs = paragraphs_for(paper)
        context = f"Title: {paper['title']}\n\n" + "\n\n".join(
            f"[{p['id']}] {p['section']}\n{p['text']}" for p in paragraphs
        )
        lookup = {}
        for paragraph in paragraphs:
            lookup.setdefault(norm(paragraph["text"]), []).append(paragraph["id"])
        for question in paper["qas"]:
            annotations = [a["answer"] for a in question.get("answers", [])]
            if not annotations:
                rejected["no_annotations"] += 1
                continue
            flags = {a["unanswerable"] for a in annotations}
            if len(flags) != 1:
                rejected["answerability_disagreement"] += 1
                continue
            if len(context) > max_chars:
                rejected["full_context_over_character_budget"] += 1
                continue
            answerable = not annotations[0]["unanswerable"]
            references, evidence_sets = [], []
            if answerable:
                if any(e.strip().startswith("FLOAT SELECTED")
                       for a in annotations for e in a.get("evidence", [])):
                    rejected["annotated_figure_or_table_evidence"] += 1
                    continue
                yes_no = {a["yes_no"] for a in annotations if a.get("yes_no") is not None}
                if len(yes_no) > 1:
                    rejected["yes_no_disagreement"] += 1
                    continue
                for answer in annotations:
                    text = answer_text(answer)
                    evidence = answer.get("evidence", [])
                    if not text or not evidence or any(norm(e) not in lookup for e in evidence):
                        continue
                    ids = sorted({pid for e in evidence for pid in lookup[norm(e)]})
                    references.append(text)
                    evidence_sets.append(ids)
                if not references:
                    rejected["no_reference_with_matching_text_evidence"] += 1
                    continue
            else:
                references = ["The provided paper does not supply enough information to answer."]
                evidence_sets = [[]]
            rows.append({
                "id": f"{paper_id}:{question['question_id']}",
                "paper_id": paper_id,
                "question_id": question["question_id"],
                "source_split": split,
                "title": paper["title"],
                "question": question["question"],
                "context": context,
                "paragraphs": paragraphs,
                "answerable": answerable,
                "reference_answers": references,
                "reference_evidence_ids": evidence_sets,
                "original_answer_annotations": annotations,
                "review_status": "pending",
                "context_mode": "full_text_and_abstract_without_figures",
            })
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question IDs in {split}")
    stats = {
        "source_papers": len(papers), "source_questions": total_questions,
        "candidate_papers": len({r["paper_id"] for r in rows}),
        "candidate_questions": len(rows),
        "answerable": sum(r["answerable"] for r in rows),
        "unanswerable": sum(not r["answerable"] for r in rows),
        "rejected": dict(rejected),
    }
    if len(rows) + sum(rejected.values()) != total_questions:
        raise ValueError("Question accounting mismatch")
    return rows, stats


def select_pilot(rows, size, seed):
    rng = random.Random(seed)
    positive = [r for r in rows if r["answerable"]]
    negative = [r for r in rows if not r["answerable"]]
    rng.shuffle(positive)
    rng.shuffle(negative)
    # Aim for up to 30% naturally unanswerable questions; record realized counts.
    negative_count = min(len(negative), round(size * 0.3))
    pilot = negative[:negative_count] + positive[:size - negative_count]
    if len(pilot) < size:
        pilot.extend(negative[negative_count:negative_count + size - len(pilot)])
    rng.shuffle(pilot)
    return pilot


def review_text(rows):
    chunks = [
        "PaperProof：开发集候选样本人工复核（尚未运行模型）",
        "请检查：问题是否明确、正文是否足以回答、参考答案是否准确、证据是否完整。",
        "不可回答标签来自原始标注；正文很长时请结合完整 JSONL 核对。",
        "本文件展示前 20 条，不代表已经完成复核。",
    ]
    for i, row in enumerate(rows[:20], 1):
        chunks += ["\n" + "=" * 72, f"案例 {i} | {row['id']}", row["title"],
                   "问题：" + row["question"],
                   "原始标签：" + ("可回答" if row["answerable"] else "不可回答"),
                   "参考答案：" + json.dumps(row["reference_answers"], ensure_ascii=False),
                   "参考证据："]
        used = {pid for evidence in row["reference_evidence_ids"] for pid in evidence}
        if used:
            chunks += [f"[{p['id']}] {p['text']}" for p in row["paragraphs"] if p["id"] in used]
        else:
            chunks += ["无标注支持段落；请在 JSONL 中检查完整正文。"]
        chunks += ["人工判定：待填写；需要修正的问题：待填写"]
    return "\n".join(chunks) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-context-chars", type=int, default=24000)
    parser.add_argument("--pilot-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_context_chars <= 0 or args.pilot_size <= 0:
        parser.error("Context budget and pilot size must be positive")
    train, validation, checksum = load_source()
    processed = ROOT / "data" / "processed"
    reports = ROOT / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    train_rows, train_stats = convert(train, "train", args.max_context_chars)
    dev_rows, dev_stats = convert(validation, "validation", args.max_context_chars)
    pilot = select_pilot(dev_rows, args.pilot_size, args.seed)
    write_jsonl(processed / "train_candidates.jsonl", train_rows)
    write_jsonl(processed / "dev_candidates.jsonl", dev_rows)
    write_jsonl(processed / "pilot_dev.jsonl", pilot)
    (reports / "pilot_review.txt").write_text(review_text(pilot), encoding="utf-8")
    summary = {
        "train": train_stats, "validation": dev_stats,
        "pilot": {"questions": len(pilot), "papers": len({r['paper_id'] for r in pilot}),
                  "answerable": sum(r["answerable"] for r in pilot),
                  "unanswerable": sum(not r["answerable"] for r in pilot),
                  "review_status": "pending", "sampling": "label_stratified; not natural prevalence"},
        "model_inference_run": False,
        "notes": ["Candidate subset only; not the full QASPER benchmark.",
                  "True token lengths must be checked before inference; no silent truncation.",
                  "No official test split downloaded by this script."],
    }
    write_json(reports / "data_summary.json", summary)
    manifest = {
        "dataset": "allenai/qasper", "version": "v0.3", "source_url": URL,
        "hf_loader_revision": SOURCE_REVISION, "raw_archive_sha256": checksum,
        "downloaded_splits": ["train", "validation"], "seed": args.seed,
        "max_context_characters": args.max_context_chars,
        "pilot_question_ids": [r["id"] for r in pilot],
        "license_as_declared_by_dataset_card": "CC-BY-4.0",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(processed.glob("*.jsonl"))},
    }
    write_json(reports / "data_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
