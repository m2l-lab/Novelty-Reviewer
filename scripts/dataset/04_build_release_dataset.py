"""
Build the public release version of the novelty dataset.

Merges, per paper (keyed by OpenReview forum id):
  - official reviews                              (openreviewer_dataset/dataset.zstd.parquet, neurips_2025.parquet)
  - title and abstract                            (OpenReview API via script 03; fallback: extracted from the paper text)
  - human-review novelty extractions              (Dataset/Human_annotated_data/*.parquet; none for NeurIPS 2025)
  - LLM-generated novelty summary and score       (Dataset/llm_summarized_dataset/*.parquet + NeurIPS 2025 summaries)
  - train/test split and evaluation subset        (the split used for fine-tuning and evaluation in the paper)

The full paper text is NOT released. For privacy, every released text field of a paper is checked against that
paper's author names (from script 03) and against e-mail addresses; matches are replaced by [REDACTED]. All
redactions are listed in --privacy_report (keep it outside the release folder: it contains the matched names).

Outputs (in --out_dir):
  data/train-0000X-of-0000N.parquet, data/test-00000-of-00001.parquet   one row per paper
  baseline_reviews/test-00000-of-00001.parquet                          model-generated reviews for the test papers
  stats.json

Usage:
  python scripts/dataset/04_build_release_dataset.py \
      --openreviewer_dir <raw crawl: dataset.zstd.parquet, dataset_test.zstd.parquet> \
      --novelty_dir <stage-1/2 outputs: Human_annotated_data/, llm_summarized_dataset/> \
      --metadata openreview_metadata.parquet \
      --privacy_report privacy_redactions.csv \
      --neurips2025_reviews neurips_2025.parquet \
      --neurips2025_summaries <NeurIPS 2025 stage-2 outputs> \
      --train_ids train_ids.parquet --test_ids test_ids.parquet --eval_ids eval_ids.parquet \
      --out_dir novelty_dataset_release
"""

import argparse
import glob
import json
import math
import os
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from novelty_reviewer.parsing import parse_aggregated_assessment

BASELINE_MODELS = [
    "Llama-OpenReviewer-8B",
    "Llama-3.1-70B-Instruct",
    "Llama-3.1-8B-Instruct",
    "gpt-4o-2024-11-20",
    "claude-3.5-sonnet",
]

# Review section header -> output field. Values are the leading integer of the section, e.g. "6: marginally above ...".
REVIEW_SCORE_FIELDS = {
    "Rating": "rating",
    "Recommendation": "rating",
    "Confidence": "confidence",
    "Soundness": "soundness",
    "Presentation": "presentation",
    "Contribution": "contribution",
    "Correctness": "correctness",
    "Technical Novelty And Significance": "technical_novelty_and_significance",
    "Empirical Novelty And Significance": "empirical_novelty_and_significance",
}
# NeurIPS 2025 reviews are plain text with upper-case headers ("ORIGINALITY:\n2") and no overall rating.
NEURIPS2025_SCORE_FIELDS = {
    "ORIGINALITY": "originality",
    "SIGNIFICANCE": "significance",
    "QUALITY": "quality",
    "CLARITY": "clarity",
}
REVIEW_SCORE_COLUMNS = sorted(set(REVIEW_SCORE_FIELDS.values()) | set(NEURIPS2025_SCORE_FIELDS.values()))

EXTRACTION_KEYS = [
    "novelty_excerpts",
    "contribution_scope_excerpts",
    "contribution_scope_statements",
    "implicit_novelty_assessments",
]


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------

def parse_review(text):
    """Split a markdown review into sections and pull out the numeric scores."""
    sections = {}
    for m in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        sections[m.group(1).strip()] = m.group(2).strip()
    scores = {c: None for c in REVIEW_SCORE_COLUMNS}
    for header, field in REVIEW_SCORE_FIELDS.items():
        if header in sections:
            n = re.match(r"\s*(\d+)", sections[header])
            if n:
                scores[field] = int(n.group(1))
    return scores


def parse_review_neurips2025(text):
    scores = {c: None for c in REVIEW_SCORE_COLUMNS}
    for header, field in NEURIPS2025_SCORE_FIELDS.items():
        n = re.search(rf"^{header}:\s*(\d+)", text, re.M)
        if n:
            scores[field] = int(n.group(1))
    return scores


def parse_venue(or_venue):
    # "ICLR.cc/2024/Conference" -> ("ICLR", 2024)
    conf, year = or_venue.split("/")[:2]
    return conf.split(".")[0], int(year)


def extract_title(paper_text):
    # First markdown heading near the top; ICLR 2025 texts start with a "000 001 ... 053" line-number ruler.
    for m in re.finditer(r"^#\s+(.+)$", paper_text[:3000], re.M):
        title = m.group(1).strip()
        if title.lower() not in {"abstract", "introduction", "1 introduction"}:
            # Reject headings that swallowed the author block (names, e-mails).
            return title if len(title) <= 250 and not EMAIL.search(title) else None
    return None


def extract_abstract(paper_text):
    # Text between the "Abstract" heading and the next heading. The PDF conversion sometimes moves the author
    # block below the heading; reject anything containing an e-mail address or too long to be an abstract.
    m = re.search(r"^#+\s*Abstract\s*$\n(.*?)(?=^#)", paper_text[:30000], re.M | re.I | re.S)
    abstract = m.group(1).strip() if m else ""
    return abstract if abstract and len(abstract) <= 4000 and not EMAIL.search(abstract) else None


# ---------------------------------------------------------------------------
# Privacy: author names and e-mail addresses
# ---------------------------------------------------------------------------

EMAIL = re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
REDACTED = "[REDACTED]"


ORGANIZATION = re.compile(r"\b(Laborator\w*|Consortium|Collaboration|Team|Group|Institute|University|Lab)\b", re.I)


def author_pattern(authors):
    """Whole-word match of an author's name: full name ("John A. Smith"), reversed ("Smith John", "Smith, John")
    and initialised ("J. Smith", "J. A. Smith"). Organization authors (e.g. a consortium) are skipped.
    Bare surnames are deliberately not matched (they would hit unrelated citations such as "Chen et al.")."""
    alts = set()
    for a in authors:
        parts = (a or "").replace(",", " ").split()
        if len(parts) < 2 or ORGANIZATION.search(a):
            continue
        first, last = re.escape(parts[0]), re.escape(parts[-1])
        alts.add(r"(?i:" + r"\s+".join(map(re.escape, parts)) + ")")
        alts.add(rf"(?i:{last},?\s+{first})")
        # Initialised forms are case-sensitive so that e.g. "a. chen" in running text is not matched.
        alts.add(rf"{re.escape(parts[0][0].upper())}\.\s*(?:[A-Z]\.\s*)*{last}")
    if not alts:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(sorted(alts, key=len, reverse=True)) + r")(?!\w)")


URL = re.compile(r"(?:https?://|www\.)[^\s)\]}>\"']+"
                 r"|\b(?:github|gitlab|huggingface|bitbucket|gitee)\.(?:com|co|io)/[^\s)\]}>\"']+"
                 r"|\b[\w-]+\.github\.io[^\s)\]}>\"']*", re.I)
URL_REDACTED = "[URL]"


def author_tokens(authors):
    """Lower-cased name parts (>= 4 letters) used to spot links to the authors' own pages, e.g. github.com/<name>."""
    return {t.lower() for a in authors if a and not ORGANIZATION.search(a)
            for t in re.split(r"[\s.,-]+", a) if len(t) >= 4}


class Redactor:
    def __init__(self):
        self.log = []

    def _log(self, paper_id, field, kind, m, text):
        self.log.append({"paper_id": paper_id, "field": field, "kind": kind, "match": m.group(0),
                         "context": text[max(0, m.start() - 100):m.end() + 100]})

    def __call__(self, text, paper_id, field, pattern, tokens=frozenset(), all_urls=False):
        """Redact author names and e-mails; redact links that are all_urls or contain an author name part."""
        if not isinstance(text, str):
            return None
        if not text:
            return text
        for kind, pat in [("author_name", pattern), ("email", EMAIL)]:
            if pat is None:
                continue
            for m in pat.finditer(text):
                self._log(paper_id, field, kind, m, text)
            text = pat.sub(REDACTED, text)

        def url(m):
            norm = re.sub(r"[^a-z]", "", m.group(0).lower())
            if all_urls or any(t in norm for t in tokens):
                self._log(paper_id, field, "url", m, text)
                return URL_REDACTED
            return m.group(0)

        return URL.sub(url, text)


# ---------------------------------------------------------------------------
# Human novelty extractions (LLM-written JSON, ~5% malformed)
# ---------------------------------------------------------------------------

_BAD_ESCAPE = re.compile(r'\\(\\|"|/|u[0-9a-fA-F]{4})|\\')


def _escape_stray_backslashes(t):
    # Keep valid \\, \", \/, \uXXXX; double every other backslash (mostly LaTeX such as \tilde, \mathcal).
    return _BAD_ESCAPE.sub(lambda m: m.group(0) if m.group(1) else "\\\\", t)


def _line_heuristic(t):
    """Fallback for JSON broken by unescaped quotes: one list item per line, keys as `"key": [`."""
    out, key = {}, None
    for line in t.splitlines():
        s = line.strip()
        k = re.match(r'"(\w+)"\s*:\s*\[', s)
        if k:
            key = k.group(1)
            out.setdefault(key, [])
            s = s[k.end():].strip()
            if not s:
                continue
        if key is None:
            continue
        s = re.sub(r"\]\s*,?\s*\}?\s*$", "", s).rstrip().rstrip(",").strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            out[key].append(s[1:-1].replace('\\"', '"'))
    return out if any(out.values()) else None


def parse_extraction(t):
    """Returns (dict, status) with status in {ok, repaired, repaired_truncated, recovered_heuristic, failed}."""
    try:
        return json.loads(t), "ok"
    except json.JSONDecodeError:
        pass
    fixed = _escape_stray_backslashes(t).rstrip()
    # "" -> bad escapes/control chars; "}" / "]}" -> missing closers; '"]}' -> generation cut off mid-item.
    for suffix, status in [("", "repaired"), ("}", "repaired"), ("]}", "repaired"), ('"]}', "repaired_truncated")]:
        try:
            return json.loads(fixed + suffix, strict=False), status
        except json.JSONDecodeError:
            pass
    h = _line_heuristic(t)
    return (h, "recovered_heuristic") if h else ({}, "failed")


def _as_str_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in v]


# ---------------------------------------------------------------------------
# LLM novelty summaries
# ---------------------------------------------------------------------------

# Score (None outside -1..2), justification and detailed assessment; see novelty_reviewer.parsing.
parse_summary = parse_aggregated_assessment


# ---------------------------------------------------------------------------

def read_dir(pattern):
    files = sorted(glob.glob(pattern))
    assert files, f"no files match {pattern}"
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def write_shards(df, schema, path_prefix, split, max_rows):
    n = max(1, math.ceil(len(df) / max_rows))
    for i in range(n):
        part = df.iloc[i * max_rows:(i + 1) * max_rows]
        table = pa.Table.from_pandas(part, schema=schema, preserve_index=False)
        pq.write_table(table, f"{path_prefix}/{split}-{i:05d}-of-{n:05d}.parquet", compression="zstd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openreviewer_dir", required=True)
    ap.add_argument("--novelty_dir", required=True)
    ap.add_argument("--metadata", required=True, help="output of 03_fetch_openreview_metadata.py")
    ap.add_argument("--privacy_report", required=True, help="CSV listing every redaction; keep outside out_dir")
    ap.add_argument("--neurips2025_reviews", required=True, help="neurips_2025.parquet (paper_id, paper, reviews)")
    ap.add_argument("--neurips2025_summaries", required=True, help="directory with NeurIPS 2025 novelty summaries")
    ap.add_argument("--train_ids", required=True, help="parquet with the paper_ids used for fine-tuning")
    ap.add_argument("--test_ids", required=True, help="parquet with the held-out paper_ids")
    ap.add_argument("--eval_ids", required=True, help="parquet with the paper_ids of the evaluation subset")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rows_per_shard", type=int, default=5000)
    args = ap.parse_args()

    # --- raw reviews: one row per (paper, review) ---
    raw = pq.read_table(os.path.join(args.openreviewer_dir, "dataset.zstd.parquet")).to_pandas()
    g = raw.groupby("id", sort=False)
    assert (g.paper_text.nunique() == 1).all() and (g.or_venue.nunique() == 1).all()
    assert (g.review_fields.nunique() == 1).all()

    scores = pd.DataFrame([parse_review(r) for r in raw.review], index=raw.index)
    raw = pd.concat([raw, scores], axis=1)
    raw["review_struct"] = [
        {"review_text": r, **{c: (None if pd.isna(v) else int(v)) for c, v in zip(REVIEW_SCORE_COLUMNS, vals)}}
        for r, vals in zip(raw.review, raw[REVIEW_SCORE_COLUMNS].itertuples(index=False))
    ]
    papers = g.agg(
        or_venue=("or_venue", "first"),
        paper_text=("paper_text", "first"),
        review_template=("review_fields", "first"),
    )
    papers["reviews"] = raw.groupby("id", sort=False).review_struct.agg(list)
    papers = papers.reset_index().rename(columns={"id": "paper_id"})
    papers[["venue", "year"]] = pd.DataFrame(papers.or_venue.map(parse_venue).tolist(), index=papers.index)

    # --- NeurIPS 2025: separate crawl with a different review format and no review form template ---
    n25 = pd.read_parquet(args.neurips2025_reviews, columns=["paper_id", "paper", "reviews"])
    n25 = pd.DataFrame({
        "paper_id": n25.paper_id, "or_venue": "NeurIPS.cc/2025/Conference", "paper_text": n25.paper,
        "review_template": None, "venue": "NeurIPS", "year": 2025,
        "reviews": [[{"review_text": r, **parse_review_neurips2025(r)} for r in rs] for rs in n25.reviews],
    })
    assert not set(n25.paper_id) & set(papers.paper_id)
    papers = pd.concat([papers, n25], ignore_index=True)
    papers["num_reviews"] = papers.reviews.map(len)

    # --- title / abstract / authors: OpenReview first, paper text as fallback; paper text is dropped afterwards ---
    meta = pd.read_parquet(args.metadata, columns=["paper_id", "title", "abstract", "authors"])
    meta = meta.drop_duplicates("paper_id").rename(columns={"title": "or_title", "abstract": "or_abstract"})
    papers = papers.merge(meta, on="paper_id", how="left", validate="1:1")
    papers["authors"] = papers.authors.map(lambda a: [] if a is None or isinstance(a, float) else list(a))
    clean = lambda s: " ".join(s.split()) if isinstance(s, str) and s.strip() else None
    or_title, or_abstract = papers.or_title.map(clean), papers.or_abstract.map(str.strip, na_action="ignore")
    or_abstract = or_abstract.where(or_abstract.str.len() > 0)
    papers["title"] = or_title.fillna(papers.paper_text.map(extract_title))
    papers["abstract"] = or_abstract.fillna(papers.paper_text.map(extract_abstract))
    source_counts = {
        "title_from_openreview": int(or_title.notna().sum()),
        "title_from_paper_text": int((or_title.isna() & papers.title.notna()).sum()),
        "abstract_from_openreview": int(or_abstract.notna().sum()),
        "abstract_from_paper_text": int((or_abstract.isna() & papers.abstract.notna()).sum()),
        "papers_with_author_list": int((papers.authors.map(len) > 0).sum()),
    }
    papers = papers.drop(columns=["paper_text", "or_title", "or_abstract"])

    # --- split: the train/test split used for fine-tuning and evaluation in the paper ---
    # NeurIPS 2025 was collected after the experiments and is released as additional training data.
    train_ids = set(pd.read_parquet(args.train_ids, columns=["paper_id"]).paper_id)
    test_ids = set(pd.read_parquet(args.test_ids, columns=["paper_id"]).paper_id)
    eval_ids = set(pd.read_parquet(args.eval_ids, columns=["paper_id"]).paper_id)
    assert not train_ids & test_ids and eval_ids <= test_ids
    assert train_ids | test_ids == set(papers.paper_id) - set(n25.paper_id), "split does not cover 2022-2025 papers"
    papers["split"] = papers.paper_id.map(lambda i: "test" if i in test_ids else "train")
    papers["in_eval_subset"] = papers.paper_id.isin(eval_ids)

    # --- human novelty extractions (not available for NeurIPS 2025) ---
    hum = read_dir(os.path.join(args.novelty_dir, "Human_annotated_data", "*.parquet"))
    assert hum.paper_id.is_unique
    parsed = hum.extracted_novelty_statements.map(parse_extraction)
    hum["novelty_extraction_parse_status"] = parsed.map(lambda x: x[1])
    for k in EXTRACTION_KEYS:
        hum[k] = parsed.map(lambda x: _as_str_list(x[0].get(k)))
    hum = hum.rename(columns={"extracted_novelty_statements": "novelty_extraction_raw"})

    # --- LLM novelty summaries ---
    llm = pd.concat([read_dir(os.path.join(args.novelty_dir, "llm_summarized_dataset", "*.parquet")),
                     read_dir(os.path.join(args.neurips2025_summaries, "*.parquet"))[["paper_id", "generated_text"]]],
                    ignore_index=True)
    assert llm.paper_id.is_unique
    llm = pd.concat([llm, pd.DataFrame(llm.generated_text.map(parse_summary).tolist(), index=llm.index)], axis=1)
    llm = llm.rename(columns={"generated_text": "novelty_summary"})

    # --- merge ---
    assert set(hum.paper_id) <= set(papers.paper_id) and set(llm.paper_id) == set(papers.paper_id)
    df = papers.merge(hum, on="paper_id", how="left", validate="1:1").merge(llm, on="paper_id", validate="1:1")
    no_extraction = df.novelty_extraction_parse_status.isna()
    df.loc[no_extraction, "novelty_extraction_parse_status"] = "not_extracted"
    for k in EXTRACTION_KEYS:
        df[k] = [v if isinstance(v, list) else [] for v in df[k]]
    df = df.sort_values(["venue", "year", "paper_id"]).reset_index(drop=True)

    # --- privacy: redact the paper's own author names, e-mail addresses and links to the authors' pages from every
    # released text field. Abstracts lose all links (code/project pages there are mostly the authors' own). ---
    redact = Redactor()
    patterns = dict(zip(df.paper_id, df.authors.map(author_pattern)))
    tokens = dict(zip(df.paper_id, df.authors.map(author_tokens)))
    for col in ["title", "abstract", "novelty_extraction_raw", "novelty_score_justification",
                "novelty_detailed_assessment", "novelty_summary"]:
        df[col] = [redact(t, pid, col, patterns[pid], tokens[pid], all_urls=(col == "abstract"))
                   for t, pid in zip(df[col], df.paper_id)]
    df["novelty_excerpts"] = [[redact(t, pid, "novelty_excerpts", patterns[pid], tokens[pid]) for t in lst]
                              for lst, pid in zip(df.novelty_excerpts, df.paper_id)]
    df["reviews"] = [
        [{**r, "review_text": redact(r["review_text"], pid, "reviews", patterns[pid], tokens[pid])} for r in rs]
        for rs, pid in zip(df.reviews, df.paper_id)
    ]

    columns = [
        "paper_id", "title", "venue", "year", "split", "in_eval_subset",
        "abstract",
        "num_reviews", "reviews", "review_template",
        "novelty_excerpts", "novelty_extraction_raw",
        "novelty_score", "novelty_score_justification", "novelty_detailed_assessment", "novelty_summary",
    ]
    # The other extraction lists and the parse status are internal; the status is only reported in stats.json.
    parse_status = df.novelty_extraction_parse_status
    df = df[columns]

    review_type = pa.struct(
        [("review_text", pa.string())] + [(c, pa.int8()) for c in REVIEW_SCORE_COLUMNS]
    )
    str_list = pa.list_(pa.string())
    schema = pa.schema([
        ("paper_id", pa.string()), ("title", pa.string()), ("venue", pa.string()), ("year", pa.int16()),
        ("split", pa.string()), ("in_eval_subset", pa.bool_()), ("abstract", pa.string()),
        ("num_reviews", pa.int8()), ("reviews", pa.list_(review_type)), ("review_template", pa.string()),
        ("novelty_excerpts", str_list), ("novelty_extraction_raw", pa.string()),
        ("novelty_score", pa.int8()), ("novelty_score_justification", pa.string()),
        ("novelty_detailed_assessment", pa.string()), ("novelty_summary", pa.string()),
    ])

    data_dir = os.path.join(args.out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    for split in ["train", "test"]:
        write_shards(df[df.split == split], schema, data_dir, split, args.rows_per_shard)

    # --- baseline model reviews for the 400 OpenReviewer benchmark papers (long format) ---
    test = pq.read_table(os.path.join(args.openreviewer_dir, "dataset_test.zstd.parquet"),
                         columns=["id"] + BASELINE_MODELS).to_pandas()
    base = test.melt(id_vars=["id"], value_vars=BASELINE_MODELS, var_name="model", value_name="generated_review")
    base = base.rename(columns={"id": "paper_id"}).sort_values(["paper_id", "model"]).reset_index(drop=True)
    base["generated_review"] = [redact(t, pid, "baseline_reviews", patterns[pid], tokens[pid])
                                for t, pid in zip(base.generated_review, base.paper_id)]
    base_dir = os.path.join(args.out_dir, "baseline_reviews")
    os.makedirs(base_dir, exist_ok=True)
    base_schema = pa.schema([("paper_id", pa.string()), ("model", pa.string()), ("generated_review", pa.string())])
    write_shards(base, base_schema, base_dir, "test", len(base))

    # --- privacy report (contains the matched names: never put it in the release folder) ---
    report = pd.DataFrame(redact.log, columns=["paper_id", "field", "kind", "match", "context"])
    os.makedirs(os.path.dirname(os.path.abspath(args.privacy_report)), exist_ok=True)
    report.to_csv(args.privacy_report, index=False)

    # --- stats ---
    all_reviews = pd.DataFrame([r for rs in df.reviews for r in rs])
    stats = {
        "num_papers": len(df),
        "num_reviews": int(df.num_reviews.sum()),
        "papers_per_split": df.split.value_counts().to_dict(),
        "papers_per_venue_year": {f"{v} {y}": int(n) for (v, y), n in df.groupby(["venue", "year"]).size().items()},
        "reviews_per_venue_year": {f"{v} {y}": int(n) for (v, y), n in df.groupby(["venue", "year"]).num_reviews.sum().items()},
        "reviews_per_paper": {str(k): int(v) for k, v in df.num_reviews.value_counts().sort_index().items()},
        "eval_subset_size": int(df.in_eval_subset.sum()),
        "novelty_score_by_split": {s: {str(int(k)): int(v) for k, v in g.novelty_score.value_counts().sort_index().items()}
                                   for s, g in df.groupby("split")},
        "novelty_score_distribution": {str(int(k)): int(v) for k, v in df.novelty_score.value_counts(dropna=False).sort_index().items() if pd.notna(k)},
        "novelty_score_missing": int(df.novelty_score.isna().sum()),
        "novelty_score_missing_ids": sorted(df.paper_id[df.novelty_score.isna()]),
        "novelty_summary_without_justification": int(df.novelty_score_justification.isna().sum()),
        "novelty_extraction_parse_status": parse_status.value_counts().to_dict(),
        "papers_with_no_novelty_excerpts": int((df.novelty_excerpts.map(len) == 0).sum()),
        "mean_novelty_excerpts_per_paper": round(float(df.novelty_excerpts.map(len).mean()), 2),
        "reviews_missing_rating": int(all_reviews.rating.isna().sum()),
        "reviews_missing_confidence": int(all_reviews.confidence.isna().sum()),
        "titles_missing": int(df.title.isna().sum()),
        "abstracts_missing": int(df.abstract.isna().sum()),
        **source_counts,
        "redactions": {f"{f}/{k}": int(n) for (f, k), n in report.groupby(["field", "kind"]).size().items()},
        "papers_with_redactions": int(report.paper_id.nunique()),
        "baseline_reviews": {"num_rows": len(base), "models": BASELINE_MODELS,
                             "empty": int(base.generated_review.fillna("").str.strip().eq("").sum())},
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
