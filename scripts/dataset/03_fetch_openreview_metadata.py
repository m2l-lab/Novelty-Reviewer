"""
Fetch official title, abstract and author list of every paper in the dataset from the OpenReview API.

Titles and abstracts go into the release dataset. Author names are ONLY used by script 04 to check that no
author identity leaks into the released text fields; they are never written to the release.

Needs an OpenReview account (anonymous API access is blocked by a challenge page). Credentials are read
from OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD, or from a JSON file {"username": ..., "password": ...}.

Usage:
  python scripts/dataset/03_fetch_openreview_metadata.py \
      --openreviewer_dir <raw crawl: dataset.zstd.parquet> \
      --neurips2025_reviews neurips_2025.parquet \
      --credentials_file ~/.openreview_credentials.json \
      --out_file openreview_metadata.parquet  # contains author names: keep private
"""

import argparse
import json
import os
import time

import pandas as pd
import pyarrow.parquet as pq
import requests

API_V2 = "https://api2.openreview.net"
API_V1 = "https://api.openreview.net"
PAGE = 1000


def _wait(r, attempt):
    retry_after = r.headers.get("Retry-After", "")
    time.sleep(int(retry_after) if retry_after.isdigit() else min(120, 5 * 2 ** attempt))


def login(base, username, password, retries=8):
    for attempt in range(retries):
        r = requests.post(f"{base}/login", json={"id": username, "password": password}, timeout=60)
        if r.status_code == 429:
            _wait(r, attempt)
            continue
        r.raise_for_status()
        return r.json()["token"]
    r.raise_for_status()


class LazySession(requests.Session):
    """Logs in on first request, so the v1 API is only contacted for venues that need it."""

    def __init__(self, base, username, password):
        super().__init__()
        self.base, self._cred = base, (username, password)

    def get(self, *a, **kw):
        if "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {login(self.base, *self._cred)}"
        return super().get(*a, **kw)


def get(session, url, params, retries=8):
    for attempt in range(retries):
        r = session.get(url, params=params, timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
            _wait(r, attempt)
            continue
        # 404: unknown id; 403: note exists but is not visible to us (e.g. withdrawn and hidden).
        # A 403 challenge page means we are not authenticated, which must not be silently skipped.
        if r.status_code == 404 or (r.status_code == 403 and "Challenge" not in r.text):
            return {"notes": []}
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def get_all(session, url, params):
    notes, offset = [], 0
    while True:
        batch = get(session, url, {**params, "limit": PAGE, "offset": offset}).get("notes", [])
        notes += batch
        if len(batch) < PAGE:
            return notes
        offset += PAGE


def _v(x):
    # API v2 wraps content values as {"value": ...}
    return x.get("value") if isinstance(x, dict) else x


def parse_v2(note):
    c = note.get("content", {})
    return {"paper_id": note["forum"], "title": _v(c.get("title")), "abstract": _v(c.get("abstract")),
            "authors": _v(c.get("authors")) or [], "api_version": 2}


def parse_v1(note):
    c = dict(note.get("content", {}))
    # Blind submissions show "Anonymous"; the original note holds the real authors once released.
    orig = (note.get("details") or {}).get("original") or {}
    authors = (orig.get("content") or {}).get("authors") or c.get("authors") or []
    authors = [a for a in authors if "anonymous" not in a.lower()]
    return {"paper_id": note["forum"], "title": c.get("title"), "abstract": c.get("abstract"),
            "authors": authors, "api_version": 1}


def fetch_venue(s2, s1, venue):
    rows = [parse_v2(n) for n in get_all(s2, f"{API_V2}/notes", {"invitation": f"{venue}/-/Submission"})]
    if rows:
        return rows
    for inv in ["Blind_Submission", "Submission"]:
        rows += [parse_v1(n) for n in get_all(s1, f"{API_V1}/notes", {"invitation": f"{venue}/-/{inv}",
                                                                       "details": "original"})]
    return rows


def fetch_one(s2, s1, paper_id):
    notes = get(s2, f"{API_V2}/notes", {"id": paper_id}).get("notes", [])
    if notes:
        return parse_v2(notes[0])
    notes = get(s1, f"{API_V1}/notes", {"id": paper_id, "details": "original"}).get("notes", [])
    return parse_v1(notes[0]) if notes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openreviewer_dir", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--credentials_file", default=None)
    ap.add_argument("--neurips2025_reviews", default=None, help="neurips_2025.parquet (papers not in openreviewer_dir)")
    args = ap.parse_args()

    if args.credentials_file:
        with open(os.path.expanduser(args.credentials_file)) as f:
            cred = json.load(f)
        username, password = cred["username"], cred["password"]
    else:
        username, password = os.environ.get("OPENREVIEW_USERNAME"), os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        raise ValueError("provide --credentials_file or set OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD")

    papers = pq.read_table(os.path.join(args.openreviewer_dir, "dataset.zstd.parquet"),
                           columns=["id", "or_venue"]).to_pandas().drop_duplicates("id")
    if args.neurips2025_reviews:
        n25 = pd.read_parquet(args.neurips2025_reviews, columns=["paper_id"]).rename(columns={"paper_id": "id"})
        papers = pd.concat([papers, n25.assign(or_venue="NeurIPS.cc/2025/Conference")]).drop_duplicates("id")

    # Resume: keep papers already fetched into out_file and only query the rest.
    found = {}
    if os.path.exists(args.out_file):
        prev = pd.read_parquet(args.out_file)
        found = {r["paper_id"]: {**r, "authors": list(r["authors"])} for r in prev.to_dict("records")}
        print(f"resuming: {len(found)} papers already in {args.out_file}", flush=True)
    wanted = set(papers.id)
    papers = papers[~papers.id.isin(found)]

    s2, s1 = LazySession(API_V2, username, password), LazySession(API_V1, username, password)

    for venue in sorted(papers.or_venue.unique()):
        rows = [r for r in fetch_venue(s2, s1, venue) if r["paper_id"] in wanted]
        for r in rows:
            found.setdefault(r["paper_id"], r)
        n_venue = (papers.or_venue == venue).sum()
        print(f"{venue}: {len({r['paper_id'] for r in rows})} / {n_venue} papers", flush=True)

    missing = sorted(wanted - set(found))
    print(f"fetching {len(missing)} remaining papers by id", flush=True)
    for i, pid in enumerate(missing):
        r = fetch_one(s2, s1, pid)
        if r:
            found[pid] = r
        if i % 200 == 0:
            print(f"  {i}/{len(missing)}", flush=True)

    df = pd.DataFrame(found.values())
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    df.to_parquet(args.out_file, index=False)
    print(f"wrote {len(df)} / {len(wanted)} papers to {args.out_file}")
    print(f"  with title: {df.title.notna().sum()}, abstract: {df.abstract.notna().sum()}, "
          f"authors: {(df.authors.map(len) > 0).sum()}")


if __name__ == "__main__":
    main()
