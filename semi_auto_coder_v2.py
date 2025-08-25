#!/usr/bin/env python3
"""
semi_auto_coder_v2.py — Semi-automatisches Kodieren mit Redirects & Valenz
Install:
    pip install python-docx pandas pyyaml rapidfuzz
    # optional für bessere Treffer:
    # pip install spacy && python -m spacy download de_core_news_sm

Beispiel:
    python semi_auto_coder_v2.py ./frageboegen TAP_codebook.yaml out.csv -r -k 3 \
      --min-score 1.0 --fuzzy-threshold 85 --lemmatize
"""
import argparse, csv, re, sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml, pandas as pd
from rapidfuzz import fuzz
from docx import Document

try:
    import spacy
    _NLP = None
except Exception:
    spacy = None
    _NLP = None

Location = Tuple[str, str]

def _load_spacy():
    global _NLP
    if _NLP is None and spacy is not None:
        try:
            _NLP = spacy.load("de_core_news_sm")
        except Exception:
            _NLP = None
    return _NLP

def find_docx_paths(path: Path, recursive: bool) -> List[Path]:
    if path.is_file() and path.suffix.lower() == ".docx":
        return [path]
    if path.is_dir():
        return [p for p in (path.rglob("*.docx") if recursive else path.glob("*.docx")) if p.is_file()]
    return []

def _iter_paragraphs(container, prefix: str):
    for i, p in enumerate(container.paragraphs, start=1):
        text = p.text.strip()
        if text:
            yield (f"{prefix} paragraph {i}", text)

def _iter_tables(container, prefix: str):
    for ti, table in enumerate(container.tables, start=1):
        for r, row in enumerate(table.rows, start=1):
            for c, cell in enumerate(row.cells, start=1):
                t = cell.text.strip()
                if t:
                    yield (f"{prefix} table {ti} r{r}c{c}", t)

def extract_locations(doc: Document) -> List[Location]:
    out: List[Location] = []
    out.extend(_iter_paragraphs(doc, "body"))
    out.extend(_iter_tables(doc, "body"))
    for si, section in enumerate(doc.sections, start=1):
        header, footer = section.header, section.footer
        out.extend(_iter_paragraphs(header, f"header s{si}"))
        out.extend(_iter_tables(header, f"header s{si}"))
        out.extend(_iter_paragraphs(footer, f"footer s{si}"))
        out.extend(_iter_tables(footer, f"footer s{si}"))
    return out

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def lemmatize_de(text: str) -> str:
    nlp = _load_spacy()
    if not nlp:
        return text
    doc = nlp(text)
    return " ".join(tok.lemma_ for tok in doc)

def compile_word_regex(term: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)

def apply_redirects(matches: Dict[str, float], evidence: Dict[str, List[str]], text: str, redirects: List[Dict]) -> Tuple[Dict[str,float],Dict[str,List[str]]]:
    """If a redirect pattern matches, move score/evidence to the target category, overriding source if equal."""
    moved = False
    for rd in redirects or []:
        pat, to = rd.get("pattern"), rd.get("to")
        if not pat or not to:
            continue
        try:
            if re.search(pat, text, flags=re.IGNORECASE):
                ev = evidence.get("self", [])
                ev.append(f"REDIR→{to}:{pat}")
                evidence["self"] = ev
                matches["__redirect_to__"] = to
                moved = True
        except re.error:
            continue
    return matches, evidence

def score_text(text: str, codebook: Dict, fuzzy_threshold: int,
               w_keyword: float, w_regex: float, w_fuzzy: float, w_excl: float,
               use_lemma: bool = False):
    raw = normalize(text)
    proc = lemmatize_de(raw) if use_lemma else raw
    lower = proc.lower()

    cat_scores: Dict[str, float] = {}
    cat_hits: Dict[str, List[str]] = {}

    for cat, rules in codebook["categories"].items():
        score = 0.0
        hits: List[str] = []
        redirects = rules.get("redirects", [])

        # Keywords (ganze Wörter)
        for kw in rules.get("keywords", []) or []:
            if compile_word_regex(kw).search(proc):
                score += w_keyword; hits.append(f"KW:{kw}")

        # Regex
        for rx in rules.get("regex", []) or []:
            try:
                if re.search(rx, proc, flags=re.IGNORECASE|re.MULTILINE):
                    score += w_regex; hits.append(f"RX:{rx}")
            except re.error:
                pass

        # Fuzzy
        for kw in rules.get("keywords", []) or []:
            m = fuzz.partial_ratio(lower, kw.lower())
            if m >= fuzzy_threshold:
                score += w_fuzzy; hits.append(f"FZ:{kw}({m})")

        # Excludes (ziehen ab)
        for ex in rules.get("excludes", []) or []:
            if compile_word_regex(ex).search(proc):
                score += w_excl; hits.append(f"EX:{ex}")

        # Redirect signals
        tmp_scores = {"self": score}
        tmp_hits = {"self": hits.copy()}
        tmp_scores, tmp_hits = apply_redirects(tmp_scores, tmp_hits, proc, redirects)
        redirect_to = tmp_scores.pop("__redirect_to__", None)

        # Persist
        if score != 0:
            cat_scores[cat] = score
            cat_hits[cat] = hits

        # If redirected, credit target with at least this score
        if redirect_to and score > 0:
            cat_scores[redirect_to] = max(cat_scores.get(redirect_to, 0.0), score + 0.01)  # tiny boost
            ev = cat_hits.get(redirect_to, [])
            ev.append(f"REDIR from {cat}: " + ";".join(tmp_hits["self"]))
            cat_hits[redirect_to] = ev
            cat_scores[cat] = max(0.0, cat_scores.get(cat, 0.0) - 0.2)

        # Valence signals (nicht score-wirksam; Meta)
        vs = rules.get("valence_signals", {})
        if vs:
            for tag in vs.get("positive", []):
                if re.search(re.escape(tag), proc, flags=re.IGNORECASE):
                    hits.append(f"VAL:+:{tag}")
            for tag in vs.get("negative", []):
                if re.search(re.escape(tag), proc, flags=re.IGNORECASE):
                    hits.append(f"VAL:-:{tag}")
            cat_hits[cat] = hits

    total = sum(max(s, 0) for s in cat_scores.values())
    confidences = {c: (max(s, 0)/total) if total>0 else 0.0 for c,s in cat_scores.items()}
    return cat_scores, confidences, cat_hits

def main():
    ap = argparse.ArgumentParser(description="Semi-automatisches Kodieren (Redirects & Valenz)")
    ap.add_argument("inpath")
    ap.add_argument("codebook")
    ap.add_argument("outcsv")
    ap.add_argument("-r","--recursive", action="store_true")
    ap.add_argument("-k","--topk", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=1.0)
    ap.add_argument("--fuzzy-threshold", type=int, default=85)
    ap.add_argument("--w-keyword", type=float, default=1.0)
    ap.add_argument("--w-regex", type=float, default=1.5)
    ap.add_argument("--w-fuzzy", type=float, default=0.75)
    ap.add_argument("--w-excl", type=float, default=-1.0)
    ap.add_argument("--lemmatize", action="store_true", help="spaCy de_core_news_sm verwenden (falls installiert)")
    args = ap.parse_args()

    inpath = Path(args.inpath).expanduser().resolve()
    codebook_path = Path(args.codebook).expanduser().resolve()
    outcsv = Path(args.outcsv).expanduser().resolve()

    with open(codebook_path, "r", encoding="utf-8") as f:
        codebook = (yaml.safe_load(f) or {"categories": {}})

    paths = find_docx_paths(inpath, args.recursive)
    if not paths:
        print("Keine .docx-Dateien gefunden.", file=sys.stderr); sys.exit(1)

    rows = []
    for doc_path in paths:
        try:
            doc = Document(str(doc_path))
        except Exception as e:
            rows.append([str(doc_path), "ERROR", "", "", 0.0, "", str(e)]); continue

        for where, text in extract_locations(doc):
            if not text.strip(): continue
            scores, confs, hits = score_text(
                text, codebook, args.fuzzy_threshold,
                args.w_keyword, args.w_regex, args.w_fuzzy, args.w_excl,
                use_lemma=args.lemmatize
            )
            ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top = [c for c, s in ordered[:args.topk] if s >= args.min_score]
            top_scores = [scores[c] for c in top]
            top_conf = [confs.get(c, 0.0) for c in top]
            top_hits = [";".join(hits.get(c, [])) for c in top]

            rows.append([
                str(doc_path), where, normalize(text),
                "|".join(top),
                "|".join(f"{x:.2f}" for x in top_scores),
                "|".join(f"{x:.2f}" for x in top_conf),
                "|".join(top_hits),
            ])

    outcsv.parent.mkdir(parents=True, exist_ok=True)
    with open(outcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file","where","text","suggested_labels","scores","confidences","evidence"])
        w.writerows(rows)

    print(f"Fertig. Dateien: {len(paths)}  → {outcsv}")

if __name__ == "__main__":
    main()
