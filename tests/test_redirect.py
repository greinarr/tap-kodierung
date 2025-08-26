import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from semi_auto_coder_v2 import score_text


def test_redirect_scores_and_evidence():
    codebook = {
        "categories": {
            "A": {
                "keywords": ["foo"],
                "regex": [],
                "excludes": [],
                "redirects": [
                    {"pattern": "bar", "to": "B"}
                ],
            },
            "B": {"keywords": []},
        }
    }

    scores, _, hits = score_text(
        "foo bar", codebook,
        fuzzy_threshold=100,
        w_keyword=1.0,
        w_regex=0.0,
        w_fuzzy=0.0,
        w_excl=-1.0,
        use_lemma=False,
    )

    assert scores["B"] == pytest.approx(1.01)
    assert scores["A"] == pytest.approx(0.8)
    assert any("REDIR from A" in h for h in hits["B"])
    assert any("REDIR→B:bar" in h for h in hits["B"])
