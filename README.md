# TAP Kodierung (halbautomatisch)

## Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional:
# pip install spacy && python -m spacy download de_core_news_sm

## Nutzung
python semi_auto_coder_v2.py ./data TAP_codebook.yaml output/out.csv -r \
  -k 3 --min-score 1.0 --fuzzy-threshold 82 --w-fuzzy 1.0 --lemmatize

Review:
streamlit run review_app.py -- --csv output/out.csv --codebook TAP_codebook.yaml

## Struktur
- data/   (DOCX, nicht versionieren)
- output/ (CSV, nicht versionieren)
