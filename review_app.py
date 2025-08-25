import argparse, sys
import pandas as pd
import streamlit as st
import yaml
from pathlib import Path

def load_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--csv", required=True)
    p.add_argument("--codebook", required=True)
    args, _ = p.parse_known_args(sys.argv[1:])
    return args

args = load_args()
df = pd.read_csv(args.csv)
with open(args.codebook, "r", encoding="utf-8") as f:
    codebook = yaml.safe_load(f) or {}

categories = list((codebook.get("categories") or {}).keys())

st.set_page_config(page_title="Coding Review", layout="wide")
st.title("Halbautomatisches Kodieren – Review")

if "idx" not in st.session_state:
    st.session_state.idx = 0

def save(df):
    out = Path(args.csv).with_suffix(".reviewed.csv")
    df.to_csv(out, index=False, encoding="utf-8")
    st.success(f"Gespeichert → {out}")

c1, c2, c3 = st.columns([1,6,1])
with c2:
    st.write(f"Eintrag {st.session_state.idx+1} / {len(df)}")

if c1.button("◀ Zurück"):
    st.session_state.idx = max(0, st.session_state.idx - 1)
if c3.button("Weiter ▶"):
    st.session_state.idx = min(len(df)-1, st.session_state.idx + 1)

row = df.iloc[st.session_state.idx]

st.subheader("Text")
st.write(row.get("text", ""))

st.subheader("Vorschläge")
st.write(f"**Labels:** {row.get('suggested_labels','')}")
st.write(f"**Scores:** {row.get('scores','')} | **Conf:** {row.get('confidences','')}")
st.write(f"**Evidenz:** {row.get('evidence','')}")

st.subheader("Korrigieren / Bestätigen")
current = (row.get("suggested_labels","").split("|")[0] if isinstance(row.get("suggested_labels",""), str) and row.get("suggested_labels","") else "")
choice = st.selectbox("Label wählen", [""] + categories, index=([""] + categories).index(current) if current in categories else 0)
notes = st.text_input("Kommentar/Notiz", value=row.get("review_note", ""))

c4, c5 = st.columns(2)
if c4.button("Label übernehmen"):
    df.at[row.name, "final_label"] = choice
    df.at[row.name, "review_note"] = notes
    save(df)
if c5.button("Überspringen"):
    st.session_state.idx = min(len(df)-1, st.session_state.idx + 1)

# Optional: Leitfaden-Schnipsel anzeigen (falls im YAML meta/desc vorhanden)
with st.expander("Leitfaden (Ausschnitt)"):
    st.write(codebook.get("meta", {}))
