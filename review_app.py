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

def confidence_text(c):
    if c is None:
        return "unbekannt"
    if c >= 0.8:
        return "sehr sicher"
    if c >= 0.6:
        return "eher sicher"
    if c >= 0.4:
        return "eher unsicher"
    return "sehr unsicher"

evidence = (row.get("evidence", "") or "").split("|")[0]
current = (row.get("suggested_labels", "") or "").split("|")[0]
conf_raw = (row.get("confidences", "") or "").split("|")[0]
try:
    conf_val = float(conf_raw)
except Exception:
    conf_val = None

st.subheader("Automatische Zuordnung")
st.write(f"**Gefundener String:** {evidence}")
st.write(f"**Kategorie:** {current}")
st.write(f"**Sicherheit:** {confidence_text(conf_val)}")

if "changing" not in st.session_state:
    st.session_state.changing = False

if not st.session_state.changing:
    c4, c5 = st.columns(2)
    if c4.button("Bestätigen"):
        df.at[row.name, "final_label"] = current
        save(df)
        st.session_state.idx = min(len(df)-1, st.session_state.idx + 1)
    if c5.button("Ändern"):
        st.session_state.changing = True
else:
    new_label = st.selectbox("Neue Kategorie wählen", categories, index=categories.index(current) if current in categories else 0)
    if st.button("Übernehmen"):
        df.at[row.name, "final_label"] = new_label
        save(df)
        st.session_state.changing = False
        st.session_state.idx = min(len(df)-1, st.session_state.idx + 1)

# Optional: Leitfaden-Schnipsel anzeigen (falls im YAML meta/desc vorhanden)
with st.expander("Leitfaden (Ausschnitt)"):
    st.write(codebook.get("meta", {}))
