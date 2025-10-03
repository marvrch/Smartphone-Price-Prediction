from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, glob
from encoders import KFoldTargetEncoder  # needed for unpickle if used in pipeline
import re

st.set_page_config(page_title="Smartphone Price Predictor", page_icon="📱", layout="centered")
st.title("📱 Smartphone Price Predictor")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"
META_PATH  = BASE_DIR / "artifacts" / "meta.json"
BRAND_MODEL_CSV = BASE_DIR/ "phone_brand_model_list.csv"

# ---------- helpers ----------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open("artifacts/meta.json","r",encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

def load_brand_model_map(path=BRAND_MODEL_CSV):
    """
    Load brand→[models] mapping from a 2-column CSV (brand, model).
    Values are normalized to lowercase for consistency with the pipeline.
    Falls back to meta choices when the file isn't available.
    """
    try:
        df = pd.read_csv(path)
        # normalize
        df['brand'] = df['brand'].astype(str).str.strip().str.lower()
        df['model'] = df['model'].astype(str).str.strip().str.lower()
        # drop missing/dupe pairs and build map
        df = df.dropna(subset=['brand','model']).drop_duplicates()
        mapping = (
            df.groupby('brand')['model']
              .apply(lambda x: sorted(set(x)))
              .to_dict()
        )
        brand_choices = sorted(mapping.keys())
        all_models = sorted(set(df['model']))
        return mapping, brand_choices, all_models
    except Exception:
        # fallback to meta if CSV missing on cloud
        brands = sorted(set(str(x).strip().lower()
                            for x in meta.get("choices", {}).get("brand", [])))
        models = sorted(set(str(x).strip().lower()
                            for x in meta.get("choices", {}).get("model", [])))
        mapping = {b: models for b in brands}
        return mapping, brands, models

model, meta = load_assets()
st.caption(f"Model: **{meta['model_name']}** • y transform: "
           f"{'log1p → expm1' if meta.get('y_is_log1p', False) else 'none'}")


# higher rank = “more premium”
VARIANTS = [
    ("ultra", 5),
    ("pro max", 5), ("promax", 5), ("max pro", 5),
    ("pro plus", 4), ("pro+", 4), ("pro", 3),
    ("max", 2), ("plus", 2),
    ("se", 1), ("mini", 1), ("fe", 1), ("lite", 1),
]
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

def pick_variant(text: str):
    """
    Return (label_normalized, rank). If nothing matches → ('standard', 0).
    Includes light normalization to merge spelling variants.
    """
    t = _norm(text)
    # normalize spelling to reduce ambiguity
    t = (t.replace("pro+", "pro plus")
           .replace("promax", "pro max")
           .replace("max pro", "pro max"))
    # quick special handling for s10/s10+ type strings (optional)
    if t == "s10+":
        t = t.replace("s10+", "s10 plus")

    for label, rank in VARIANTS:
        # word-boundary match, so 'pro max' isn’t caught inside longer tokens
        if re.search(rf"\b{re.escape(label)}\b", t):
            # unify output label a bit
            out = "pro max" if label in ("promax","max pro") else ("pro plus" if label=="pro+" else label)
            return out, rank
    return "standard", 0

def proper(x: str) -> str:
    """Display helper: ProperCase without changing the underlying value."""
    try:
        return str(x).strip().title()
    except Exception:
        return str(x)

# ---------- Inputs ----------
st.sidebar.header("Masukkan Spesifikasi")

inputs = {}

# 1) Categorical (non-binary)
cat_cols = (meta.get("cat_low_features", []) + meta.get("cat_high_features", []))
cat_cols = [c for c in cat_cols if c not in set(meta.get("binary_features", []))]
cat_cols = [c for c in cat_cols if c.lower() not in ("brand", "model")]  # <-- exclude here

# --- Dependent dropdown: brand -> model ---
brand_model_map, brand_choices, all_models = load_brand_model_map()

# Brand dropdown
brand_sel = st.sidebar.selectbox("brand", options=brand_choices, format_func=proper)
inputs["brand"] = str(brand_sel).strip().lower()

# Model dropdown depends on brand
model_options = brand_model_map.get(inputs["brand"], all_models)
model_sel = st.sidebar.selectbox("model", options=model_options, format_func=proper)
inputs["model"] = str(model_sel).strip().lower()

# --- Other categorical columns (if any) ---
for c in cat_cols:
    choices = meta.get("choices", {}).get(c, [])
    if len(choices) > 80:
        default = choices[0] if choices else ""
        inputs[c] = st.sidebar.text_input(c, value=default).strip().lower()
    else:
        sel = st.sidebar.selectbox(c, options=choices if choices else [""], format_func=proper)
        inputs[c] = str(sel).strip().lower()


# 2) Numeric (RAM / Storage as dropdown of unique values if provided)

# --- Variant input (dropdown) ---
st.sidebar.subheader("Variant")

# dropdown labels + mapping to rank
VARIANT_LABELS = ["standard"] + [lab for lab, _ in VARIANTS]
VARIANT_TO_RANK = {"standard": 0, **{lab: rank for lab, rank in VARIANTS}}

variant_label = st.sidebar.selectbox("Variant", options=VARIANT_LABELS, index=0, format_func=proper)
variant_rank = VARIANT_TO_RANK[variant_label]
st.sidebar.caption(f"Terpilih: **{proper(variant_label)}** → rank = **{variant_rank}**")

# add into features (do NOT reset `inputs`)
inputs["variant_rank"] = variant_rank

num_cols = [c for c in meta["num_features"] if c != "variant_rank"]
numeric_choices = meta.get("numeric_choices", {})


for c in num_cols:
    if c.lower() in ("ram","storage"):
        # use dropdown from metadata if available; else number_input
        numeric_choices = meta.get("numeric_choices", {}).get(c, [])
        if numeric_choices:
            inputs[c] = st.sidebar.selectbox(c, options=numeric_choices, index=0)
        else:
            step = 1.0 if c.lower()=="ram" else 16.0
            default = 8.0 if c.lower()=="ram" else 128.0
            inputs[c] = st.sidebar.number_input(c, min_value=1.0, value=default, step=step)
    else:
        # generic numeric
        inputs[c] = st.sidebar.number_input(c, value=0.0)

# 3) Binary → Yes/No (store as 0/1 for model, show Yes/No in table)
bin_cols = meta.get("binary_features", [])
bin_display = {}
for c in bin_cols:
    val = st.sidebar.selectbox(c, ["No","Yes"], index=0)  # default No
    inputs[c] = 1 if val == "Yes" else 0
    bin_display[c] = val

# ---------- Show input table (with Yes/No for binaries) ----------
df_preview = pd.DataFrame([inputs]).copy()
df_preview.insert(0, "variant", proper(variant_label))  # show friendly label

for c in cat_cols:
    df_preview[c] = df_preview[c].apply(proper)

# show Yes/No for binaries if you haven’t already
for c in meta["binary_features"]:
    df_preview[c] = "Yes" if inputs[c] == 1 else "No"

st.subheader("Input")
st.write(df_preview)


# ---------- Predict ----------
if st.button("Prediksi Harga"):
    # Build DataFrame in the training column order
    X = pd.DataFrame([[inputs.get(col, np.nan) for col in meta["feature_order"]]],
                     columns=meta["feature_order"])

    # Ensure strings for categorical columns
    for c in cat_cols:
        X[c] = X[c].astype(str).str.strip().str.lower().fillna("")

    # Predict (pipeline handles all preprocessing)
    y_pred = model.predict(X)
    price = float(np.expm1(y_pred[0])) if meta.get("y_is_log1p", False) else float(y_pred[0])

    # show number without currency
    formatted = f"{price:,.0f}"
    st.success(f"Perkiraan harga: **{formatted}**")

    with st.expander("Detail Prediksi"):
        st.write({
            "Model": meta.get("model_name", "model.pkl"),
            "y_pred_raw": float(y_pred[0]),
            "y_transform": "log1p→expm1" if meta.get("y_is_log1p", False) else "none"
    })

st.info("Catatan: Jika brand/model belum pernah muncul saat training, encoder akan menggunakan rata-rata global (fallback).")
