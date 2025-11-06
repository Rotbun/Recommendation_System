import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="E-commerce Recommender — L2R (Minimal UI)", layout="wide")

# ---------- Helpers ----------
def safe_read_csv(p: Path):
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception as e:
        st.warning(f"Failed to read {p}: {e}")
    return None

def show_png_if_exists(p: Path, caption: str):
    if not p.exists():
        return
    try:
        # Newer Streamlit
        st.image(str(p), caption=caption, use_container_width=True)
    except TypeError:
        # Older Streamlit fallback
        st.image(str(p), caption=caption)

def _hex(x):  # clamp and convert 0..255 to hex
    x = int(min(255, max(0, round(x))))
    return f"{x:02x}"

def _interp_color(v, vmin, vmax, c0=(230,245,255), c1=(0,92,175)):
    # light blue -> deep blue ramp by rank (lower rank = darker)
    if vmax <= vmin:
        t = 0.0
    else:
        t = (v - vmin) / (vmax - vmin)
    # invert so rank 1 is strongest
    t = 1.0 - t
    r = (1-t)*c0[0] + t*c1[0]
    g = (1-t)*c0[1] + t*c1[1]
    b = (1-t)*c0[2] + t*c1[2]
    return f"#{_hex(r)}{_hex(g)}{_hex(b)}"

def styled_table(df: pd.DataFrame, gradient_on: str = "rank", seen_col: str = "seen"):
    """Return HTML with a custom CSS gradient on `gradient_on` and highlight `seen`—no matplotlib required."""
    sty = df.copy()
    styler = sty.style.set_table_attributes('class="dataframe table table-striped"')

    if gradient_on in sty.columns and len(sty) > 0:
        vals = pd.to_numeric(sty[gradient_on], errors="coerce")
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        def _colorize_rank(col):
            styles = []
            for v in pd.to_numeric(col, errors="coerce"):
                if pd.isna(v):
                    styles.append("")
                else:
                    styles.append(f"background-color: {_interp_color(float(v), vmin, vmax)}")
            return styles
        styler = styler.apply(_colorize_rank, subset=[gradient_on])

    if seen_col in sty.columns:
        def _hi_seen(s):
            return ['background-color: #ffe9e9' if bool(v) else '' for v in s]
        styler = styler.apply(_hi_seen, subset=[seen_col])

    return styler.to_html()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Configuration")
    out_root = Path(st.text_input("Outputs folder", value=str(Path("./outputs_l2r").resolve())))
    eval_dir = out_root / "evaluation"
    explain_dir = out_root / "explain"

    topk_display = st.slider("Top-N to display", 5, 100, 10, 1)
    unseen_only = st.checkbox("Show unseen items only", value=True)
    keyword = st.text_input("Search by keyword (item_id or text contains)", value="").strip()
    show_raw = st.checkbox("Show raw tables under results", value=False)

st.title("AI-Powered E-commerce Recommendations System")

# ---------- Load data ----------
clean_fp = out_root / "clean.csv"
recs_all_fp = eval_dir / "recs_all.csv"

df_clean = safe_read_csv(clean_fp)
df_recs = safe_read_csv(recs_all_fp)

tabs = st.tabs(["🛒 Recommendations", "🧠 Explainability", "📊 Data Explorer"])

# ---------- Recommendations Tab ----------
with tabs[0]:
    st.subheader("Per-User Top-N Recommendations")
    if df_clean is None or df_clean.empty or df_recs is None or df_recs.empty:
        st.info("Need clean.csv and recs_all.csv in your outputs folder.")
    else:
        df_clean["user_id"] = df_clean["user_id"].astype(str)
        df_clean["item_id"] = df_clean["item_id"].astype(str)
        if "review_text" not in df_clean.columns:
            df_clean["review_text"] = ""

        users = sorted(df_clean["user_id"].unique().tolist())
        user = st.selectbox("Select user", users)

        models_available = sorted(df_recs["model"].astype(str).unique().tolist())
        models_pick = st.multiselect("Select models", models_available, default=models_available)

        seen_items = set(df_clean.loc[df_clean["user_id"] == user, "item_id"])
        st.caption(f"Items seen by user (sample): {list(seen_items)[:min(25, len(seen_items))]}")

        for m in models_pick:
            st.markdown(f"### Model: {m}")
            top = (
                df_recs[
                    (df_recs["user_id"].astype(str) == user)
                    & (df_recs["model"].astype(str) == m)
                ]
                .sort_values("rank")
                .head(topk_display)
                .copy()
            )
            if top.empty:
                st.info("No recommendations for this model/user.")
                continue

            # Sample text per item for quick context
            tex = (
                df_clean[["item_id", "review_text"]]
                .dropna()
                .groupby("item_id")["review_text"]
                .apply(lambda s: str(s.iloc[0]) if len(s) > 0 else "")
                .to_dict()
            )

            top["seen"] = top["item_id"].astype(str).isin(seen_items)
            top["sample_text"] = top["item_id"].map(tex).fillna("")

            # Keyword filter
            if keyword:
                kw = keyword.lower()
                mask = top["item_id"].astype(str).str.lower().str.contains(kw) | top["sample_text"].str.lower().str.contains(kw)
                top = top[mask]

            # Unseen filter
            if unseen_only:
                top = top[~top["seen"]]

            top["sample_text"] = top["sample_text"].astype(str).str.slice(0, 220)

            show_cols = ["rank", "item_id", "seen", "sample_text"]
            show = top[show_cols].reset_index(drop=True)
            html = styled_table(show, gradient_on="rank", seen_col="seen")
            st.markdown(html, unsafe_allow_html=True)

            if show_raw:
                st.caption("Raw rows")
                st.dataframe(show, use_container_width=True)

# ---------- Explainability Tab ----------
with tabs[1]:
    st.subheader("SHAP Explainability (from LTR model)")
    shap_summary = out_root / "explain" / "shap_summary.png"
    shap_bar = out_root / "explain" / "shap_bar.png"

    if shap_summary.exists() or shap_bar.exists():
        col1, col2 = st.columns(2)
        with col1:
            show_png_if_exists(shap_summary, "SHAP Summary (global feature importance)")
        with col2:
            show_png_if_exists(shap_bar, "Mean |SHAP| per feature")
    else:
        st.info("SHAP images not found. Re-run the pipeline to generate them.")

# ---------- Data Explorer ----------
with tabs[2]:
    st.subheader("Dataset Explorer")
    if df_clean is None or df_clean.empty:
        st.info("clean.csv not found in outputs.")
    else:
        st.write("Rows:", len(df_clean))
        st.write("Users:", df_clean["user_id"].nunique(), "Items:", df_clean["item_id"].nunique())
        col1, col2 = st.columns(2)
        with col1:
            vc = df_clean["user_id"].value_counts()
            st.write("User interactions (top 20)")
            st.dataframe(vc.head(20).to_frame("count"))
        with col2:
            ic = df_clean["item_id"].value_counts()
            st.write("Item interactions (top 20)")
            st.dataframe(ic.head(20).to_frame("count"))

        if st.checkbox("Show sample rows", value=False):
            st.dataframe(df_clean.head(50), use_container_width=True)

st.caption("Use the sidebar to point to your outputs, adjust Top-N, filter by unseen, and search by keyword.")
