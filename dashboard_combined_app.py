import datetime
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Inside AI – Retail Dashboard (Combined)",
    page_icon="👁️",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
conn = st.connection("postgresql", type="sql")

# ─────────────────────────────────────────────
# CSV PATHS  ← change these if your working
#              directory is different
# ─────────────────────────────────────────────
CSV_DETECTION = "synthetic_data_export/detection_events.csv"
CSV_EMOTION   = "synthetic_data_export/emotion_events.csv"

# Floor plan background image for the Zone Visit Frequency Heatmap.
# Grayscale version used deliberately: the original floor plan's saturated
# blue/purple/red/orange zone-type colors visually compete with the bubble
# colorscale (which also runs blue->purple->red for visitor count), making
# it hard to tell "this is a store" apart from "this zone has high
# traffic." Grayscale recedes into the background and lets the bubble
# colors carry 100% of the data signal.
FLOOR_PLAN_PATH = "assets/mall_floor_plan_grayscale.png"
if not os.path.exists(FLOOR_PLAN_PATH):
    FLOOR_PLAN_PATH = None  # heatmap falls back to markers-only, no crash

# Chunk size for streaming large CSVs. Tune down (e.g. 50_000) if you're
# still seeing memory pressure on a constrained host; tune up if you have
# headroom and want fewer iterations.
CSV_CHUNK_SIZE = 200_000

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    h1, h2, h3, h4 { color: #1E3A5F !important; text-align: center !important; }
    [data-testid="stMetricLabel"] { text-align: center !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SIDEBAR – DATE FILTER
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📅 Date Filter")
    st.caption("Select a date range to filter all charts. Set both dates to the same day to view a single day.")

    filter_start = st.date_input(
        label="Start Date",
        value=datetime.date(2026, 1, 1),
        format="DD/MM/YYYY",
        key="filter_start",
    )
    filter_end = st.date_input(
        label="End Date",
        value=datetime.date(2026, 1, 1),
        format="DD/MM/YYYY",
        key="filter_end",
    )

    if filter_start > filter_end:
        st.error("⚠️ Start date must be on or before end date.")
        st.stop()

    dt_start = datetime.datetime.combine(filter_start, datetime.time.min)
    dt_end   = datetime.datetime.combine(filter_end,   datetime.time.max)

    st.divider()
    if filter_start == filter_end:
        st.info(f"Showing: **{filter_start.strftime('%d %b %Y')}** only")
    else:
        st.info(f"Showing: **{filter_start.strftime('%d %b %Y')}** → **{filter_end.strftime('%d %b %Y')}**")

    st.divider()
    include_postgres = st.checkbox(
        "Include Postgres data",
        value=False,
        help=(
            "When off, the dashboard reads from the synthetic CSVs only — much "
            "lighter on memory and ideal for local dev/testing. Turn on to merge "
            "in live Postgres rows (e.g. real production data alongside synthetic)."
        ),
    )

    st.divider()
    st.caption("If you've updated the CSV files on disk, click below to force a reload — otherwise cached data (up to 10 min old) is used.")
    if st.button("🔄 Reload data from source", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("Retail Dashboard (Combined Data)", anchor=False)

# ─────────────────────────────────────────────
# BRAND PALETTE & GLOBAL PLOTLY THEME
# ─────────────────────────────────────────────
PALETTE = [
    "#2563EB", "#7C3AED", "#14B8A6", "#0EA5E9",
    "#4F46E5", "#10B981", "#F59E0B", "#F43F5E"
]

C = {
    "dark_navy":  "#7C3AED",
    "blue":       "#2563EB",
    "cyan":       "#0EA5E9",
    "teal":       "#14B8A6",
    "green":      "#10B981",
    "amber":      "#F59E0B",
    "orange":     "#EC6B26",
    "red":        "#F43F5E",
    "black":      "#000000",
    "white":      "#ffffff",
}

EMOTION_COLORS_AREA = {
    "happy":    "rgba(16, 185, 129, 0.45)",
    "neutral":  "rgba(37, 99, 235, 0.45)",
    "surprise": "rgba(245, 158, 11, 0.45)",
    "sad":      "rgba(14, 165, 233, 0.45)",
    "fear":     "rgba(124, 58, 237, 0.45)",
    "angry":    "rgba(244, 63, 94, 0.45)",
    "disgust":  "rgba(236, 107, 38, 0.45)",
}

EMOTION_COLORS = {
    "happy":    C["green"],
    "neutral":  C["blue"],
    "surprise": C["amber"],
    "sad":      C["cyan"],
    "fear":     C["dark_navy"],
    "angry":    C["red"],
    "disgust":  C["orange"],
}

SENTIMENT_COLORS = {
    "Positive": C["green"],
    "Neutral":  C["blue"],
    "Negative": C["red"],
}

SANKEY_NODE_COLORS = [
    "#2563EB", "#7C3AED", "#14B8A6", "#0EA5E9",
    "#4F46E5", "#10B981", "#F59E0B", "#F43F5E",
    "#06B6D4", "#8B5CF6", "#34D399", "#FCD34D",
]

def brand_layout(fig, margin_top=20, margin_bottom=20, **kwargs):
    fig.update_layout(
        font=dict(family="sans-serif", color=C["black"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f4f6f9",
        colorway=PALETTE,
        margin=dict(t=margin_top, b=margin_bottom),
        **kwargs,
    )
    fig.update_xaxes(
        gridcolor="#ffffff", zerolinecolor="#c4cad6",
        title_font=dict(color=C["black"]), tickfont=dict(color=C["black"]),
    )
    fig.update_yaxes(
        gridcolor="#ffffff", zerolinecolor="#c4cad6",
        title_font=dict(color=C["black"]), tickfont=dict(color=C["black"]),
    )
    return fig

def hex_to_rgba(hex_color, alpha=0.35):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# Generic store-name collapse pattern: matches any zone name ending in the
# literal suffix "_Entry" or "_Exit" (e.g. "Pharmacy_Entry", "Store_A_Exit",
# "Kitchenware_Entry") and captures everything before it. Verified safe
# against the rest of the zone naming scheme: Mall_Entrance_* / Mall_Exit_*
# zones end in directional words (_East, _North, _South, _West,
# _Underground) or "Mall_Main_Entrance" — none of which end in the exact
# substring "_Entry" or "_Exit", so they correctly pass through unchanged.
STORE_NAME_PATTERN = r"^(.+)_(Entry|Exit)$"

def collapse_store_entry_exit(zone_name_series: pd.Series) -> pd.Series:
    """
    Collapse any '<StoreName>_Entry' / '<StoreName>_Exit' pair into a single
    '<StoreName>' (e.g. 'Store_A_Entry' -> 'Store_A', 'Pharmacy_Exit' ->
    'Pharmacy', 'Kitchenware_Entry' -> 'Kitchenware'). Matches any store
    name, not just ones literally prefixed "Store_" — required once
    category-named stores (Pharmacy, Grocery, Clothes, Shoes, Gaming,
    Kitchenware) were added alongside Store_A/Store_B.
    Non-store zones (corridors, atrium, mall entrances/exits) pass through
    unchanged, since "Mall_Entrance_North" etc. don't end in _Entry/_Exit
    immediately after a single name segment in the same way — but to be
    safe we still anchor on the exact store zone names from ZONES rather
    than a bare suffix regex (see STORE_NAME_PATTERN below).
    """
    return zone_name_series.str.replace(
        STORE_NAME_PATTERN, r"\1", regex=True
    )


def clean_id_col(series: pd.Series) -> pd.Series:
    """
    Normalise an ID column (zone_id / previous_zone_id) to a clean string
    for use as a merge key, handling the float64 '12.0' vs int64 '12'
    mismatch that happens when a column has nulls and gets read back
    from CSV as float64 while a sibling column (no nulls) stays int64.
    True nulls (NaN/None/empty) become Python None, not the string 'nan'.
    """
    s = series.astype(str).str.strip()
    # Strip a trailing '.0' that appears when pandas reads a nullable
    # int column as float64 (e.g. '12.0' -> '12'), but leave genuine
    # decimals alone (there shouldn't be any for ID columns).
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"None": None, "none": None, "nan": None, "NaT": None, "<NA>": None})
    return s


def get_csv_row_count(path: str) -> int | None:
    """Cheap line count for sidebar diagnostics, without loading the file."""
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f) - 1  # minus header
    except FileNotFoundError:
        return None


def read_csv_filtered_by_date(
    path: str,
    start: datetime.datetime,
    end: datetime.datetime,
    timestamp_format: str = "ISO8601",
    chunksize: int = CSV_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Stream a CSV in chunks and keep only rows whose 'timestamp' column falls
    within [start, end]. This bounds peak memory to roughly one chunk's
    worth of data rather than the full file, which matters a lot once the
    detection_events CSV spans months of per-frame synthetic data.

    Falls back to format='mixed' (slower, but tolerant of inconsistent
    timestamp strings) if the strict ISO8601 parse fails on a chunk.
    """
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        try:
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], format=timestamp_format)
        except (ValueError, TypeError):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], format="mixed")
        mask = (chunk["timestamp"] >= start) & (chunk["timestamp"] <= end)
        if mask.any():
            chunks.append(chunk.loc[mask])
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    # Return an empty frame with the right columns by peeking the header only.
    try:
        header_only = pd.read_csv(path, nrows=0)
        return header_only
    except FileNotFoundError:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# DATA LOADING & MERGING
# ─────────────────────────────────────────────
# Strategy:
#   1. Stream detection_events / emotion_events CSVs in chunks, filtering to
#      the selected date range as we go (synthetic data, priority source).
#   2. Optionally pull matching Postgres rows (real data) — gated behind a
#      sidebar checkbox, since for local dev against months of synthetic
#      data the Postgres round trip is pure overhead.
#   3. Concatenate each pair, then drop duplicate unique_detection_id rows,
#      keeping the CSV row (it appears first after concat).
#   4. zone_name/zone_type for CSV rows come from the CSV's OWN columns
#      (written directly by the generator at insertion time — see
#      generator script, where every insert_detection_event() call passes
#      zone_name explicitly). We do NOT re-derive zone_name for CSV rows
#      via a Postgres zone_id join, because the CSV's fake zone_id (1-19,
#      assigned by dict insertion order in the generator) has no guaranteed
#      relationship to Postgres's real zone_id values — joining on it
#      either matches nothing (empty zones table) or matches the wrong
#      rows (different insertion order), silently nulling out zone_name
#      for some or all CSV rows. The Postgres zones lookup is only used
#      to enrich genuine Postgres-sourced rows (pg_de) and as a last-resort
#      backfill if a CSV is missing zone_name entirely (e.g. an older CSV
#      generated before this column existed).

@st.cache_data(ttl="10m")
def load_data(
    start: datetime.datetime,
    end: datetime.datetime,
    _csv_mtimes: tuple,
    use_postgres: bool,
):
    # _csv_mtimes is part of the cache key (unused inside the function) —
    # whenever either CSV's last-modified time changes, Streamlit treats
    # this as a new set of arguments and re-runs the function instead of
    # returning stale cached data.

    # ── CSV: detection_events (PRIORITY SOURCE, streamed + pre-filtered) ──
    try:
        csv_de = read_csv_filtered_by_date(CSV_DETECTION, start, end)
        if csv_de.empty and "zone_id" not in csv_de.columns:
            raise FileNotFoundError  # header peek also failed → treat as missing
    except FileNotFoundError:
        st.warning(f"⚠️ CSV not found: `{CSV_DETECTION}`. Falling back to Postgres data only.")
        csv_de = pd.DataFrame()

    if not csv_de.empty:
        # Normalise zone_id / previous_zone_id as clean, comparable strings.
        csv_de["zone_id"] = clean_id_col(csv_de["zone_id"])
        if "previous_zone_id" in csv_de.columns:
            csv_de["previous_zone_id"] = clean_id_col(csv_de["previous_zone_id"])
        else:
            csv_de["previous_zone_id"] = None

        # zone_name / zone_type: trust the CSV's own columns (written
        # directly by the generator at insertion time — guaranteed
        # non-null). No Postgres join needed for the happy path.
        if "zone_name" not in csv_de.columns or csv_de["zone_name"].isna().all():
            zones_df_bootstrap = conn.query("SELECT zone_id, zone_name, zone_type FROM zones", ttl="60m")
            zones_df_bootstrap["zone_id"] = clean_id_col(zones_df_bootstrap["zone_id"])
            csv_de = csv_de.merge(
                zones_df_bootstrap[["zone_id", "zone_name", "zone_type"]],
                on="zone_id", how="left",
            )

        # prev_zone_name: derive from THIS CSV's own zone_id -> zone_name
        # mapping (self-referential merge), now that both zone_id and
        # previous_zone_id are normalised to the SAME clean string format
        # (no more '12' vs '12.0' mismatch).
        csv_zone_name_lookup = (
            csv_de[["zone_id", "zone_name"]]
            .dropna(subset=["zone_name"])
            .drop_duplicates(subset=["zone_id"])
            .rename(columns={"zone_id": "previous_zone_id", "zone_name": "prev_zone_name"})
        )
        csv_de = csv_de.merge(csv_zone_name_lookup, on="previous_zone_id", how="left")

    # ── CSV: emotion_events (PRIORITY SOURCE) ──
    # Typically much smaller than detection_events (one row per *face*
    # detection, not per frame), so a plain read is usually fine — but we
    # still narrow it to only the unique_detection_ids we kept above,
    # which also caps its effective size to whatever survived the date
    # filter rather than the full file.
    try:
        csv_ee = pd.read_csv(CSV_EMOTION)
        if not csv_de.empty and "unique_detection_id" in csv_de.columns:
            csv_ee = csv_ee[
                csv_ee["unique_detection_id"].isin(csv_de["unique_detection_id"])
            ]
        if "confidence" in csv_ee.columns and "emotion_confidence" not in csv_ee.columns:
            csv_ee = csv_ee.rename(columns={"confidence": "emotion_confidence"})
    except FileNotFoundError:
        st.warning(f"⚠️ CSV not found: `{CSV_EMOTION}`. Falling back to Postgres emotion data only.")
        csv_ee = pd.DataFrame()

    # ── Postgres: detection_events (OPTIONAL FALLBACK / SUPPLEMENTARY) ──
    # Pulled in to fill gaps — e.g. real production data alongside
    # synthetic data, or dates the CSV doesn't cover — but CSV rows win
    # on any unique_detection_id collision (see drop_duplicates below).
    # Gated behind use_postgres: skipping this avoids loading a second,
    # potentially large, full-range result set into memory on top of the
    # CSV data, which is pure overhead for local dev against synthetic data.
    if use_postgres:
        try:
            pg_de = conn.query(
                """
                SELECT
                    de.unique_detection_id,
                    de.timestamp,
                    de.camera_id,
                    de.zone_id,
                    de.reid_id,
                    de.previous_zone_id,
                    de.movement_type,
                    de.dwell_time,
                    de.object_count_in_zone,
                    de.confidence,
                    de.frame_number,
                    z.zone_name,
                    z.zone_type,
                    zprev.zone_name AS prev_zone_name
                FROM detection_events de
                LEFT JOIN zones z     ON de.zone_id          = z.zone_id
                LEFT JOIN zones zprev ON de.previous_zone_id = zprev.zone_id
                WHERE de.timestamp >= :start AND de.timestamp <= :end
                ORDER BY de.timestamp
                """,
                params={"start": start, "end": end},
                ttl="10m",
            )
            pg_de["zone_id"] = clean_id_col(pg_de["zone_id"])
            pg_de["previous_zone_id"] = clean_id_col(pg_de["previous_zone_id"])
        except Exception as e:
            st.warning(f"⚠️ Postgres detection_events query failed, continuing with CSV data only: {e}")
            pg_de = pd.DataFrame(columns=list(csv_de.columns) if not csv_de.empty else [])

        # ── Postgres: emotion_events (OPTIONAL FALLBACK / SUPPLEMENTARY) ──
        try:
            pg_ee = conn.query(
                """
                SELECT
                    ee.unique_detection_id,
                    ee.emotion,
                    ee.confidence AS emotion_confidence
                FROM emotion_events ee
                JOIN detection_events de ON ee.unique_detection_id = de.unique_detection_id
                WHERE de.timestamp >= :start AND de.timestamp <= :end
                """,
                params={"start": start, "end": end},
                ttl="10m",
            )
        except Exception as e:
            st.warning(f"⚠️ Postgres emotion_events query failed, continuing with CSV data only: {e}")
            pg_ee = pd.DataFrame(columns=list(csv_ee.columns) if not csv_ee.empty else [])
    else:
        pg_de = pd.DataFrame(columns=list(csv_de.columns) if not csv_de.empty else [])
        pg_ee = pd.DataFrame(columns=list(csv_ee.columns) if not csv_ee.empty else [])

    # ── Combine detection_events (CSV first → CSV wins on dedup) ──
    all_de_cols = list(csv_de.columns) if not csv_de.empty else list(pg_de.columns)
    for col in all_de_cols:
        if col not in pg_de.columns:
            pg_de[col] = None
        if col not in csv_de.columns:
            csv_de[col] = None
    pg_de = pg_de[all_de_cols] if not pg_de.empty else pg_de
    csv_de = csv_de[all_de_cols] if not csv_de.empty else csv_de

    combined_de = (
        pd.concat([csv_de, pg_de], ignore_index=True)
        .drop_duplicates(subset=["unique_detection_id"], keep="first")  # CSV rows are first → CSV wins
    )
    combined_de["timestamp"] = pd.to_datetime(combined_de["timestamp"])
    combined_de["zone_id"] = clean_id_col(combined_de["zone_id"])

    # Downcast numeric columns to shrink memory footprint of the frame that
    # stays resident for the rest of the session (cached + referenced by
    # every chart below).
    for col in ("object_count_in_zone", "frame_number"):
        if col in combined_de.columns:
            combined_de[col] = pd.to_numeric(combined_de[col], errors="coerce").astype("float32")
    for col in ("dwell_time", "confidence", "processing_latency"):
        if col in combined_de.columns:
            combined_de[col] = pd.to_numeric(combined_de[col], errors="coerce").astype("float32")
    for col in ("camera_id", "movement_type", "zone_name", "zone_type", "prev_zone_name", "reid_id"):
        if col in combined_de.columns:
            combined_de[col] = combined_de[col].astype("category")

    # ── Combine emotion_events (CSV first → CSV wins on dedup) ──
    all_ee_cols = list(csv_ee.columns) if not csv_ee.empty else list(pg_ee.columns)
    for col in all_ee_cols:
        if col not in pg_ee.columns:
            pg_ee[col] = None
        if col not in csv_ee.columns:
            csv_ee[col] = None
    pg_ee = pg_ee[all_ee_cols] if not pg_ee.empty else pg_ee
    csv_ee = csv_ee[all_ee_cols] if not csv_ee.empty else csv_ee

    combined_ee = (
        pd.concat([csv_ee, pg_ee], ignore_index=True)
        .drop_duplicates(subset=["unique_detection_id"], keep="first")
    )
    if "emotion" in combined_ee.columns:
        combined_ee["emotion"] = combined_ee["emotion"].astype("category")
    if "emotion_confidence" in combined_ee.columns:
        combined_ee["emotion_confidence"] = pd.to_numeric(
            combined_ee["emotion_confidence"], errors="coerce"
        ).astype("float32")

    return combined_de, combined_ee


# Load once; Streamlit re-runs this when the date filter changes
# Cache auto-invalidation: include each CSV's last-modified time in the
# cache key. If you overwrite either CSV with new data, this changes and
# Streamlit reloads automatically — no manual cache clear needed.
def _get_csv_mtimes():
    mtimes = []
    for path in (CSV_DETECTION, CSV_EMOTION):
        try:
            mtimes.append(os.path.getmtime(path))
        except FileNotFoundError:
            mtimes.append(0.0)
    return tuple(mtimes)

combined_de, combined_ee = load_data(dt_start, dt_end, _get_csv_mtimes(), include_postgres)

# Show a small data source summary in the sidebar
with st.sidebar:
    st.divider()
    st.caption(f"**Rows loaded**")
    st.caption(f"Detection events: {len(combined_de):,}")
    st.caption(f"Emotion events: {len(combined_ee):,}")
    st.caption(f"Postgres merge: {'on' if include_postgres else 'off'}")

    with st.expander("🔍 Data diagnostics"):
        de_rows_on_disk = get_csv_row_count(CSV_DETECTION)
        ee_rows_on_disk = get_csv_row_count(CSV_EMOTION)
        st.write("**detection_events.csv total rows (on disk):**", f"{de_rows_on_disk:,}" if de_rows_on_disk is not None else "file not found")
        st.write("**emotion_events.csv total rows (on disk):**", f"{ee_rows_on_disk:,}" if ee_rows_on_disk is not None else "file not found")
        st.write("**combined_de memory usage:**", f"{combined_de.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        st.write("**zone_name null count:**", combined_de["zone_name"].isna().sum(), "/", len(combined_de))
        st.write("**reid_id non-null count:**", combined_de["reid_id"].notna().sum())
        st.write("**camera_id values:**", combined_de["camera_id"].dropna().unique().tolist())
        st.write("**zone_id dtype:**", combined_de["zone_id"].dtype)
        st.write("**Sample rows:**")
        st.dataframe(combined_de.head(10), use_container_width=True)


# ══════════════════════════════════════════════
# 1. EMOTION ANALYSIS
# ══════════════════════════════════════════════
st.header("Emotion Analysis", divider="rainbow", anchor=False)

# Join emotion onto detection to get timestamps + zone names
df_emotion = combined_ee.merge(
    combined_de[["unique_detection_id", "timestamp", "zone_id", "zone_name"]],
    on="unique_detection_id",
    how="inner",
).rename(columns={"emotion_confidence": "confidence"})
df_emotion["timestamp"] = pd.to_datetime(df_emotion["timestamp"])
df_emotion["emotion"]   = df_emotion["emotion"].astype(str).str.lower()

POSITIVE_EMOTIONS = {"happy", "surprise"}
NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}

def emotion_sentiment(emotion: str) -> str:
    e = emotion.lower()
    if e in POSITIVE_EMOTIONS:
        return "Positive"
    if e in NEGATIVE_EMOTIONS:
        return "Negative"
    return "Neutral"

df_emotion["sentiment"] = df_emotion["emotion"].apply(emotion_sentiment)

if df_emotion.empty:
    st.warning("No emotion data found for the selected date range.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Emotion Distribution", anchor=False)
        emotion_counts = df_emotion["emotion"].value_counts().reset_index()
        emotion_counts.columns = ["emotion", "count"]
        fig_bar_emotion = px.bar(
            emotion_counts, x="emotion", y="count", color="emotion",
            color_discrete_map=EMOTION_COLORS,
            labels={"emotion": "Emotion", "count": "Count"}, text_auto=True,
        )
        fig_bar_emotion.update_layout(showlegend=False)
        brand_layout(fig_bar_emotion)
        st.plotly_chart(fig_bar_emotion, use_container_width=True)

    with col2:
        st.subheader("Sentiment Breakdown", anchor=False)
        sentiment_counts = df_emotion["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        fig_pie = px.pie(
            sentiment_counts, names="sentiment", values="count",
            color="sentiment", color_discrete_map=SENTIMENT_COLORS, hole=0.35,
        )
        brand_layout(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Emotion Composition Over Time", anchor=False)

    # ── Custom date/time range override for this chart only ──
    # The sidebar's date filter sets the global range (whole days, no time
    # component). This widget lets the user zoom into a specific window
    # (e.g. 14:00–15:30 on a single day) without changing what every other
    # chart on the page shows. Defaults to the sidebar's global range.
    st.caption("Optionally override the date range above for this chart only (with time-of-day precision).")
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        comp_start_date = st.date_input(
            "Start date", value=filter_start, format="DD/MM/YYYY", key="comp_start_date",
        )
        comp_start_time = st.time_input(
            "Start time", value=datetime.time.min, key="comp_start_time",
        )
    with comp_col2:
        comp_end_date = st.date_input(
            "End date", value=filter_end, format="DD/MM/YYYY", key="comp_end_date",
        )
        comp_end_time = st.time_input(
            "End time", value=datetime.time.max.replace(microsecond=0), key="comp_end_time",
        )

    comp_dt_start = datetime.datetime.combine(comp_start_date, comp_start_time)
    comp_dt_end   = datetime.datetime.combine(comp_end_date, comp_end_time)

    if comp_dt_start > comp_dt_end:
        st.error("⚠️ Start date/time must be on or before end date/time.")
        df_ts = df_emotion.iloc[0:0].copy()  # empty slice, same columns
    else:
        df_ts = df_emotion[
            (df_emotion["timestamp"] >= comp_dt_start) & (df_emotion["timestamp"] <= comp_dt_end)
        ].copy()

    if df_ts.empty:
        st.info("No emotion data found for the selected custom date/time range.")
    else:
        df_ts["minute"] = df_ts["timestamp"].dt.floor("min")
        df_comp   = df_ts.groupby(["minute", "emotion"]).size().reset_index(name="count")
        df_pivot  = df_comp.pivot(index="minute", columns="emotion", values="count").fillna(0)
        row_totals = df_pivot.sum(axis=1).replace(0, 1)
        df_pct    = (df_pivot.div(row_totals, axis=0) * 100).reset_index()
        df_pct_long = df_pct.melt(id_vars="minute", var_name="emotion", value_name="pct")

        fig_comp = go.Figure()
        for emotion in df_pct_long["emotion"].unique():
            subset = df_pct_long[df_pct_long["emotion"] == emotion].sort_values("minute")
            fig_comp.add_trace(go.Scatter(
                x=subset["minute"], y=subset["pct"],
                name=emotion.capitalize(), mode="lines",
                line=dict(shape="spline", smoothing=0.8, width=1.5,
                          color=EMOTION_COLORS.get(emotion, C["cyan"])),
                stackgroup="one",
                fillcolor=EMOTION_COLORS_AREA.get(emotion, "rgba(14,165,233,0.4)"),
                hovertemplate="%{fullData.name}: %{y:.1f}%<br>%{x}<extra></extra>",
            ))
        fig_comp.update_layout(
            yaxis=dict(range=[0, 100], ticksuffix="%", title="Composition"),
            xaxis=dict(title="Time"),
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
            hovermode="x unified",
        )
        brand_layout(fig_comp)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Overall Happiness Score", anchor=False)
    happy_rows = df_emotion[df_emotion["emotion"] == "happy"]
    happiness_score = round((len(happy_rows) / len(df_emotion)) * 100, 1)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=happiness_score,
        delta={"reference": 50, "increasing": {"color": C["green"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": C["black"]},
            "bar": {"color": C["teal"]},
            "steps": [
                {"range": [0, 33],  "color": "#ffb1b1"},
                {"range": [33, 66], "color": "#9ce2f9"},
                {"range": [66, 100],"color": "#90ffb0"},
            ],
            "threshold": {"line": {"color": C["red"], "width": 4}, "thickness": 0.75, "value": 50},
        },
        title={"text": "Happy %", "font": {"color": C["black"]}},
        number={"suffix": "%", "font": {"color": C["black"]}},
    ))
    brand_layout(fig_gauge, margin_top=40)
    st.plotly_chart(fig_gauge, use_container_width=True)


# ══════════════════════════════════════════════
# 2. REID & CUSTOMER JOURNEY
# ══════════════════════════════════════════════
st.header("Customer Journey", divider="rainbow", anchor=False)

df_journey = combined_de[combined_de["reid_id"].notna()].copy()

# Defensive: older CSVs (generated before this column existed) may lack
# zone_type entirely. Without this, any .str accessor on a missing column
# raises a KeyError and silently kills every chart below it (this was the
# root cause of "Avg Dwell Time" showing nothing even after zone_name was
# fixed — zone_type was never written to detection_rows in the generator).
if "zone_type" not in df_journey.columns:
    df_journey["zone_type"] = None

if df_journey.empty:
    st.warning("No customer journey data found for the selected date range.")
else:
    st.subheader("Customer Flow Between Zones", anchor=False)

    # Collapse Store_X_Entry / Store_X_Exit into a single "Store_X" node
    # so journeys like Store_A_Entry → Store_A_Exit collapse to just Store_A
    df_journey_collapsed = df_journey.copy()
    df_journey_collapsed["zone_name_collapsed"]      = collapse_store_entry_exit(df_journey_collapsed["zone_name"].astype(str))
    df_journey_collapsed["prev_zone_name_collapsed"] = collapse_store_entry_exit(df_journey_collapsed["prev_zone_name"].astype(str))

    transitions_full = (
        df_journey_collapsed.dropna(subset=["previous_zone_id", "zone_id"])
        # Drop hops that become a self-loop after collapsing
        # (e.g. Store_A_Entry → Store_A_Exit becomes Store_A → Store_A)
        .query("zone_name_collapsed != prev_zone_name_collapsed")
        .drop_duplicates(subset=["reid_id", "prev_zone_name_collapsed", "zone_name_collapsed"])
        .groupby(["prev_zone_name_collapsed", "zone_name_collapsed"], as_index=False)["reid_id"]
        .nunique()
        .rename(columns={"prev_zone_name_collapsed": "source", "zone_name_collapsed": "target", "reid_id": "value"})
    )

    transitions_full["source"] = transitions_full["source"].astype(str)
    transitions_full["target"] = transitions_full["target"].astype(str)

    # ── Source / target node filter widgets ──
    # "All" keeps the full unfiltered Sankey; picking a specific source
    # and/or target narrows the diagram to flows through those node(s).
    all_source_options = ["All"] + sorted(transitions_full["source"].unique().tolist())
    all_target_options = ["All"] + sorted(transitions_full["target"].unique().tolist())

    with st.expander("🔍 Sankey diagnostics (debug)"):
        n_unique_nodes = len(pd.unique(transitions_full[["source", "target"]].values.ravel()))
        st.write("**Unique source zones:**", transitions_full["source"].nunique())
        st.write("**Unique target zones:**", transitions_full["target"].nunique())
        st.write("**Total unique nodes (source+target combined):**", n_unique_nodes)
        st.write("**Number of edges (source→target rows):**", len(transitions_full))
        st.write("**Sum of all edge values (total link weight):**", int(transitions_full["value"].sum()))
        st.write("**Sample node labels:**", sorted(pd.unique(transitions_full[["source", "target"]].values.ravel()).tolist())[:50])

    sankey_col1, sankey_col2 = st.columns(2)
    with sankey_col1:
        selected_source = st.selectbox(
            "Filter: Start node (source)",
            options=all_source_options,
            index=0,
            key="sankey_source_filter",
        )
    with sankey_col2:
        selected_target = st.selectbox(
            "Filter: Target node",
            options=all_target_options,
            index=0,
            key="sankey_target_filter",
        )

    transitions = transitions_full.copy()
    if selected_source != "All":
        transitions = transitions[transitions["source"] == selected_source]
    if selected_target != "All":
        transitions = transitions[transitions["target"] == selected_target]

    if transitions.empty:
        st.info("No transitions match the selected source/target filter.")
    else:
        # Force plain Python str/int/float before handing anything to
        # Plotly. transitions["source"]/["target"] can carry pandas
        # `category` dtype this far upstream (zone_name/prev_zone_name are
        # cast to category when combined_de is built, and that dtype
        # survives groupby/rename). Category scalars are NOT plain str —
        # passing them into go.Sankey's node/link dicts means Plotly's
        # JSON encoder has to serialize pandas Categorical objects, which
        # is a known source of runaway/incorrect serialization in some
        # plotly+pandas version combinations and can produce a much larger
        # or malformed payload than the same data as plain strings would.
        # This dataset is tiny (22 nodes / 190 edges) — converting costs
        # nothing and removes the dtype as a variable entirely.
        transitions = transitions.astype({"source": str, "target": str})
        transitions["value"] = transitions["value"].astype(int)

        all_zones  = sorted(set(transitions["source"]) | set(transitions["target"]))
        zone_index = {z: i for i, z in enumerate(all_zones)}
        node_colors = [SANKEY_NODE_COLORS[i % len(SANKEY_NODE_COLORS)] for i in range(len(all_zones))]
        link_colors = [hex_to_rgba(node_colors[zone_index[s]]) for s in transitions["source"]]

        fig_sankey = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, label=all_zones, color=node_colors,
                      line=dict(color=C["white"], width=0.5)),
            link=dict(
                source=[zone_index[s] for s in transitions["source"]],
                target=[zone_index[t] for t in transitions["target"]],
                value=transitions["value"].tolist(),
                color=link_colors,
            ),
        ))
        fig_sankey.update_traces(textfont=dict(color=C["white"], size=12), selector=dict(type="sankey"))
        brand_layout(fig_sankey)
        st.plotly_chart(fig_sankey, use_container_width=True)

    # ── Avg dwell time – uses the dwell_time column directly ──
    # Only rows where movement_type == 'dwell' carry a real dwell_time
    # (set at generation time). Exclude store_entry/store_exit/entrance/exit
    # zone types — those are crossing points, not dwell locations.
    st.subheader("Avg Dwell Time", anchor=False)
    df_dwell_raw = df_journey[
        (df_journey["movement_type"] == "dwell")
        & (df_journey["dwell_time"].notna())
        & (~df_journey["zone_type"].astype(str).str.lower().isin(
            ["entrance", "exit", "store_entry", "store_exit"]
        ))
    ].copy()
    # Collapse Store_X_Entry/Exit → Store_X for display (same as Sankey),
    # though store_entry/store_exit rows are already excluded above —
    # this only affects naming consistency if a "store" zone_type is ever added.
    df_dwell_raw["zone_name_clean"] = collapse_store_entry_exit(df_dwell_raw["zone_name"].astype(str))

    df_dwell_avg = (
        df_dwell_raw.groupby("zone_name_clean", as_index=False)["dwell_time"]
        .mean()
        .rename(columns={"dwell_time": "dwell_seconds"})
        .sort_values("dwell_seconds", ascending=False)
        .reset_index(drop=True)
    )
    if df_dwell_avg.empty:
        st.info("No dwell-type detections found for the selected date range.")
    else:
        zone_colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(df_dwell_avg["zone_name_clean"])}
        fig_dwell = px.bar(
            df_dwell_avg, x="zone_name_clean", y="dwell_seconds",
            color="zone_name_clean", color_discrete_map=zone_colors,
            labels={"zone_name_clean": "Zone", "dwell_seconds": "Avg Dwell (s)"}, text_auto=".1f",
        )
        fig_dwell.update_layout(showlegend=False)
        brand_layout(fig_dwell)
        st.plotly_chart(fig_dwell, use_container_width=True)

    # ── Store dwell time – NOT the dwell_time column (stores never trigger
    # the dwell movement_type — Store_X_Entry/Exit are instant doorway
    # crossings, 5-15s each, well under the 60s dwell threshold). Instead,
    # store dwell time is calculated as (Store_X_Exit timestamp minus
    # Store_X_Entry timestamp) for the same reid_id — i.e. how long that
    # customer actually spent inside the store between the two checkpoints.
    st.subheader("Store Dwell Time", anchor=False)
    df_store_events = df_journey[
        df_journey["zone_type"].astype(str).str.lower().isin(["store_entry", "store_exit"])
    ].copy()
    df_store_events["store_name"] = collapse_store_entry_exit(df_store_events["zone_name"].astype(str))
    df_store_events["checkpoint"] = df_store_events["zone_type"].astype(str).str.lower().map(
        {"store_entry": "entry", "store_exit": "exit"}
    )

    df_store_pivot = (
        df_store_events
        .sort_values("timestamp")
        .drop_duplicates(subset=["reid_id", "store_name", "checkpoint"], keep="first")
        .pivot(index=["reid_id", "store_name"], columns="checkpoint", values="timestamp")
        .reset_index()
    )

    if "entry" not in df_store_pivot.columns or "exit" not in df_store_pivot.columns:
        st.info("No paired store entry/exit events found for the selected date range.")
    else:
        df_store_pivot = df_store_pivot.dropna(subset=["entry", "exit"])
        df_store_pivot["store_dwell_seconds"] = (
            df_store_pivot["exit"] - df_store_pivot["entry"]
        ).dt.total_seconds()
        # Guard against bad pairings (exit before entry) — shouldn't happen
        # given journey generation order, but keep the chart honest.
        df_store_pivot = df_store_pivot[df_store_pivot["store_dwell_seconds"] >= 0]

        if df_store_pivot.empty:
            st.info("No valid store entry→exit pairs found for the selected date range.")
        else:
            df_store_avg = (
                df_store_pivot.groupby("store_name", as_index=False)["store_dwell_seconds"]
                .mean()
                .sort_values("store_dwell_seconds", ascending=False)
                .reset_index(drop=True)
            )
            store_colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(df_store_avg["store_name"])}
            fig_store_dwell = px.bar(
                df_store_avg, x="store_name", y="store_dwell_seconds",
                color="store_name", color_discrete_map=store_colors,
                labels={"store_name": "Store", "store_dwell_seconds": "Avg Time In-Store (s)"},
                text_auto=".1f",
            )
            fig_store_dwell.update_layout(showlegend=False)
            brand_layout(fig_store_dwell)
            st.plotly_chart(fig_store_dwell, use_container_width=True)
            st.caption(
                f"Based on {len(df_store_pivot):,} completed entry→exit visits "
                f"across {df_store_avg['store_name'].nunique()} store(s)."
            )

    # ── Zone visit frequency heatmap (real floor plan overlay) ──
    st.subheader("Zone Visit Frequency Heatmap", anchor=False)
    # Collapse Store_X_Entry / Store_X_Exit into a single "Store_X" entry
    # (same logic as the Sankey and dwell time charts above) so each store
    # counts as one zone for visit frequency, not two separate checkpoints.
    df_zone_visits_src = df_journey.dropna(subset=["reid_id", "zone_name"]).copy()
    df_zone_visits_src["zone_name_clean"] = collapse_store_entry_exit(df_zone_visits_src["zone_name"].astype(str))
    zone_visits = (
        df_zone_visits_src
        .groupby("zone_name_clean")["reid_id"].nunique()
        .reset_index().rename(columns={"zone_name_clean": "zone_name", "reid_id": "unique_visitors"})
        .sort_values("unique_visitors", ascending=False).reset_index(drop=True)
    )

    # Normalized (0-1) zone center coordinates, hand-mapped from
    # mall_floor_plan.png (origin top-left, x right, y down — matches how
    # the image itself reads). Only zones that appear as boxes on the
    # floor plan are mapped; any zone_name not in this dict (e.g. a
    # generator zone added later that hasn't been added to the floor plan
    # yet) falls back to the old auto-grid layout so the chart never
    # breaks, it just silently won't show on the floor plan background.
    FLOOR_PLAN_COORDS = {
        # Exit row (top)
        "Mall_Exit_North":        (0.122, 0.108),
        "Mall_Exit_South":        (0.311, 0.108),
        "Mall_Exit_East":         (0.500, 0.108),
        "Mall_Exit_West":         (0.690, 0.108),
        "Mall_Exit_Underground":  (0.878, 0.108),
        # Left column
        "Renovation_Area_Restricted": (0.140, 0.255),
        "Store_A":                (0.140, 0.405),
        "Store_B":                (0.140, 0.540),
        "Pharmacy":                (0.140, 0.675),
        # Center column
        "Corridor_L1":             (0.500, 0.350),
        "Atrium_Walkway":          (0.500, 0.495),
        "Corridor_Ground":         (0.500, 0.645),
        "Grocery":                 (0.395, 0.730),
        "Clothes":                 (0.620, 0.730),
        # Right column
        "Customer_Seating_Area":   (0.820, 0.255),
        "Shoes":                   (0.860, 0.405),
        "Gaming":                  (0.860, 0.540),
        "Kitchenware":             (0.860, 0.675),
        # Entrance row (bottom)
        "Mall_Main_Entrance":      (0.122, 0.845),
        "Mall_Entrance_North":     (0.311, 0.845),
        "Mall_Entrance_South":     (0.500, 0.845),
        "Mall_Entrance_East":      (0.690, 0.845),
        "Mall_Entrance_West":      (0.878, 0.845),
    }

    zone_visits["x"] = zone_visits["zone_name"].map(lambda z: FLOOR_PLAN_COORDS.get(z, (None, None))[0])
    zone_visits["y"] = zone_visits["zone_name"].map(lambda z: FLOOR_PLAN_COORDS.get(z, (None, None))[1])

    unmapped = zone_visits[zone_visits["x"].isna()]["zone_name"].tolist()
    mapped   = zone_visits.dropna(subset=["x", "y"]).copy()

    if mapped.empty:
        st.info("No zones in the current data match the floor plan's known zones.")
    else:
        # Scale bubble size by unique_visitors: larger count = larger bubble.
        min_visitors = mapped["unique_visitors"].min()
        max_visitors = mapped["unique_visitors"].max()
        if max_visitors > min_visitors:
            mapped["bubble_size"] = 28 + (
                (mapped["unique_visitors"] - min_visitors) / (max_visitors - min_visitors)
            ) * 50
        else:
            mapped["bubble_size"] = 45

        fig_heatmap = go.Figure(go.Scatter(
            x=mapped["x"], y=mapped["y"],
            mode="markers+text",
            marker=dict(
                size=mapped["bubble_size"],
                color=mapped["unique_visitors"],
                colorscale=[[0.0, "#2563EB"], [0.5, "#A855F7"], [1.0, "#F43F5E"]],
                showscale=True,
                colorbar=dict(title="Unique Visitors", tickfont=dict(color=C["black"]), x=1.08),
                line=dict(width=2, color=C["white"]),
                opacity=0.85,
            ),
            text=mapped["zone_name"].str.replace("_", " ", regex=False),
            textposition="bottom center",
            textfont=dict(size=10, color=C["black"]),
            customdata=mapped["unique_visitors"],
            hovertemplate="<b>%{text}</b><br>Unique Visitors: %{customdata}<extra></extra>",
        ))

        # Floor plan as background image. Plotly's images use the same
        # axis coordinate space as the traces — sizing the image to
        # exactly [0,1] x [0,1] (with y reversed, since image row 0 is the
        # top) means our normalized FLOOR_PLAN_COORDS line up directly
        # without any separate pixel math at render time.
        #
        # IMPORTANT for aspect ratio: sizex/sizey are both 1 (the x-axis
        # and y-axis ranges), but Streamlit's container is wide (~900px+)
        # while this floor plan is nearly square (2252x2380px, ratio
        # 0.946). If we let Plotly fill a wide container at a fixed pixel
        # height with sizing="stretch", the image gets squashed
        # horizontally relative to its real proportions and every
        # FLOOR_PLAN_COORDS point drifts off its zone. Instead we fix the
        # *axis* aspect ratio itself (scaleanchor/scaleratio) so x and y
        # units are always equal regardless of container width — Plotly
        # then letterboxes rather than stretches.
        if FLOOR_PLAN_PATH:
            img_w, img_h = Image.open(FLOOR_PLAN_PATH).size
            img_aspect = img_w / img_h  # ~0.946 for this floor plan

            fig_heatmap.add_layout_image(
                dict(
                    source=Image.open(FLOOR_PLAN_PATH),
                    xref="x", yref="y",
                    x=0, y=0,
                    sizex=1, sizey=1,
                    xanchor="left", yanchor="top",
                    sizing="stretch",
                    layer="below",
                )
            )
            fig_heatmap.update_xaxes(
                visible=False, range=[0, 1],
                scaleanchor="y", scaleratio=img_aspect,
                constrain="domain",
            )
            fig_heatmap.update_yaxes(
                visible=False, range=[1, 0],  # reversed: 0 at top, matches image + our y coords
                constrain="domain",
            )
            plot_height = 700
        else:
            st.caption(f"⚠️ Floor plan image not found at `{FLOOR_PLAN_PATH or 'assets/mall_floor_plan_grayscale.png'}` — showing bubbles without background.")
            fig_heatmap.update_xaxes(visible=False, range=[0, 1])
            fig_heatmap.update_yaxes(visible=False, range=[1, 0])
            plot_height = 560

        fig_heatmap.update_layout(
            height=plot_height,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        brand_layout(fig_heatmap)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        if unmapped:
            st.caption(
                f"⚠️ {len(unmapped)} zone(s) not yet positioned on the floor plan and excluded from this view: "
                f"{', '.join(unmapped)}. Add their coordinates to `FLOOR_PLAN_COORDS` to include them."
            )


# ══════════════════════════════════════════════
# 3. CROWD ANALYSIS
# ══════════════════════════════════════════════
st.header("Crowd Analysis", divider="rainbow", anchor=False)

df_crowd_all = combined_de[
    (combined_de["camera_id"] == "CAM_CAFE_MAIN_AREA") &
    (combined_de["zone_name"] == "Customer_Seating_Area")
].copy()
df_crowd_all["timestamp_minute"] = df_crowd_all["timestamp"].dt.floor("min")

if df_crowd_all.empty:
    st.warning("No crowd data found for the selected date range.")
else:
    df_crowd_minute = (
        df_crowd_all.groupby(["timestamp_minute", "zone_name"], as_index=False)
        ["object_count_in_zone"].mean()
        .rename(columns={"object_count_in_zone": "avg_people"})
    )

    st.subheader("Crowd Count Over Time per Zone", anchor=False)
    fig_line_crowd = px.line(
        df_crowd_minute, x="timestamp_minute", y="avg_people", color="zone_name",
        color_discrete_sequence=PALETTE,
        labels={"timestamp_minute": "Time", "avg_people": "Avg People", "zone_name": "Zone"},
        markers=False,
        line_shape="spline",
    )
    fig_line_crowd.update_traces(line=dict(smoothing=0.8))
    brand_layout(fig_line_crowd)
    st.plotly_chart(fig_line_crowd, use_container_width=True)

    col9, col10 = st.columns(2)
    with col9:
        st.subheader("Cumulative Crowd Density Over Time", anchor=False)
        df_cumulative = df_crowd_minute.groupby("timestamp_minute", as_index=False)["avg_people"].sum()
        df_cumulative["cumulative_total"] = df_cumulative["avg_people"].cumsum()
        fig_area = px.area(
            df_cumulative, x="timestamp_minute", y="cumulative_total",
            labels={"timestamp_minute": "Time", "cumulative_total": "Cumulative Count"},
            color_discrete_sequence=[C["amber"]],
            line_shape="spline",
        )
        fig_area.update_traces(line=dict(smoothing=0.8))
        brand_layout(fig_area)
        st.plotly_chart(fig_area, use_container_width=True)

    with col10:
        st.subheader("Avg vs Peak Crowd Count per Zone", anchor=False)
        df_zone_stats = (
            df_crowd_minute.groupby("zone_name")["avg_people"]
            .agg(Average="mean", Peak="max").reset_index()
            .melt(id_vars="zone_name", var_name="Metric", value_name="People Count")
        )
        fig_bar_crowd = px.bar(
            df_zone_stats, x="zone_name", y="People Count", color="Metric", barmode="group",
            labels={"zone_name": "Zone"},
            color_discrete_map={"Average": C["orange"], "Peak": C["green"]},
            text_auto=".1f",
        )
        brand_layout(fig_bar_crowd)
        st.plotly_chart(fig_bar_crowd, use_container_width=True)


# ══════════════════════════════════════════════
# 4. INTRUSION INCIDENTS
# ══════════════════════════════════════════════
st.header("Intrusion Incidents", divider="rainbow", anchor=False)

df_intrusion = combined_de[
    (combined_de["camera_id"] == "CAM_CLOSED_CORRIDOR") &
    (combined_de["object_count_in_zone"] > 0)
].copy()

# The zone_id → zones join can come back NULL for some rows depending on
# source data quality, so backfill zone_name/zone_type explicitly for this
# camera — it only ever covers one zone.
df_intrusion["zone_name"] = "Renovation_Area_Restricted"
df_intrusion["zone_type"] = "restricted"

if df_intrusion.empty:
    st.warning("No intrusion events found for **CAM_CLOSED_CORRIDOR** in the selected date range.")
else:
    st.subheader("Intrusion Events Timeline", anchor=False)
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=df_intrusion["timestamp"],
        y=df_intrusion["zone_name"],
        mode="markers",
        name="Intrusion Event",
        marker=dict(
            size=10, symbol="diamond",
            color=df_intrusion["object_count_in_zone"],
            colorscale=[[0.0, C["amber"]], [0.5, C["orange"]], [1.0, C["red"]]],
            showscale=True,
            colorbar=dict(title="Count in Zone", tickfont=dict(color=C["black"])),
            line=dict(width=1, color=C["white"]),
        ),
        customdata=df_intrusion[["object_count_in_zone", "confidence"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>Time: %{x}<br>"
            "People in Zone: %{customdata[0]}<br>"
            "Confidence: %{customdata[1]:.2f}<extra></extra>"
        ),
    ))
    fig_timeline.update_layout(xaxis_title="Timestamp", yaxis_title="Zone")
    brand_layout(fig_timeline)
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.subheader("Incident Log", anchor=False)
    df_log = (
        df_intrusion[["timestamp", "zone_name", "object_count_in_zone", "confidence", "frame_number"]]
        .copy()
        .rename(columns={
            "timestamp":            "Timestamp",
            "zone_name":            "Zone",
            "object_count_in_zone": "People in Zone",
            "confidence":           "Confidence",
            "frame_number":         "Frame #",
        })
    )
    df_log["Timestamp"]  = df_log["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_log["Confidence"] = df_log["Confidence"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "—")
    st.dataframe(
        df_log, use_container_width=True,
        column_config={
            "People in Zone": st.column_config.NumberColumn("People in Zone"),
            "Confidence":     st.column_config.TextColumn("Confidence"),
            "Frame #":        st.column_config.NumberColumn("Frame #"),
        },
        hide_index=True,
    )