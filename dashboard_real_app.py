import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Dashboard",
    page_icon="👁️",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
conn = st.connection("postgresql", type="sql")

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
        value=datetime.date(2026, 1, 31),
        format="DD/MM/YYYY",
        key="filter_end",
    )

    if filter_start > filter_end:
        st.error("⚠️ Start date must be on or before end date.")
        st.stop()

    # Convert to datetime for SQL: start = 00:00:00, end = 23:59:59
    dt_start = datetime.datetime.combine(filter_start, datetime.time.min)  # 00:00:00
    dt_end   = datetime.datetime.combine(filter_end,   datetime.time.max)  # 23:59:59

    st.divider()
    if filter_start == filter_end:
        st.info(f"Showing: **{filter_start.strftime('%d %b %Y')}** only")
    else:
        st.info(f"Showing: **{filter_start.strftime('%d %b %Y')}** → **{filter_end.strftime('%d %b %Y')}**")

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("Retail Dashboard (Real Data)", anchor=False)

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
        gridcolor="#ffffff",
        zerolinecolor="#c4cad6",
        title_font=dict(color=C["black"]),
        tickfont=dict(color=C["black"]),
    )
    fig.update_yaxes(
        gridcolor="#ffffff",
        zerolinecolor="#c4cad6",
        title_font=dict(color=C["black"]),
        tickfont=dict(color=C["black"]),
    )
    return fig

def hex_to_rgba(hex_color, alpha=0.35):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ══════════════════════════════════════════════
# 1. EMOTION ANALYSIS
# ══════════════════════════════════════════════
st.header("Emotion Analysis", divider="rainbow", anchor=False)

df_emotion = conn.query(
    """
    SELECT
        ee.emotion,
        ee.confidence,
        de.timestamp,
        de.zone_id,
        z.zone_name
    FROM emotion_events ee
    JOIN detection_events de ON ee.unique_detection_id = de.unique_detection_id
    LEFT JOIN zones z ON de.zone_id = z.zone_id
    WHERE de.timestamp >= :start AND de.timestamp <= :end
    ORDER BY de.timestamp
    """,
    params={"start": dt_start, "end": dt_end},
    ttl="10m",
)
df_emotion["timestamp"] = pd.to_datetime(df_emotion["timestamp"])

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
        emotion_counts = df_emotion["emotion"].str.lower().value_counts().reset_index()
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
    df_ts = df_emotion.copy()
    df_ts["minute"] = df_ts["timestamp"].dt.floor("min")
    df_comp = df_ts.groupby(["minute", "emotion"]).size().reset_index(name="count")
    df_pivot = df_comp.pivot(index="minute", columns="emotion", values="count").fillna(0)
    row_totals = df_pivot.sum(axis=1).replace(0, 1)
    df_pct = (df_pivot.div(row_totals, axis=0) * 100).reset_index()
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
    happy_rows = df_emotion[df_emotion["emotion"].str.lower() == "happy"]
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

df_journey = conn.query(
    """
    SELECT
        de.reid_id,
        de.zone_id,
        de.previous_zone_id,
        de.movement_type,
        de.dwell_time,
        de.timestamp,
        z_cur.zone_name  AS zone_name,
        z_prev.zone_name AS prev_zone_name,
        z_cur.zone_type  AS zone_type
    FROM detection_events de
    LEFT JOIN zones z_cur  ON de.zone_id          = z_cur.zone_id
    LEFT JOIN zones z_prev ON de.previous_zone_id = z_prev.zone_id
    WHERE de.reid_id IS NOT NULL
      AND de.timestamp >= :start AND de.timestamp <= :end
    ORDER BY de.timestamp
    """,
    params={"start": dt_start, "end": dt_end},
    ttl="10m",
)
df_journey["timestamp"] = pd.to_datetime(df_journey["timestamp"])

if df_journey.empty:
    st.warning("No customer journey data found for the selected date range.")
else:
    st.subheader("Customer Flow Between Zones", anchor=False)
    transitions = (
        df_journey.dropna(subset=["previous_zone_id", "zone_id"])
        .query("previous_zone_id != zone_id")
        .drop_duplicates(subset=["reid_id", "prev_zone_name", "zone_name"])
        .groupby(["prev_zone_name", "zone_name"], as_index=False)["reid_id"]
        .nunique()
        .rename(columns={"prev_zone_name": "source", "zone_name": "target", "reid_id": "value"})
    )

    if not transitions.empty:
        all_zones = list(pd.unique(transitions[["source", "target"]].values.ravel()))
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

    st.subheader("Avg Dwell Time – Stores", anchor=False)
    df_store_raw = conn.query(
        """
        SELECT
            de.reid_id,
            z.zone_name,
            z.zone_type,
            de.timestamp
        FROM detection_events de
        JOIN zones z ON de.zone_id = z.zone_id
        WHERE de.reid_id IS NOT NULL
          AND LOWER(z.zone_type) NOT LIKE '%corridor%'
          AND LOWER(z.zone_type) NOT LIKE '%intrusion%'
          AND de.timestamp >= :start AND de.timestamp <= :end
        ORDER BY de.timestamp
        """,
        params={"start": dt_start, "end": dt_end},
        ttl="10m",
    )
    df_store_raw["timestamp"] = pd.to_datetime(df_store_raw["timestamp"])
    df_store_raw["store_name"] = df_store_raw["zone_name"].str.replace(
        r"^(Store_A)_(Entry|Exit)$", r"\1", regex=True
    )
    df_store_raw = df_store_raw[
        ~df_store_raw["store_name"].str.contains(r"(?i)(entrance|exit)", regex=True)
    ]
    df_store_dwell = (
        df_store_raw.groupby(["reid_id", "store_name"])["timestamp"]
        .agg(enter_ts="min", exit_ts="max").reset_index()
    )
    df_store_dwell["dwell_seconds"] = (
        df_store_dwell["exit_ts"] - df_store_dwell["enter_ts"]
    ).dt.total_seconds()
    df_store_avg = (
        df_store_dwell[df_store_dwell["dwell_seconds"] >= 0]
        .groupby("store_name", as_index=False)["dwell_seconds"]
        .mean().sort_values("dwell_seconds", ascending=False).reset_index(drop=True)
    )
    if not df_store_avg.empty:
        store_colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(df_store_avg["store_name"])}
        fig_store = px.bar(
            df_store_avg, x="store_name", y="dwell_seconds",
            color="store_name", color_discrete_map=store_colors,
            labels={"store_name": "Store", "dwell_seconds": "Avg Dwell (s)"}, text_auto=".1f",
        )
        fig_store.update_layout(showlegend=False)
        brand_layout(fig_store)
        st.plotly_chart(fig_store, use_container_width=True)

    st.subheader("Zone Visit Frequency Heatmap", anchor=False)
    zone_visits = (
        df_journey.dropna(subset=["reid_id", "zone_name"])
        .groupby("zone_name")["reid_id"].nunique()
        .reset_index().rename(columns={"reid_id": "unique_visitors"})
        .sort_values("unique_visitors", ascending=False).reset_index(drop=True)
    )
    n = len(zone_visits)
    cols_per_row = max(1, int(np.ceil(np.sqrt(n))))
    zone_visits["grid_col"] = zone_visits.index % cols_per_row
    zone_visits["grid_row"] = zone_visits.index // cols_per_row

    fig_heatmap = go.Figure(go.Scatter(
        x=zone_visits["grid_col"], y=zone_visits["grid_row"],
        mode="markers+text",
        marker=dict(
            size=60, color=zone_visits["unique_visitors"],
            colorscale=[[0.0, "#2563EB"], [0.5, "#A855F7"], [1.0, "#F43F5E"]],
            showscale=True,
            colorbar=dict(title="Unique Visitors", tickfont=dict(color=C["black"])),
            line=dict(width=1, color=C["white"]),
        ),
        text=zone_visits["zone_name"].str.replace("_", " ", regex=False),
        textposition="middle center",
        textfont=dict(size=9, color=C["white"]),
        customdata=zone_visits["unique_visitors"],
        hovertemplate="<b>%{text}</b><br>Unique Visitors: %{customdata}<extra></extra>",
    ))
    fig_heatmap.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed"), height=380,
    )
    brand_layout(fig_heatmap)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption("💡 Replace with your actual floor plan image when available.")


# ══════════════════════════════════════════════
# 3. CROWD ANALYSIS
# ══════════════════════════════════════════════
st.header("Crowd Analysis", divider="rainbow", anchor=False)

df_crowd_all = conn.query(
    """
    SELECT
        de.timestamp,
        de.camera_id,
        de.object_count_in_zone,
        z.zone_name
    FROM detection_events de
    LEFT JOIN zones z ON de.zone_id = z.zone_id
    WHERE de.camera_id = 'CAM_CAFE_MAIN_AREA'
      AND de.timestamp >= :start AND de.timestamp <= :end
    ORDER BY de.timestamp
    """,
    params={"start": dt_start, "end": dt_end},
    ttl="10m",
)
df_crowd_all["timestamp"] = pd.to_datetime(df_crowd_all["timestamp"])
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
    )
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
        )
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

# Source: detection_events only (notifications table is empty).
# An intrusion event = any row where camera_id = 'CAM_CLOSED_CORRIDOR'
# and object_count_in_zone > 0.
df_intrusion = conn.query(
    """
    SELECT
        de.timestamp,
        de.camera_id,
        de.object_count_in_zone,
        de.confidence,
        de.frame_number,
        z.zone_name
    FROM detection_events de
    LEFT JOIN zones z ON de.zone_id = z.zone_id
    WHERE de.camera_id = 'CAM_CLOSED_CORRIDOR'
      AND de.object_count_in_zone > 0
      AND de.timestamp >= :start AND de.timestamp <= :end
    ORDER BY de.timestamp
    """,
    params={"start": dt_start, "end": dt_end},
    ttl="10m",
)
df_intrusion["timestamp"] = pd.to_datetime(df_intrusion["timestamp"])

if df_intrusion.empty:
    st.warning("No intrusion events found for **CAM_CLOSED_CORRIDOR** in the selected date range.")
else:
    # ── 4-A  Timeline – intrusion events over time ──
    st.subheader("Intrusion Events Timeline", anchor=False)

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=df_intrusion["timestamp"],
        y=df_intrusion["zone_name"],
        mode="markers",
        name="Intrusion Event",
        marker=dict(
            size=10,
            symbol="diamond",
            color=df_intrusion["object_count_in_zone"],
            colorscale=[[0.0, C["amber"]], [0.5, C["orange"]], [1.0, C["red"]]],
            showscale=True,
            colorbar=dict(title="Count in Zone", tickfont=dict(color=C["black"])),
            line=dict(width=1, color=C["white"]),
        ),
        customdata=df_intrusion[["object_count_in_zone", "confidence"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Time: %{x}<br>"
            "People in Zone: %{customdata[0]}<br>"
            "Confidence: %{customdata[1]:.2f}"
            "<extra></extra>"
        ),
    ))
    fig_timeline.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Zone",
    )
    brand_layout(fig_timeline)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── 4-B  Incident log table ──
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
        df_log,
        use_container_width=True,
        column_config={
            "People in Zone": st.column_config.NumberColumn("People in Zone"),
            "Confidence":     st.column_config.TextColumn("Confidence"),
            "Frame #":        st.column_config.NumberColumn("Frame #"),
        },
        hide_index=True,
    )