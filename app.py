"""Customer Segmentation Dashboard.

Run with:   streamlit run app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster import hierarchy

import segmentation as seg

PROJECT_AUTHOR = "dipeeka patidar"  # <- put your name here (shown in the footer)
ALGORITHMS = ["K-Means", "DBSCAN", "Hierarchical"]

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")
st.title("Customer Segmentation Dashboard")
st.caption("Preprocessing · K-Means · DBSCAN · Hierarchical · Elbow & Silhouette · Business insights")


# ------------------------------------------------------------ helpers
@st.cache_data
def get_sample():
    return seg.load_sample()


@st.cache_data
def cached_elbow(X, k_min, k_max):
    return seg.elbow_silhouette(X, k_min, k_max)


def label_text(labels):
    return np.where(labels == -1, "Noise", labels.astype(str))


def color_map(labels):
    palette = px.colors.qualitative.Safe
    cmap, i = {}, 0
    for value in sorted(set(labels.tolist())):
        if value == -1:
            cmap["Noise"] = "#9e9e9e"
        else:
            cmap[str(value)] = palette[i % len(palette)]
            i += 1
    return cmap


# ------------------------------------------------------------ sidebar
st.sidebar.header("1. Data")
source = st.sidebar.radio("Data source", ["Sample dataset", "Upload CSV"])
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Customer CSV file", type="csv")
    if uploaded is None:
        st.info("Upload a CSV with numeric customer columns (e.g. age, income, spending) or switch to the sample dataset.")
        st.stop()
    df = pd.read_csv(uploaded)
else:
    df = get_sample()

num_cols = seg.numeric_columns(df)
if len(num_cols) < 2:
    st.error("The data needs at least two numeric columns to cluster.")
    st.stop()

st.sidebar.header("2. Preprocessing")
preferred = [c for c in ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency", "Avg_Order_Value"] if c in num_cols]
features = st.sidebar.multiselect("Features to cluster on", num_cols, default=preferred or num_cols[:5])
if len(features) < 2:
    st.warning("Pick at least two features in the sidebar.")
    st.stop()
scaler_name = st.sidebar.selectbox("Scaling method", ["Standard (Z-score)", "Min-Max (0-1)"])
cap_outliers = st.sidebar.checkbox("Cap outliers (IQR rule)", value=True)

st.sidebar.header("3. Clustering")
algo = st.sidebar.selectbox("Algorithm to explore", ALGORITHMS)
k = st.sidebar.slider("Number of clusters (K-Means / Hierarchical)", 2, 10, 5)
linkage = st.sidebar.selectbox("Hierarchical linkage", ["ward", "complete", "average", "single"])
is_standard = scaler_name.startswith("Standard")
eps = st.sidebar.slider("DBSCAN eps (neighbourhood radius)", 0.05, 3.0, 0.7 if is_standard else 0.15, 0.05)
min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5)

# ------------------------------------------------------------ pipeline
clean, X, report = seg.preprocess(df, features, scaler_name, cap_outliers)


def get_labels(name):
    if name == "K-Means":
        return seg.run_kmeans(X, k)
    if name == "DBSCAN":
        return seg.run_dbscan(X, eps, min_samples)
    return seg.run_hierarchical(X, k, linkage)


labels = get_labels(algo)
cmap = color_map(labels)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. Data & Preprocessing", "2. Elbow & Silhouette", "3. Clusters", "4. Compare Methods", "5. Business Insights"]
)

# ------------------------------------------------ tab 1: preprocessing
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", len(df))
    c2.metric("Missing values filled (median)", report["missing_filled"])
    c3.metric("Outlier values capped", report["outliers_capped"])

    st.subheader("Raw data (first 10 rows)")
    st.dataframe(df.head(10))

    st.subheader("Effect of normalization")
    feat = st.selectbox("Feature to inspect", features)
    idx = features.index(feat)
    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(clean, x=feat, nbins=30, title=f"{feat}: original scale"), key="hist_raw"
    )
    right.plotly_chart(
        px.histogram(x=X[:, idx], nbins=30, title=f"{feat}: after {scaler_name}", labels={"x": feat}),
        key="hist_scaled",
    )

    st.subheader("Feature correlation")
    st.plotly_chart(
        px.imshow(clean.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
        key="corr",
    )
    st.subheader("Scaled data used for clustering (first 10 rows)")
    st.dataframe(pd.DataFrame(X, columns=features).head(10).round(3))

# ------------------------------------------------ tab 2: elbow + silhouette
with tab2:
    st.markdown(
        "K-Means needs the number of clusters in advance. Two standard checks help choose it:\n"
        "- **Elbow method**: plot inertia (within-cluster squared distance) against K and pick the bend where adding clusters stops helping much.\n"
        "- **Silhouette score** (-1 to 1): how well each customer fits its own cluster versus the nearest other one. Higher is better."
    )
    k_min, k_max = st.slider("Range of K to test", 2, 12, (2, 10))
    res = cached_elbow(X, k_min, k_max)
    elbow_k = seg.find_elbow(res["k"].values, res["Inertia (WCSS)"].values)
    best_k = int(res.loc[res["Silhouette"].idxmax(), "k"])

    m1, m2 = st.columns(2)
    m1.metric("Elbow method suggests K =", elbow_k)
    m2.metric("Best silhouette score at K =", best_k, f"score {res['Silhouette'].max():.3f}", delta_color="off")

    a, b = st.columns(2)
    fig = px.line(res, x="k", y="Inertia (WCSS)", markers=True, title="Elbow curve")
    fig.add_vline(x=elbow_k, line_dash="dash", annotation_text=f"elbow K={elbow_k}")
    a.plotly_chart(fig, key="elbow")
    fig = px.line(res, x="k", y="Silhouette", markers=True, title="Silhouette score by K")
    fig.add_vline(x=best_k, line_dash="dash", annotation_text=f"best K={best_k}")
    b.plotly_chart(fig, key="silhouette")

    st.dataframe(res.round(3))
    if elbow_k == best_k:
        st.success(f"Both methods agree: K = {best_k} is a good choice.")
    else:
        st.info(f"The methods disagree (elbow {elbow_k}, silhouette {best_k}). Try both and judge by the business insights tab.")

# ------------------------------------------------ tab 3: clusters
with tab3:
    st.subheader(f"{algo} result")
    metrics = seg.evaluate(X, labels)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters found", metrics["Clusters"])
    m2.metric("Silhouette (higher is better)", "n/a" if np.isnan(metrics["Silhouette"]) else f"{metrics['Silhouette']:.3f}")
    m3.metric("Davies-Bouldin (lower is better)", "n/a" if np.isnan(metrics["Davies-Bouldin"]) else f"{metrics['Davies-Bouldin']:.3f}")
    m4.metric("Noise points", metrics["Noise points"])

    view = st.radio("Plot axes", ["PCA (2D projection of all features)", "Choose two features"], horizontal=True)
    plot_df = clean.reset_index(drop=True).copy()
    plot_df["Cluster"] = label_text(labels)
    if view.startswith("PCA"):
        coords, var = seg.pca_2d(X)
        plot_df["axis_x"], plot_df["axis_y"] = coords[:, 0], coords[:, 1]
        x_label, y_label = f"PC1 ({var[0]:.0%} of variance)", f"PC2 ({var[1]:.0%} of variance)"
    else:
        sx, sy = st.columns(2)
        fx = sx.selectbox("X axis", features, index=0)
        fy = sy.selectbox("Y axis", features, index=1)
        plot_df["axis_x"], plot_df["axis_y"] = clean[fx].values, clean[fy].values
        x_label, y_label = fx, fy

    fig = px.scatter(
        plot_df, x="axis_x", y="axis_y", color="Cluster", color_discrete_map=cmap,
        hover_data=features, labels={"axis_x": x_label, "axis_y": y_label}, opacity=0.85,
    )
    fig.update_traces(marker=dict(size=9))
    st.plotly_chart(fig, key="scatter")

    sizes = plot_df["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Customers")
    st.plotly_chart(
        px.bar(sizes, x="Cluster", y="Customers", color="Cluster", color_discrete_map=cmap, title="Customers per cluster"),
        key="sizes",
    )

    if algo == "Hierarchical":
        st.subheader("Dendrogram")
        Z = hierarchy.linkage(X, method=linkage)
        fig_d, ax = plt.subplots(figsize=(10, 4))
        hierarchy.dendrogram(Z, truncate_mode="lastp", p=30, ax=ax)
        if 2 <= k <= len(Z):
            ax.axhline((Z[-k, 2] + Z[-(k - 1), 2]) / 2, color="red", linestyle="--", label=f"cut for K={k}")
            ax.legend()
        ax.set_xlabel("Customers (merged groups shown in brackets)")
        ax.set_ylabel("Merge distance")
        st.pyplot(fig_d)
        st.caption("Cutting the tree at the red line gives the chosen number of clusters.")

    if algo == "DBSCAN":
        st.subheader("k-distance plot (helps choose eps)")
        kd = seg.k_distances(X, min_samples)
        fig_k = px.line(x=np.arange(len(kd)), y=kd, labels={"x": "Customers sorted by distance", "y": f"Distance to neighbour #{min_samples}"})
        fig_k.add_hline(y=eps, line_dash="dash", annotation_text=f"eps = {eps}")
        st.plotly_chart(fig_k, key="kdist")
        st.caption("A good eps sits near the 'knee' where the curve starts rising steeply. Points in no dense area are labelled Noise.")

# ------------------------------------------------ tab 4: compare
with tab4:
    st.subheader("All three algorithms, side by side")
    st.caption("Uses the K, linkage, eps and min_samples values from the sidebar.")
    all_labels = {name: get_labels(name) for name in ALGORITHMS}
    table = pd.DataFrame.from_dict({n: seg.evaluate(X, l) for n, l in all_labels.items()}, orient="index")
    st.dataframe(table.round(3))

    coords, var = seg.pca_2d(X)
    cols = st.columns(3)
    for col, (name, lab) in zip(cols, all_labels.items()):
        pdf = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": label_text(lab)})
        fig = px.scatter(pdf, x="PC1", y="PC2", color="Cluster", color_discrete_map=color_map(lab), title=name, opacity=0.85)
        fig.update_layout(showlegend=False)
        col.plotly_chart(fig, key=f"cmp_{name}")

    st.markdown(
        "**How to read this**\n"
        "- **K-Means**: fast, needs K, finds round similar-sized groups.\n"
        "- **Hierarchical**: needs K (or a cut height), shows how groups merge, good for small and medium data.\n"
        "- **DBSCAN**: no K needed, finds dense groups of any shape and flags unusual customers as noise, but is sensitive to eps."
    )

# ------------------------------------------------ tab 5: business insights
with tab5:
    st.subheader(f"Business insights ({algo})")
    profile, segments, overview = seg.build_segments(clean, labels)
    for line in overview:
        st.markdown("- " + line)

    st.markdown("#### Segment profile (average of each feature)")
    st.dataframe(profile)

    left, right = st.columns(2)
    share_cols = ["Share (%)"] + (["Revenue share (%)"] if "Revenue share (%)" in profile.columns else [])
    share_df = profile.reset_index()[["Segment"] + share_cols].melt(id_vars="Segment", var_name="Metric", value_name="Percent")
    left.plotly_chart(
        px.bar(share_df, x="Segment", y="Percent", color="Metric", barmode="group", title="Customer share vs revenue share"),
        key="share",
    )

    means = profile[features]
    spread = (means.max() - means.min()).replace(0, 1)
    norm = (means - means.min()) / spread
    radar = go.Figure()
    for c in norm.index:
        vals = norm.loc[c].tolist()
        radar.add_trace(
            go.Scatterpolar(
                r=vals + vals[:1], theta=features + features[:1], name=profile.loc[c, "Segment"],
                fill="toself", opacity=0.55, line_color=cmap.get("Noise" if c == -1 else str(c)),
            )
        )
    radar.update_layout(title="Segment fingerprints (0 = lowest segment, 1 = highest)", polar=dict(radialaxis=dict(range=[0, 1])))
    right.plotly_chart(radar, key="radar")

    st.markdown("#### Segment playbook")
    for s in segments:
        with st.expander(f"{s['segment']}  -  {s['size']} customers ({s['share']}%)", expanded=True):
            st.markdown("**What defines them:** " + "; ".join(s["traits"]) + ".")
            st.markdown("**Recommended action:** " + s["action"])

    out = df.copy()
    out["Cluster"] = labels
    out["Segment"] = out["Cluster"].map(profile["Segment"])
    st.download_button(
        "Download customers with segments (CSV)", out.to_csv(index=False).encode("utf-8"),
        file_name="customers_segmented.csv", mime="text/csv",
    )

st.divider()
st.caption(f"Customer Segmentation project by {PROJECT_AUTHOR} · built with scikit-learn, Plotly and Streamlit")
