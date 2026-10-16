"""Core logic for the Customer Segmentation project (no Streamlit code in here)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "data" / "customers.csv"


# ---------------------------------------------------------------- 1. data
def load_sample():
    """Load the bundled sample dataset (creates it if the CSV is missing)."""
    if not DATA_PATH.exists():
        from generate_data import make_customers

        DATA_PATH.parent.mkdir(exist_ok=True)
        make_customers().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def numeric_columns(df):
    """Numeric columns that are not ID columns."""
    cols = df.select_dtypes(include="number").columns
    return [c for c in cols if not str(c).lower().endswith("id")]


def preprocess(df, features, scaler="Standard (Z-score)", cap_outliers=True):
    """Fill missing values, optionally cap outliers (IQR rule), then scale.

    Returns (clean_df, scaled_array, report_dict).
    """
    data = df[features].apply(pd.to_numeric, errors="coerce")
    report = {"missing_filled": int(data.isna().sum().sum()), "outliers_capped": 0}
    data = data.fillna(data.median())

    if cap_outliers:
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        report["outliers_capped"] = int(((data < low) | (data > high)).sum().sum())
        data = data.clip(lower=low, upper=high, axis=1)

    model = StandardScaler() if scaler.startswith("Standard") else MinMaxScaler()
    return data, model.fit_transform(data), report


# ----------------------------------------------------- 2. choosing K
def elbow_silhouette(X, k_min=2, k_max=10):
    """Run K-Means for each K and record inertia (elbow) and silhouette score."""
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append(
            {
                "k": k,
                "Inertia (WCSS)": km.inertia_,
                "Silhouette": silhouette_score(X, km.labels_),
            }
        )
    return pd.DataFrame(rows)


def find_elbow(ks, inertias):
    """Elbow = point farthest from the straight line joining first and last points."""
    ks, inertias = np.asarray(ks, float), np.asarray(inertias, float)
    if len(ks) < 3 or inertias.max() == inertias.min():
        return int(ks[0])
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    dist = np.abs(dx * (y - y[0]) - dy * (x - x[0])) / np.hypot(dx, dy)
    return int(ks[np.argmax(dist)])


# ------------------------------------------------------- 3. clustering
def run_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)


def run_dbscan(X, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def run_hierarchical(X, k, linkage="ward"):
    return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)


def evaluate(X, labels):
    """Cluster quality metrics. DBSCAN noise points (label -1) are ignored."""
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels[mask].tolist()))
    out = {
        "Clusters": n_clusters,
        "Noise points": int((~mask).sum()),
        "Silhouette": np.nan,
        "Davies-Bouldin": np.nan,
    }
    if n_clusters >= 2 and mask.sum() > n_clusters:
        out["Silhouette"] = silhouette_score(X[mask], labels[mask])
        out["Davies-Bouldin"] = davies_bouldin_score(X[mask], labels[mask])
    return out


def pca_2d(X):
    """Project the scaled data to 2 dimensions for plotting."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def k_distances(X, k):
    """Sorted distance of every point to its k-th neighbour (helps pick DBSCAN eps)."""
    dist, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    return np.sort(dist[:, -1])


# --------------------------------------------------- 4. business insights
def _find(features, *keys):
    for f in features:
        if any(k in f.lower() for k in keys):
            return f
    return None


def _level(z):
    return 1 if z > 0.4 else (-1 if z < -0.4 else 0)


# (income level, spending level) -> (segment name, recommended action)
PLAYBOOK = {
    (1, 1): ("Premium Spenders", "Reward and retain: VIP/loyalty tier, early access to launches, personalised service."),
    (1, 0): ("Affluent Customers", "Upsell premium ranges and bundles; invite them into the loyalty programme."),
    (1, -1): ("Cautious High Earners", "Biggest growth opportunity: high income but low spend. Use personalised recommendations and quality-focused campaigns."),
    (0, 1): ("Active Spenders", "Keep them engaged with new arrivals, flash sales and referral rewards."),
    (0, 0): ("Mainstream Customers", "Grow steadily: seasonal campaigns, cross-selling and nudges to buy more often."),
    (0, -1): ("Occasional Buyers", "Re-engage with reminders, welcome-back offers and easy first-purchase incentives."),
    (-1, 1): ("Enthusiastic Spenders", "They spend a lot relative to income: promote EMI/instalment options and cashback, avoid over-discounting."),
    (-1, 0): ("Value Seekers", "Value packs, coupons and price-match messaging keep them loyal at low cost."),
    (-1, -1): ("Budget Conscious", "Low-cost retention only: occasional coupons, email campaigns and clear value messaging."),
}
GENERIC_ACTION = "Study this group further and test small targeted campaigns before scaling up."
NOISE_ACTION = "Review individually: these may be unusual big spenders or data errors."


def _name_and_action(cluster, z, income, spend):
    if cluster == -1:
        return "Outliers (noise)", NOISE_ACTION
    if income and spend:
        return PLAYBOOK[(_level(z[income]), _level(z[spend]))]
    top = z.abs().idxmax()
    return f"{'High' if z[top] > 0 else 'Low'} {top}", GENERIC_ACTION


def build_segments(clean, labels):
    """Turn cluster labels into a business-friendly profile.

    Returns (profile_df, segments, overview):
      profile_df - one row per cluster with name, size, revenue share, feature means
      segments   - list of dicts (name, size, traits, recommended action)
      overview   - list of headline insight sentences
    """
    d = clean.copy()
    d["Cluster"] = np.asarray(labels)
    feats = list(clean.columns)
    mean_all, std_all = clean.mean(), clean.std().replace(0, 1)

    income = _find(feats, "income")
    spend = _find(feats, "spend", "score")
    freq = _find(feats, "freq")
    aov = _find(feats, "order", "basket")

    rev_note = None
    if freq and aov:
        d["_rev"] = d[freq] * d[aov]
        rev_note = f"Revenue share is estimated as {freq} x {aov}."
    elif spend:
        d["_rev"] = d[spend]
        rev_note = f"Revenue share is approximated by {spend}."

    grp = d.groupby("Cluster")
    means = grp[feats].mean()
    sizes = grp.size()

    names, raw_names, segments = {}, {}, []
    for c in means.index:
        z = (means.loc[c] - mean_all) / std_all
        raw, action = _name_and_action(c, z, income, spend)
        raw_names[c] = raw
        names[c] = raw if c == -1 else f"C{c}: {raw}"

        traits = []
        for f in z.abs().sort_values(ascending=False).index[:3]:
            if abs(z[f]) >= 0.3:
                direction = "higher" if z[f] > 0 else "lower"
                traits.append(f"{direction} {f} ({means.loc[c, f]:.1f} vs {mean_all[f]:.1f} average)")
        segments.append(
            {
                "cluster": c,
                "segment": names[c],
                "size": int(sizes[c]),
                "share": round(sizes[c] / len(d) * 100, 1),
                "traits": traits or ["close to the average on every feature"],
                "action": action,
            }
        )

    prof = means.round(2)
    prof.insert(0, "Segment", pd.Series(names))
    prof.insert(1, "Customers", sizes)
    prof.insert(2, "Share (%)", (sizes / len(d) * 100).round(1))
    has_rev = rev_note is not None
    if has_rev:
        prof.insert(3, "Revenue share (%)", (grp["_rev"].sum() / d["_rev"].sum() * 100).round(1))

    overview = []
    big = prof["Customers"].idxmax()
    overview.append(
        f"**{prof.loc[big, 'Segment']}** is the largest group: {prof.loc[big, 'Customers']} customers ({prof.loc[big, 'Share (%)']}%)."
    )
    if has_rev:
        top = prof["Revenue share (%)"].idxmax()
        overview.append(
            f"**{prof.loc[top, 'Segment']}** brings the most revenue: {prof.loc[top, 'Revenue share (%)']}% of the total from {prof.loc[top, 'Share (%)']}% of customers."
        )
    for c, raw in raw_names.items():
        if raw == "Cautious High Earners":
            overview.append(f"**{names[c]}** has high income but low spending, which makes it the best growth opportunity.")
    if rev_note:
        overview.append(f"*{rev_note}*")
    return prof, segments, overview
