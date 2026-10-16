# Customer Segmentation Dashboard

A small Streamlit project that groups customers into segments and turns the groups into business advice.

## What it covers

| Topic | Where |
|---|---|
| Data preprocessing and normalization (median fill, IQR outlier capping, Z-score or Min-Max scaling) | `segmentation.py` -> `preprocess()`, tab 1 |
| K-Means, DBSCAN and Hierarchical clustering | `segmentation.py` -> `run_*()`, tabs 3 and 4 |
| Elbow method and Silhouette score | `segmentation.py` -> `elbow_silhouette()`, tab 2 |
| Cluster visualization (PCA scatter, feature scatter, dendrogram, k-distance plot, radar chart) | tabs 3 and 5 |
| Business insights (segment names, revenue share, recommended actions) | `segmentation.py` -> `build_segments()`, tab 5 |
| Interactive dashboard | `app.py` (Streamlit) |

## How to run

1. Install Python 3.9 or newer from python.org (tick "Add Python to PATH" on Windows).
2. Unzip this folder and open a terminal inside it.
3. Install the packages and start the app:

```
pip install -r requirements.txt
streamlit run app.py
```

Shortcut: on Windows double-click `run_app.bat`; on Mac/Linux run `bash run_app.sh`.

The dashboard opens in your browser at http://localhost:8501.

If `streamlit` is not recognised, use `python -m streamlit run app.py` instead.

## Files

```
customer_segmentation/
  app.py             Streamlit dashboard (the interface)
  segmentation.py    All the data science logic
  generate_data.py   Creates the sample dataset (python generate_data.py)
  data/customers.csv Sample data: 500 synthetic customers
  requirements.txt   Packages needed
  run_app.bat / run_app.sh   One-click launchers
```

## Using your own data

Choose "Upload CSV" in the sidebar. The file needs at least two numeric columns. Columns whose name ends in "ID" are ignored automatically. Segment names and revenue estimates work best if column names contain words like `income`, `spend`, `frequency` and `order`; otherwise segments get generic names based on their most unusual feature.

## Sample data columns

`CustomerID, Gender, Age, Annual_Income (in thousands), Spending_Score (1-100), Purchase_Frequency (orders per year), Avg_Order_Value`.
The file has a few missing values and outliers on purpose so the preprocessing step has something to fix.

## Tips for the report / viva

- **Why scale first?** K-Means, DBSCAN and Hierarchical all use distances, so a feature in thousands (income) would swamp one in tens (age).
- **Elbow vs silhouette:** elbow looks for diminishing returns in inertia; silhouette measures separation. On the sample data both point to K = 5.
- **Why DBSCAN finds different results:** it has no K, it labels sparse customers as noise, and its outcome depends heavily on `eps`.
- **Davies-Bouldin index:** lower is better (clusters are compact and far apart).

To put your name in the footer, edit `PROJECT_AUTHOR` at the top of `app.py`.
