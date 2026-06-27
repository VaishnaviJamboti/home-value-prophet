"""
=============================================================
  LABMENTIX INTERNSHIP — Real Estate Investment Advisor
  Step 2: EDA — All 20 Charts (Interactive with Plotly)
  Role   : Data Analytics / Business Analytics Intern
  Author : Your Name
=============================================================
HOW TO RUN:
  1. pip install pandas numpy plotly
  2. Place this file in the SAME folder as india_housing_prices.csv
  3. python eda_interactive_plotly.py
  4. Each chart opens in your browser — hover, zoom, click legends!
=============================================================
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOAD & QUICK CLEAN
# ─────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("india_housing_cleaned.csv")

# Basic cleaning
df.dropna(subset=["Price_in_Lakhs", "Size_in_SqFt"], inplace=True)
df["Price_in_Lakhs"] = pd.to_numeric(df["Price_in_Lakhs"], errors="coerce")
df["Size_in_SqFt"]   = pd.to_numeric(df["Size_in_SqFt"],   errors="coerce")
df.dropna(subset=["Price_in_Lakhs", "Size_in_SqFt"], inplace=True)

# Recalculate Price_per_SqFt if needed
if df["Price_per_SqFt"].max() < 500:
    df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"]

# Cap outliers (1st–99th percentile)
for col in ["Price_in_Lakhs", "Price_per_SqFt", "Size_in_SqFt"]:
    lo, hi = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(lo, hi)

# Engineered features (if missing)
if "Age_of_Property" not in df.columns:
    df["Age_of_Property"] = 2024 - df["Year_Built"]

transport_map = {"Excellent": 4, "Good": 3, "Average": 2, "Poor": 1}
if "Transport_Score" not in df.columns:
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map(transport_map).fillna(2)

if "Infrastructure_Score" not in df.columns:
    df["Infrastructure_Score"] = (
        df["Transport_Score"] +
        df["Nearby_Schools"].fillna(df["Nearby_Schools"].median()) +
        df["Nearby_Hospitals"].fillna(df["Nearby_Hospitals"].median())
    ) / 3

if "Good_Investment" not in df.columns:
    city_appreciation = df.groupby("City")["Price_in_Lakhs"].pct_change().fillna(0.08)
    df["Appreciation_Rate"] = city_appreciation.clip(0.03, 0.20)
    df["Good_Investment"] = ((df["Appreciation_Rate"] > 0.08) & (df["Infrastructure_Score"] > 3)).astype(int)

if "Price_After_5Yrs" not in df.columns:
    df["Price_After_5Yrs"] = df["Price_in_Lakhs"] * ((1 + df.get("Appreciation_Rate", 0.08)) ** 5)

print(f"Dataset ready: {len(df):,} rows × {df.shape[1]} columns\n")
print("Opening charts in your browser — each chart opens as a separate tab.\n")

TEMPLATE = "plotly_white"

# ══════════════════════════════════════════════════════════
# GROUP 1 — CHARTS 1–5: Price & Size Analysis
# ══════════════════════════════════════════════════════════

# Chart 1 — Distribution of Property Prices
print("Chart 1: Price Distribution...")
mean_price = df["Price_in_Lakhs"].mean()
med_price  = df["Price_in_Lakhs"].median()
fig1 = px.histogram(
    df, x="Price_in_Lakhs", nbins=80,
    title="Chart 1 — Distribution of Property Prices (₹ Lakhs)",
    labels={"Price_in_Lakhs": "Price (₹ Lakhs)"},
    color_discrete_sequence=["#4C72B0"],
    template=TEMPLATE,
    marginal="box",
    opacity=0.8
)
fig1.add_vline(x=mean_price, line_dash="dash", line_color="red",
               annotation_text=f"Mean: ₹{mean_price:.1f}L", annotation_position="top right")
fig1.add_vline(x=med_price, line_dash="dot", line_color="green",
               annotation_text=f"Median: ₹{med_price:.1f}L", annotation_position="top left")
fig1.update_layout(bargap=0.05)
fig1.show()

# Chart 2 — Distribution of Property Sizes
print("Chart 2: Size Distribution...")
fig2 = px.histogram(
    df, x="Size_in_SqFt", nbins=80,
    title="Chart 2 — Distribution of Property Sizes (SqFt)",
    labels={"Size_in_SqFt": "Size (Square Feet)"},
    color_discrete_sequence=["#DD8452"],
    template=TEMPLATE,
    marginal="violin",
    opacity=0.8
)
fig2.update_layout(bargap=0.05)
fig2.show()

# Chart 3 — Price per SqFt by Property Type
print("Chart 3: Price/SqFt by Property Type...")
fig3 = px.box(
    df, x="Property_Type", y="Price_per_SqFt",
    title="Chart 3 — Price per SqFt by Property Type",
    labels={"Price_per_SqFt": "Price per SqFt (₹)", "Property_Type": "Property Type"},
    color="Property_Type",
    template=TEMPLATE,
    points="outliers"
)
fig3.update_layout(showlegend=False)
fig3.show()

# Chart 4 — Property Size vs Price (Scatter)
print("Chart 4: Size vs Price Scatter...")
df_sample = df.sample(min(5000, len(df)), random_state=42)
fig4 = px.scatter(
    df_sample,
    x="Size_in_SqFt", y="Price_in_Lakhs",
    color="Property_Type",
    title="Chart 4 — Property Size vs Price",
    labels={"Size_in_SqFt": "Size (SqFt)", "Price_in_Lakhs": "Price (₹ Lakhs)"},
    template=TEMPLATE,
    opacity=0.6,
    hover_data=["City", "BHK"]
)
# Manual trendline using numpy (no statsmodels needed)
x_vals = df_sample["Size_in_SqFt"].values
y_vals = df_sample["Price_in_Lakhs"].values
m, b = np.polyfit(x_vals, y_vals, 1)
x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
y_line = m * x_line + b
fig4.add_trace(go.Scatter(
    x=x_line, y=y_line,
    mode="lines", name="Trend Line",
    line=dict(color="black", width=2, dash="dash")
))
fig4.show()

# Chart 5 — Outliers: Price per SqFt & Size
print("Chart 5: Outlier Boxplots...")
fig5 = make_subplots(rows=1, cols=2,
                     subplot_titles=["Price per SqFt (₹)", "Property Size (SqFt)"])
fig5.add_trace(go.Box(y=df["Price_per_SqFt"], name="Price/SqFt",
                       marker_color="#4C72B0", boxpoints="outliers"), row=1, col=1)
fig5.add_trace(go.Box(y=df["Size_in_SqFt"], name="Size",
                       marker_color="#DD8452", boxpoints="outliers"), row=1, col=2)
fig5.update_layout(title="Chart 5 — Outlier Detection: Price/SqFt & Size",
                   template=TEMPLATE, showlegend=False)
fig5.show()

# ══════════════════════════════════════════════════════════
# GROUP 2 — CHARTS 6–10: Location-Based Analysis
# ══════════════════════════════════════════════════════════

# Chart 6 — Avg Price per SqFt by State
print("Chart 6: Avg Price/SqFt by State...")
state_avg = df.groupby("State")["Price_per_SqFt"].mean().reset_index().sort_values("Price_per_SqFt", ascending=False)
fig6 = px.bar(
    state_avg, x="State", y="Price_per_SqFt",
    title="Chart 6 — Average Price per SqFt by State",
    labels={"Price_per_SqFt": "Avg Price/SqFt (₹)", "State": "State"},
    color="Price_per_SqFt", color_continuous_scale="Blues",
    template=TEMPLATE, text_auto=".0f"
)
fig6.update_traces(textposition="outside")
fig6.show()

# Chart 7 — Avg Property Price by City (Top 20)
print("Chart 7: Avg Price by City...")
city_avg = df.groupby("City")["Price_in_Lakhs"].mean().reset_index().sort_values("Price_in_Lakhs", ascending=False).head(20)
fig7 = px.bar(
    city_avg, x="City", y="Price_in_Lakhs",
    title="Chart 7 — Top 20 Cities by Average Property Price (₹ Lakhs)",
    labels={"Price_in_Lakhs": "Avg Price (₹ Lakhs)", "City": "City"},
    color="Price_in_Lakhs", color_continuous_scale="Oranges",
    template=TEMPLATE, text_auto=".1f"
)
fig7.update_layout(xaxis_tickangle=-45)
fig7.update_traces(textposition="outside")
fig7.show()

# Chart 8 — Median Age of Properties by Locality (Top 20)
print("Chart 8: Median Property Age by Locality...")
loc_age = df.groupby("Locality")["Age_of_Property"].median().reset_index().sort_values("Age_of_Property", ascending=False).head(20)
fig8 = px.bar(
    loc_age, x="Locality", y="Age_of_Property",
    title="Chart 8 — Median Age of Properties by Locality (Top 20 Oldest)",
    labels={"Age_of_Property": "Median Age (Years)", "Locality": "Locality"},
    color="Age_of_Property", color_continuous_scale="Reds",
    template=TEMPLATE
)
fig8.update_layout(xaxis_tickangle=-45)
fig8.show()

# Chart 9 — BHK Distribution Across Cities (Top 10)
print("Chart 9: BHK Distribution by City...")
top10_cities = df["City"].value_counts().head(10).index
df_top10 = df[df["City"].isin(top10_cities)]
bhk_city = df_top10.groupby(["City", "BHK"]).size().reset_index(name="Count")
fig9 = px.bar(
    bhk_city, x="City", y="Count", color="BHK",
    title="Chart 9 — BHK Distribution Across Top 10 Cities",
    labels={"Count": "Number of Properties", "City": "City"},
    barmode="stack", template=TEMPLATE,
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig9.update_layout(xaxis_tickangle=-30)
fig9.show()

# Chart 10 — Price Trends for Top 5 Most Expensive Localities
print("Chart 10: Price Trends - Top 5 Expensive Localities...")
top5_loc = df.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5).index
df_top5 = df[df["Locality"].isin(top5_loc)]
loc_yr = df_top5.groupby(["Locality", "Year_Built"])["Price_in_Lakhs"].mean().reset_index()
fig10 = px.line(
    loc_yr, x="Year_Built", y="Price_in_Lakhs", color="Locality",
    title="Chart 10 — Price Trends for Top 5 Most Expensive Localities",
    labels={"Price_in_Lakhs": "Avg Price (₹ Lakhs)", "Year_Built": "Year Built"},
    template=TEMPLATE, markers=True
)
fig10.show()

# ══════════════════════════════════════════════════════════
# GROUP 3 — CHARTS 11–15: Feature Relationships & Correlation
# ══════════════════════════════════════════════════════════

# Chart 11 — Correlation Heatmap
print("Chart 11: Correlation Heatmap...")
num_cols = ["Price_in_Lakhs", "Price_per_SqFt", "Size_in_SqFt", "BHK",
            "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
            "Floor_No", "Total_Floors", "Parking_Space"]
num_cols = [c for c in num_cols if c in df.columns]
# Fix: force numeric, skip any Yes/No string columns
_safe = []
for _c in num_cols:
    try:
        _conv = df[_c].astype(str).str.extract(r'([\d\.\-]+)')[0].astype(float)
        if _conv.notna().mean() > 0.3:
            df[_c] = _conv
            _safe.append(_c)
    except:
        pass
num_cols = _safe
corr = df[num_cols].corr().round(2)
fig11 = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.columns.tolist(),
    colorscale="RdBu_r",
    zmin=-1, zmax=1,
    text=corr.values.round(2),
    texttemplate="%{text}",
    textfont={"size": 11}
))
fig11.update_layout(
    title="Chart 11 — Correlation Heatmap of Numeric Features",
    template=TEMPLATE, width=800, height=700
)
fig11.show()

# Chart 12 — Nearby Schools vs Price per SqFt
print("Chart 12: Nearby Schools vs Price/SqFt...")
schools_avg = df.groupby("Nearby_Schools")["Price_per_SqFt"].mean().reset_index()
fig12 = px.bar(
    schools_avg, x="Nearby_Schools", y="Price_per_SqFt",
    title="Chart 12 — Nearby Schools vs Average Price per SqFt",
    labels={"Nearby_Schools": "Number of Nearby Schools", "Price_per_SqFt": "Avg Price/SqFt (₹)"},
    color="Price_per_SqFt", color_continuous_scale="Greens",
    template=TEMPLATE, text_auto=".0f"
)
fig12.update_traces(textposition="outside")
fig12.show()

# Chart 13 — Nearby Hospitals vs Price per SqFt
print("Chart 13: Nearby Hospitals vs Price/SqFt...")
hosp_avg = df.groupby("Nearby_Hospitals")["Price_per_SqFt"].mean().reset_index()
fig13 = px.bar(
    hosp_avg, x="Nearby_Hospitals", y="Price_per_SqFt",
    title="Chart 13 — Nearby Hospitals vs Average Price per SqFt",
    labels={"Nearby_Hospitals": "Number of Nearby Hospitals", "Price_per_SqFt": "Avg Price/SqFt (₹)"},
    color="Price_per_SqFt", color_continuous_scale="Purples",
    template=TEMPLATE, text_auto=".0f"
)
fig13.update_traces(textposition="outside")
fig13.show()

# Chart 14 — Price by Furnished Status
print("Chart 14: Price by Furnished Status...")
fig14 = px.violin(
    df, x="Furnished_Status", y="Price_in_Lakhs",
    title="Chart 14 — Property Price by Furnished Status",
    labels={"Price_in_Lakhs": "Price (₹ Lakhs)", "Furnished_Status": "Furnished Status"},
    color="Furnished_Status", box=True, points="outliers",
    template=TEMPLATE,
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig14.update_layout(showlegend=False)
fig14.show()

# Chart 15 — Price per SqFt by Facing Direction
print("Chart 15: Price/SqFt by Facing Direction...")
if "Facing" in df.columns:
    facing_avg = df.groupby("Facing")["Price_per_SqFt"].mean().reset_index().sort_values("Price_per_SqFt", ascending=False)
    fig15 = px.bar(
        facing_avg, x="Facing", y="Price_per_SqFt",
        title="Chart 15 — Price per SqFt by Property Facing Direction",
        labels={"Price_per_SqFt": "Avg Price/SqFt (₹)", "Facing": "Facing Direction"},
        color="Price_per_SqFt", color_continuous_scale="Tealgrn",
        template=TEMPLATE, text_auto=".0f"
    )
    fig15.update_traces(textposition="outside")
    fig15.show()
else:
    # Fallback if Facing column missing — use Property_Type instead
    pt_avg = df.groupby("Property_Type")["Price_per_SqFt"].mean().reset_index().sort_values("Price_per_SqFt", ascending=False)
    fig15 = px.bar(
        pt_avg, x="Property_Type", y="Price_per_SqFt",
        title="Chart 15 — Price per SqFt by Property Type",
        color="Price_per_SqFt", color_continuous_scale="Tealgrn",
        template=TEMPLATE, text_auto=".0f"
    )
    fig15.show()

# ══════════════════════════════════════════════════════════
# GROUP 4 — CHARTS 16–20: Investment, Amenities & Ownership
# ══════════════════════════════════════════════════════════

# Chart 16 — Owner Type Distribution
print("Chart 16: Owner Type Distribution...")
owner_counts = df["Owner_Type"].value_counts().reset_index()
owner_counts.columns = ["Owner_Type", "Count"]
fig16 = px.pie(
    owner_counts, names="Owner_Type", values="Count",
    title="Chart 16 — Properties by Owner Type",
    template=TEMPLATE,
    color_discrete_sequence=px.colors.qualitative.Set3,
    hole=0.35
)
fig16.update_traces(textinfo="percent+label", pull=[0.05]*len(owner_counts))
fig16.show()

# Chart 17 — Availability Status Distribution
print("Chart 17: Availability Status...")
avail_counts = df["Availability_Status"].value_counts().reset_index()
avail_counts.columns = ["Availability_Status", "Count"]
fig17 = px.bar(
    avail_counts, x="Availability_Status", y="Count",
    title="Chart 17 — Properties by Availability Status",
    labels={"Count": "Number of Properties", "Availability_Status": "Status"},
    color="Count", color_continuous_scale="Viridis",
    template=TEMPLATE, text_auto=True
)
fig17.update_traces(textposition="outside")
fig17.show()

# Chart 18 — Parking Space vs Property Price
print("Chart 18: Parking Space vs Price...")
if "Parking_Space" in df.columns:
    park_avg = df.groupby("Parking_Space")["Price_in_Lakhs"].mean().reset_index()
    fig18 = px.bar(
        park_avg, x="Parking_Space", y="Price_in_Lakhs",
        title="Chart 18 — Parking Spaces vs Average Property Price",
        labels={"Parking_Space": "Number of Parking Spaces", "Price_in_Lakhs": "Avg Price (₹ Lakhs)"},
        color="Price_in_Lakhs", color_continuous_scale="Blues",
        template=TEMPLATE, text_auto=".1f"
    )
    fig18.update_traces(textposition="outside")
    fig18.show()

# Chart 19 — Amenities vs Price per SqFt
print("Chart 19: Amenities vs Price/SqFt...")
if "Amenities" in df.columns:
    amenity_avg = df.groupby("Amenities")["Price_per_SqFt"].mean().reset_index().sort_values("Price_per_SqFt", ascending=False)
    fig19 = px.bar(
        amenity_avg, x="Amenities", y="Price_per_SqFt",
        title="Chart 19 — Amenities vs Average Price per SqFt",
        labels={"Price_per_SqFt": "Avg Price/SqFt (₹)", "Amenities": "Amenity Type"},
        color="Price_per_SqFt", color_continuous_scale="Reds",
        template=TEMPLATE, text_auto=".0f"
    )
    fig19.update_layout(xaxis_tickangle=-20)
    fig19.update_traces(textposition="outside")
    fig19.show()

# Chart 20 — Public Transport Accessibility vs Price & Investment
print("Chart 20: Transport Accessibility vs Price & Investment...")
if "Public_Transport_Accessibility" in df.columns:
    transport_stats = df.groupby("Public_Transport_Accessibility").agg(
        Avg_Price=("Price_in_Lakhs", "mean"),
        Good_Investment_Pct=("Good_Investment", "mean")
    ).reset_index()
    transport_stats["Good_Investment_Pct"] *= 100

    fig20 = make_subplots(specs=[[{"secondary_y": True}]])
    fig20.add_trace(
        go.Bar(x=transport_stats["Public_Transport_Accessibility"],
               y=transport_stats["Avg_Price"],
               name="Avg Price (₹ Lakhs)",
               marker_color="#4C72B0"),
        secondary_y=False
    )
    fig20.add_trace(
        go.Scatter(x=transport_stats["Public_Transport_Accessibility"],
                   y=transport_stats["Good_Investment_Pct"],
                   mode="lines+markers",
                   name="Good Investment %",
                   marker=dict(size=10, color="red"),
                   line=dict(color="red", width=3)),
        secondary_y=True
    )
    fig20.update_layout(
        title="Chart 20 — Transport Accessibility vs Price & Investment Quality",
        template=TEMPLATE
    )
    fig20.update_yaxes(title_text="Avg Price (₹ Lakhs)", secondary_y=False)
    fig20.update_yaxes(title_text="Good Investment %", secondary_y=True)
    fig20.show()

print("\n" + "="*60)
print("✅ ALL 20 INTERACTIVE CHARTS OPENED IN YOUR BROWSER!")
print("="*60)
print("\nTips:")
print("  • Hover over any bar/point to see exact values")
print("  • Click legend items to show/hide groups")
print("  • Use mouse scroll to zoom in/out")
print("  • Drag to pan across the chart")
print("  • Double-click to reset the view")
print("  • Click the camera icon (top-right) to save as PNG")
print("\n➡️  Next: Step 3 — ML Models (Classification + Regression)")