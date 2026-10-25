"""
George Market Forecast Lab - IFS742 Group Assignment 2 (five-model comparison)
Reads saved predictions; it does not retrain models or forecast beyond August 2026.

Run locally:   streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = ["Random Forest", "XGBoost", "SVR", "LSTM", "ANN"]
BASES = ["Last-month baseline", "Same-month-last-year baseline"]
ALL = MODELS + BASES
COMMODITIES = ["Potatoes", "Brown onions", "Tomatoes"]
COLORS = {"Random Forest": "#1b5e7a", "XGBoost": "#d97706", "SVR": "#7c3aed", "LSTM": "#059669", "ANN": "#c2410c",
          "Last-month baseline": "#64748b", "Same-month-last-year baseline": "#a3adbb", "Actual": "#111827"}
PERIOD_LABEL = {"test": "Test · May–Aug 2026", "validation": "Validation · Jan–Apr 2026"}
PRICE_COL = {"Potatoes": "potatoes_r_per_kg", "Brown onions": "brown_onions_r_per_kg", "Tomatoes": "tomatoes_r_per_kg"}
METRIC_INFO = {
    "MAE": ("MAE (R/kg)", "Average size of the miss in rand per kg. Our main ranking metric."),
    "RMSE": ("RMSE (R/kg)", "Like MAE but large misses count more, so it shows which model avoids big surprises."),
    "MAPE": ("MAPE (%)", "Average miss as a % of the price, so commodities with different price levels can be compared."),
    "Bias": ("Bias (R/kg)", "Average of forecast minus actual. Negative = the model under-forecasts on average."),
    "vs_naive": ("MAE vs last-month baseline (%)", "How much lower the MAE is than just repeating last month's price. Positive = better."),
}
PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

st.set_page_config(page_title="George Market Forecast Lab", layout="wide")


@st.cache_data
def load():
    p = pd.read_csv(os.path.join(DATA, "predictions.csv"), dtype={"target_month": str})
    p["date"] = pd.to_datetime(p.target_month)
    prices = pd.read_csv(os.path.join(DATA, "monthly_prices.csv"), dtype={"month": str})
    prices["date"] = pd.to_datetime(prices.month)
    return p, prices


@st.cache_data
def metrics(df):
    rows = []
    for (proto, c, part, m), g in df.groupby(["protocol", "commodity", "partition", "model"]):
        e = g.prediction - g.actual
        rows.append({"protocol": proto, "commodity": c, "partition": part, "model": m, "n": len(g),
                     "MAE": e.abs().mean(), "RMSE": float(np.sqrt((e ** 2).mean())),
                     "MAPE": float((e.abs() / g.actual).mean() * 100), "Bias": e.mean()})
    t = pd.DataFrame(rows)
    base = t[t.model == "Last-month baseline"].set_index(["protocol", "commodity", "partition"])["MAE"]
    t["vs_naive"] = [(1 - r.MAE / base.get((r.protocol, r.commodity, r.partition), np.nan)) * 100 for r in t.itertuples()]
    return t


def style(fig, height=420, legend=True, top=False):
    pos = dict(yanchor="bottom", y=1.04) if top else dict(yanchor="top", y=-0.15)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=60 if top else 40, b=10),
                      font=dict(size=13), hoverlabel=dict(font_size=13),
                      legend=dict(orientation="h", xanchor="left", x=0, title=None, **pos) if legend else None,
                      showlegend=legend)
    return fig


def short(m):
    return {"Last-month baseline": "Last month (naive)", "Same-month-last-year baseline": "Same month last year"}.get(m, m)


preds, prices = load()
M = metrics(preds)
H = M[M.protocol == "harmonised"]  # the written justifications always quote the harmonised numbers


def hv(c, m, k="MAE", part="test"):
    return float(H[(H.commodity == c) & (H.model == m) & (H.partition == part)][k].iloc[0])


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Explore")
proto = st.sidebar.radio("Results", ["Harmonised (common protocol)", "As originally submitted"], index=0,
                         help="Harmonised: every model tuned on validation MAE and refitted on training + validation "
                              "before the test months, using 25 months / 14 training targets.")
proto_key = "harmonised" if proto.startswith("Harmonised") else "as submitted"
period = st.sidebar.radio("Period", ["test", "validation"], format_func=PERIOD_LABEL.get)
commodity = st.sidebar.selectbox("Commodity", COMMODITIES)
shown = st.sidebar.multiselect("Models on charts", ALL, default=MODELS + ["Last-month baseline"], format_func=short)
st.sidebar.caption("Tip: click a legend entry to hide a line, double-click to show only that line. Drag on a chart to "
                   "zoom; double-click to reset.")
st.sidebar.caption("George Fresh Produce Market · monthly sales-weighted price (R/kg) · 3-month lookback · "
                   "train Nov 2024–Dec 2025 · validation Jan–Apr 2026 · test May–Aug 2026.")
if not shown:
    st.sidebar.warning("Pick at least one model.")
    shown = ["Random Forest"]

T = M[M.protocol == proto_key]

# ------------------------------------------------------------------ header
st.title("George Market · Forecast Lab")
st.caption("IFS742 Group Assignment 2 · Random Forest, XGBoost, SVR, LSTM and ANN compared on the same months. "
           "Saved results only: this app does not retrain models or forecast new months.")
if proto_key == "as submitted":
    st.info("You are viewing the results **as originally submitted**. These were not trained the same way "
            "(XGBoost, SVR and ANN were fitted on training months only and the ANN used a 24-month window), so they are "
            "shown for transparency. Our decision is based on the harmonised results.")

tab_dec, tab_com, tab_val, tab_models, tab_method = st.tabs(
    ["① Decision", "② Explore a commodity", "③ Validation vs test", "④ Why each model?", "⑤ Method & data"])

# ================================================================== ① decision
with tab_dec:
    tt = T[T.partition == "test"]
    st.subheader("Which model do we recommend, and why?")
    st.markdown(
        f"We recommend **Random Forest** as our preferred *experimental* model. Under the common protocol it had the "
        f"lowest test MAE for **potatoes** (R{hv('Potatoes','Random Forest'):.3f}/kg) and **brown onions** "
        f"(R{hv('Brown onions','Random Forest'):.3f}/kg), and it came second for **tomatoes** "
        f"(R{hv('Tomatoes','Random Forest'):.3f}/kg against LSTM's R{hv('Tomatoes','LSTM'):.3f}/kg). No other model was "
        f"first for more than one commodity, so picking Random Forest means one algorithm to build, explain and maintain "
        f"instead of a different model for every product.\n\n"
        f"We are not claiming it is ready for real use. It failed our bias rule for onions and tomatoes, it improved on the "
        f"last-month baseline by only {hv('Brown onions','Random Forest','vs_naive'):.1f}% for onions, and for onions it was "
        f"actually beaten by the simplest seasonal rule (same month last year, R{hv('Brown onions','Same-month-last-year baseline'):.3f}/kg). "
        f"Each score is also based on just four months.")

    st.markdown("##### Test the decision yourself")
    st.caption("Change our success criteria and the scorecard below updates. This shows how sensitive the choice is to the thresholds we set.")
    s1, s2 = st.columns(2)
    imp_thr = s1.slider("Must beat the last-month baseline MAE by at least (%)", 0, 30, 10, 1)
    bias_thr = s2.slider("Average bias must be within ± (R/kg)", 0.1, 5.0, 0.5, 0.1)
    crit = []
    for m in MODELS:
        wins = imp = bias = seas = 0
        for c in COMMODITIES:
            tc_ = tt[tt.commodity == c].set_index("model")
            wins += tc_.loc[MODELS, "MAE"].idxmin() == m
            imp += tc_.loc[m, "vs_naive"] >= imp_thr
            bias += abs(tc_.loc[m, "Bias"]) <= bias_thr
            seas += tc_.loc[m, "MAE"] < tc_.loc["Same-month-last-year baseline", "MAE"]
        crit.append({"Model": m, "Lowest test MAE": int(wins), f"≥{imp_thr}% better than naive": int(imp),
                     f"Bias within ±R{bias_thr:.1f}": int(bias), "Beats same month last year": int(seas)})
    crit = pd.DataFrame(crit)
    crit["Criteria met (of 12)"] = crit.iloc[:, 1:].sum(axis=1)
    crit = crit.sort_values(["Lowest test MAE", "Criteria met (of 12)"], ascending=False)
    best = crit.iloc[0]
    k = st.columns(4)
    k[0].metric("Top of the scorecard", best.Model)
    k[1].metric("Lowest test MAE", f"{best['Lowest test MAE']} of 3")
    k[2].metric(f"≥{imp_thr}% better than naive", f"{best[f'≥{imp_thr}% better than naive']} / 3")
    k[3].metric(f"Bias within ±R{bias_thr:.1f}", f"{best[f'Bias within ±R{bias_thr:.1f}']} / 3")
    fig = go.Figure()
    for col, colr in zip(crit.columns[1:5], ["#1b5e7a", "#4f9bb8", "#9cc7d8", "#d6e7ee"]):
        fig.add_bar(y=crit.Model, x=crit[col], name=col, orientation="h", marker_color=colr,
                    hovertemplate="%{y}: %{x} of 3 commodities<extra>" + col + "</extra>")
    fig.update_layout(barmode="stack", xaxis_title="Criteria met (3 commodities × 4 checks = 12)",
                      yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 12], dtick=1))
    st.plotly_chart(style(fig, 330), use_container_width=True, config=PLOT_CFG)
    st.caption("Each bar segment counts how many commodities (0–3) a model passes that check for. With the default "
               "thresholds Random Forest scores highest. Even if you relax the bias rule a lot, it stays on top, because it "
               "is the only model with the lowest MAE for two commodities.")

    st.markdown("##### All five models side by side")
    mcol, _ = st.columns([2, 3])
    metric = mcol.radio("Metric", list(METRIC_INFO), horizontal=True, format_func=lambda x: METRIC_INFO[x][0].split(" (")[0]
                        if x != "vs_naive" else "vs naive", key="dec_metric")
    st.caption(METRIC_INFO[metric][1])
    d = tt[tt.model.isin(shown)].copy()
    d["label"] = d.model.map(short)
    fig = px.bar(d, x="commodity", y=metric, color="model", barmode="group", category_orders={"commodity": COMMODITIES, "model": ALL},
                 color_discrete_map=COLORS, pattern_shape="model",
                 pattern_shape_map={m: ("/" if m in BASES else "") for m in ALL},
                 custom_data=["label", "MAE", "RMSE", "MAPE", "Bias", "vs_naive"])
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b> · %{x}<br>MAE R%{customdata[1]:.3f} · RMSE R%{customdata[2]:.3f}"
                                    "<br>MAPE %{customdata[3]:.1f}% · Bias %{customdata[4]:+.3f}<br>vs naive %{customdata[5]:+.1f}%<extra></extra>")
    fig.for_each_trace(lambda tr: tr.update(name=short(tr.name)))
    fig.update_layout(yaxis_title=METRIC_INFO[metric][0], xaxis_title=None)
    if metric in ("Bias", "vs_naive"):
        fig.add_hline(y=0, line_color="#111827", line_width=1)
    if metric == "vs_naive":
        fig.add_hline(y=imp_thr, line_dash="dot", line_color="#059669", annotation_text=f"{imp_thr}% target", annotation_position="top left")
    if metric == "Bias":
        for yv in (bias_thr, -bias_thr):
            fig.add_hline(y=yv, line_dash="dot", line_color="#059669")
    st.plotly_chart(style(fig, 440), use_container_width=True, config=PLOT_CFG)

    st.markdown("##### Improvement over just repeating last month's price")
    hm = tt[tt.model.isin(ALL) & (tt.model != "Last-month baseline")].pivot_table(index="model", columns="commodity", values="vs_naive")
    hm = hm.reindex(index=[m for m in ALL if m != "Last-month baseline"], columns=COMMODITIES)
    lim = 50  # colour scale capped at ±50% so the moderate differences stay visible
    fig = go.Figure(go.Heatmap(z=hm.values, x=COMMODITIES, y=[short(m) for m in hm.index], zmid=0, zmin=-lim, zmax=lim,
                               colorscale=[[0, "#b91c1c"], [0.5, "#f8fafc"], [1, "#047857"]],
                               text=[[f"{v:+.1f}%" for v in r] for r in hm.values], texttemplate="%{text}",
                               hovertemplate="%{y} · %{x}<br>%{text} vs last-month baseline<extra></extra>",
                               colorbar=dict(title="%")))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(style(fig, 360, legend=False), use_container_width=True, config=PLOT_CFG)
    st.caption("Green = lower MAE than the last-month baseline, red = worse (colours capped at ±50%). Onions are red or pale for almost every model: the June–August "
               "price jump was not in any model's inputs, so nobody really beat the simple rule there.")

# ================================================================== ② commodity explorer
with tab_com:
    st.subheader(f"{commodity} · {PERIOD_LABEL[period]}")
    tc = T[(T.commodity == commodity) & (T.partition == period)].set_index("model").reindex(ALL).reset_index()
    win = tc[tc.model.isin(MODELS)].sort_values("MAE").iloc[0]
    k = st.columns(4)
    k[0].metric("Lowest-MAE model", win.model)
    k[1].metric("MAE", f"R{win.MAE:.3f}/kg")
    k[2].metric("RMSE", f"R{win.RMSE:.3f}/kg")
    k[3].metric("vs last-month baseline", f"{win.vs_naive:+.1f}%")

    pc = preds[(preds.protocol == proto_key) & (preds.commodity == commodity)]
    hist = prices[["date", PRICE_COL[commodity]]].dropna().rename(columns={PRICE_COL[commodity]: "price"})
    fig = go.Figure()
    for x0, x1, lab, colr in [("2024-10-15", "2025-12-15", "Training", "#f1f5f9"), ("2025-12-15", "2026-04-15", "Validation", "#fef3c7"),
                              ("2026-04-15", "2026-08-15", "Test", "#dcfce7")]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=colr, opacity=0.6, line_width=0, layer="below",
                      annotation_text=lab, annotation_position="top left", annotation_font_color="#475569")
    fig.add_scatter(x=hist.date, y=hist.price, name="Actual", mode="lines+markers", line=dict(color=COLORS["Actual"], width=3),
                    marker=dict(size=7), hovertemplate="Actual: R%{y:.2f}/kg<extra></extra>")
    for m in [m for m in ALL if m in shown]:
        g = pc[pc.model == m].sort_values("date")
        fig.add_scatter(x=g.date, y=g.prediction, name=short(m), mode="lines+markers",
                        line=dict(color=COLORS[m], width=2, dash="dash" if m in BASES else "solid"), marker=dict(size=6),
                        customdata=np.c_[g.actual, g.prediction - g.actual],
                        hovertemplate=short(m) + ": R%{y:.2f} (miss %{customdata[1]:+.2f})<extra></extra>")
    fig.update_layout(hovermode="x unified", yaxis_title="R/kg", xaxis=dict(rangeslider=dict(visible=True, thickness=0.06),
                      range=["2025-09-01", "2026-08-31"], tickformat="%b %Y"))
    st.markdown("**Actual versus predicted price** · the full price history is shown; predictions exist for the validation and test months")
    st.plotly_chart(style(fig, 520, top=True), use_container_width=True, config=PLOT_CFG)
    st.caption("Drag the slider under the chart to look back at the training months. Hover anywhere to compare every model for that month.")

    c1, c2 = st.columns([3, 2])
    with c1:
        emode = st.radio("Error view", ["Signed error (forecast − actual)", "Absolute error"], horizontal=True)
        pe = pc[(pc.partition == period) & pc.model.isin(shown)].copy()
        pe["err"] = pe.prediction - pe.actual
        if emode.startswith("Absolute"):
            pe["err"] = pe.err.abs()
        pe["Month"] = pe.date.dt.strftime("%b %Y")
        pe["label"] = pe.model.map(short)
        fig = px.bar(pe.sort_values("date"), x="Month", y="err", color="model", barmode="group", color_discrete_map=COLORS,
                     category_orders={"model": ALL}, custom_data=["label", "actual", "prediction"])
        fig.update_traces(hovertemplate="<b>%{customdata[0]}</b> · %{x}<br>Actual R%{customdata[1]:.2f} · "
                                        "Forecast R%{customdata[2]:.2f}<br>Error %{y:+.2f}<extra></extra>")
        fig.for_each_trace(lambda tr: tr.update(name=short(tr.name)))
        fig.add_hline(y=0, line_color="#111827", line_width=1)
        fig.update_layout(yaxis_title="R/kg", xaxis_title=None)
        st.markdown("**Error in each month**")
        st.plotly_chart(style(fig, 400), use_container_width=True, config=PLOT_CFG)
        st.caption("Bars below zero mean the model under-forecast. One bad month can dominate a four-month score.")
    with c2:
        mm = st.selectbox("Rank by", list(METRIC_INFO)[:3], format_func=lambda x: METRIC_INFO[x][0])
        b = tc[tc.model.isin(shown)].sort_values(mm)
        fig = go.Figure(go.Bar(y=b.model.map(short), x=b[mm], orientation="h", marker_color=[COLORS[m] for m in b.model],
                               text=[f"{v:.2f}" for v in b[mm]], textposition="outside",
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title=METRIC_INFO[mm][0] + " · lower is better",
                          xaxis_range=[0, b[mm].max() * 1.25])
        st.markdown("**Ranking**")
        st.plotly_chart(style(fig, 400, legend=False), use_container_width=True, config=PLOT_CFG)

    st.markdown(f"**What we take from the {commodity.lower()} results**")
    cnote = {
        "Potatoes": f"Random Forest (R{hv('Potatoes','Random Forest'):.3f}) and LSTM (R{hv('Potatoes','LSTM'):.3f}) are almost tied, "
                    f"and both beat the last-month baseline (R{hv('Potatoes','Last-month baseline'):.3f}). A gap of less than one cent per kg "
                    f"over four months is not a real difference, so we treat them as equal here. Random Forest's bias "
                    f"(+R{hv('Potatoes','Random Forest','Bias'):.2f}) is inside our ±R0.50 rule. SVR is the warning sign: best on "
                    f"validation, worst on test (R{hv('Potatoes','SVR'):.3f}), over-forecasting every test month.",
        "Brown onions": f"Random Forest has the lowest model MAE (R{hv('Brown onions','Random Forest'):.3f}), but honestly no model did well: "
                        f"the same-month-last-year rule scored R{hv('Brown onions','Same-month-last-year baseline'):.3f} and the last-month "
                        f"rule had the lowest RMSE. Prices jumped in June–August and every model under-forecast "
                        f"(Random Forest bias {hv('Brown onions','Random Forest','Bias'):+.2f}). We would not use any of these models for onions yet.",
        "Tomatoes": f"LSTM wins (R{hv('Tomatoes','LSTM'):.3f}), with Random Forest (R{hv('Tomatoes','Random Forest'):.3f}) and SVR "
                    f"(R{hv('Tomatoes','SVR'):.3f}) close behind, and all three beat both baselines. Neither LSTM "
                    f"({hv('Tomatoes','LSTM','Bias'):+.2f}) nor Random Forest ({hv('Tomatoes','Random Forest','Bias'):+.2f}) meets the bias rule. "
                    f"Tomatoes are the most volatile series, so a R6/kg average miss is about 40–55% of the price.",
    }
    st.markdown(cnote[commodity] + ("" if proto_key == "harmonised" else " *(Numbers quoted from the harmonised results.)*"))

    with st.expander("All metrics and monthly predictions (tables)"):
        show = tc[tc.model.notna()][["model", "MAE", "RMSE", "MAPE", "Bias", "vs_naive"]].rename(columns={
            "model": "Model", "MAE": "MAE (R/kg)", "RMSE": "RMSE (R/kg)", "MAPE": "MAPE (%)", "Bias": "Bias (R/kg)", "vs_naive": "MAE vs naive (%)"})
        show["Model"] = show.Model.map(short)
        st.dataframe(show.style.format({"MAE (R/kg)": "{:.3f}", "RMSE (R/kg)": "{:.3f}", "MAPE (%)": "{:.1f}", "Bias (R/kg)": "{:+.3f}",
                                        "MAE vs naive (%)": "{:+.1f}"}), hide_index=True, use_container_width=True)
        pp = pc[pc.partition == period]
        wide = pp.pivot_table(index="target_month", columns="model", values="prediction")
        wide.insert(0, "Actual", pp.drop_duplicates("target_month").set_index("target_month")["actual"])
        wide.index = pd.to_datetime(wide.index).strftime("%b %Y")
        wide = wide[["Actual"] + [m for m in ALL if m in wide.columns]].rename(columns=short)
        st.dataframe(wide.style.format("{:.3f}"), use_container_width=True)
        st.download_button("Download these predictions (CSV)", pp.drop(columns="date").to_csv(index=False).encode(),
                           file_name=f"{commodity}_{period}_{proto_key}.csv".replace(" ", "_"))

# ================================================================== ③ validation vs test
with tab_val:
    st.subheader("Did the validation months predict the test winner?")
    st.markdown("We chose each model's settings on January–April 2026 (validation) and only then looked at May–August "
                "(test). If validation were a reliable guide, the model that did best on validation would also do best on test. "
                "It mostly didn't, which is one of the main reasons we call our choice *provisional*.")
    vt = T[T.model.isin(shown)].pivot_table(index=["commodity", "model"], columns="partition", values="MAE").reset_index()
    vt["label"] = vt.model.map(short)
    fig = px.scatter(vt, x="validation", y="test", color="model", facet_col="commodity", color_discrete_map=COLORS,
                     category_orders={"commodity": COMMODITIES, "model": ALL}, symbol="model",
                     symbol_map={m: ("diamond" if m in BASES else "circle") for m in ALL}, custom_data=["label"])
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")),
                      hovertemplate="<b>%{customdata[0]}</b><br>Validation MAE R%{x:.3f}<br>Test MAE R%{y:.3f}<extra></extra>")
    fig.for_each_trace(lambda tr: tr.update(name=short(tr.name)))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(matches=None, title="Validation MAE (R/kg)")
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_yaxes(title="Test MAE (R/kg)", col=1)
    for i, c in enumerate(COMMODITIES, start=1):
        sub = vt[vt.commodity == c]
        hi = float(np.nanmax(sub[["validation", "test"]].values)) * 1.1
        fig.add_shape(type="line", x0=0, y0=0, x1=hi, y1=hi, line=dict(dash="dot", color="#94a3b8"), row=1, col=i)
    st.plotly_chart(style(fig, 460), use_container_width=True, config=PLOT_CFG)
    st.caption("Dotted line = same error on validation and test. Almost every point sits above it because the test months had bigger price "
               "swings, so errors rose for everyone. What matters is whether the *order* held, shown in the next chart.")

    rk = T[T.model.isin(MODELS)].copy()
    rk["rank"] = rk.groupby(["commodity", "partition"]).MAE.rank().astype(int)
    rr = rk.pivot_table(index="model", columns=["commodity", "partition"], values="rank").reindex(MODELS)
    fig = go.Figure()
    for m in MODELS:
        xs, ys = [], []
        for c in COMMODITIES:
            xs += [f"{c}<br>validation", f"{c}<br>test", None]
            ys += [rr.loc[m, (c, "validation")], rr.loc[m, (c, "test")], None]
        fig.add_scatter(x=xs, y=ys, name=m, mode="lines+markers", line=dict(color=COLORS[m], width=3), marker=dict(size=11),
                        hovertemplate=m + " · %{x}: rank %{y}<extra></extra>", connectgaps=False)
    fig.update_layout(yaxis=dict(autorange="reversed", dtick=1, title="Rank among five models (1 = best MAE)"), xaxis_title=None)
    st.markdown("**How each model's rank moved from validation to test**")
    st.plotly_chart(style(fig, 420), use_container_width=True, config=PLOT_CFG)
    st.caption("Steep lines = the validation ranking did not hold. SVR, for example, ranked first on validation for potatoes "
               "and tomatoes. LSTM went from last to second for potatoes.")

    st.markdown(f"**Original versus harmonised test MAE · {commodity}** (change the commodity in the sidebar)")
    st.caption("Our first comparison mixed training rules. Re-running XGBoost, SVR and ANN the same way as Random Forest and LSTM changed some scores a lot, mostly the ANN for potatoes, but not the overall decision.")
    ov = M[(M.partition == "test") & M.model.isin(MODELS)].pivot_table(index=["commodity", "model"], columns="protocol", values="MAE").reset_index()
    ov = ov[ov.commodity == commodity]
    fig = go.Figure()
    for r in ov.itertuples():
        fig.add_scatter(x=[r._3, r.harmonised], y=[r.model, r.model], mode="lines", line=dict(color="#cbd5e1", width=4),
                        showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=ov["as submitted"], y=ov.model, mode="markers", name="As submitted", marker=dict(size=13, color="#cbd5e1",
                    line=dict(width=2, color="#64748b")), hovertemplate="%{y} as submitted: R%{x:.3f}<extra></extra>")
    fig.add_scatter(x=ov.harmonised, y=ov.model, mode="markers", name="Harmonised", marker=dict(size=13, color=[COLORS[m] for m in ov.model]),
                    hovertemplate="%{y} harmonised: R%{x:.3f}<extra></extra>")
    fig.update_layout(xaxis_title=f"{commodity} test MAE (R/kg)", yaxis=dict(categoryorder="array", categoryarray=MODELS[::-1]))
    st.plotly_chart(style(fig, 340), use_container_width=True, config=PLOT_CFG)

# ================================================================== ④ model justifications
with tab_models:
    st.subheader("Why we tried each model, what happened, and our verdict")
    st.caption("Numbers are from the harmonised run (same data, same split, settings picked on validation MAE, refitted on "
               "training + validation). Test = May–Aug 2026.")

    def mini(m):
        d = H[(H.model.isin([m, "Last-month baseline"])) & (H.partition == "test")]
        fig = px.bar(d, x="commodity", y="MAE", color="model", barmode="group", color_discrete_map=COLORS,
                     category_orders={"commodity": COMMODITIES, "model": [m, "Last-month baseline"]})
        fig.for_each_trace(lambda tr: tr.update(name=short(tr.name)))
        fig.update_traces(hovertemplate="%{x}: R%{y:.3f}<extra></extra>")
        fig.update_layout(yaxis_title="Test MAE", xaxis_title=None, legend=dict(y=-0.25))
        return style(fig, 260)

    notes = {
        "Random Forest": (
            "Random Forest averages many decision trees, each built on a random resample of the data (Breiman, 2001). We "
            "expected the averaging to be stable on a tiny dataset like ours (14 training rows), and it can pick up non-linear "
            "patterns that a single tree would overfit. We let it predict the *change* from last month rather than the price "
            "itself, because trees cannot predict outside the range they were trained on, and that choice won on validation.",
            f"Lowest test MAE for potatoes (R{hv('Potatoes','Random Forest'):.3f}) and onions (R{hv('Brown onions','Random Forest'):.3f}); "
            f"second for tomatoes. It beat the last-month baseline by {hv('Potatoes','Random Forest','vs_naive'):.1f}% for potatoes and "
            f"{hv('Tomatoes','Random Forest','vs_naive'):.1f}% for tomatoes, but only {hv('Brown onions','Random Forest','vs_naive'):.1f}% "
            f"for onions. Bias was fine for potatoes and too large for onions ({hv('Brown onions','Random Forest','Bias'):+.2f}) and tomatoes "
            f"({hv('Tomatoes','Random Forest','Bias'):+.2f}).",
            "✅ **Selected** as our preferred experimental model: the most consistent across all three commodities and simple "
            "to explain to market staff. Not yet good enough for onions."),
        "XGBoost": (
            "XGBoost builds trees one after another, each fixing the errors of the last, with regularisation to limit "
            "overfitting (Chen and Guestrin, 2016). It often wins tabular-data competitions, so we wanted to see whether "
            "boosting beats Random Forest's averaging.",
            f"Worst or near-worst everywhere: R{hv('Potatoes','XGBoost'):.3f} / R{hv('Brown onions','XGBoost'):.3f} / "
            f"R{hv('Tomatoes','XGBoost'):.3f}. It over-forecast potatoes every test month (bias {hv('Potatoes','XGBoost','Bias'):+.2f}). "
            "With only 18 rows to fit there is very little for later trees to correct, and like any tree model it predicts "
            "price *levels* inside the training range, so it could not follow new highs or lows.",
            "❌ **Not selected.** Boosting needs more data than we have. It might work better with a longer series or the "
            "change-target trick we used for Random Forest."),
        "SVR": (
            "Support Vector Regression ignores errors smaller than a margin ε and fits a smooth function, using a kernel for "
            "non-linear patterns (Smola and Schölkopf, 2004). It is known to work on small samples, which matched our situation.",
            f"The most interesting result. SVR had the **best validation MAE** for potatoes ({hv('Potatoes','SVR','MAE','validation'):.3f}) "
            f"and tomatoes ({hv('Tomatoes','SVR','MAE','validation'):.3f}), then the **worst test MAE** for potatoes "
            f"(R{hv('Potatoes','SVR'):.3f}, bias {hv('Potatoes','SVR','Bias'):+.2f}). For tomatoes it stayed competitive "
            f"(R{hv('Tomatoes','SVR'):.3f}, third). The chosen RBF settings fitted the validation months very closely and did not "
            "carry over.",
            "❌ **Not selected.** A clear example of why four validation months are not enough to choose settings, and why we "
            "looked at all metrics and all commodities before deciding."),
        "LSTM": (
            "An LSTM is a recurrent neural network that remembers patterns across a sequence (Hochreiter and Schmidhuber, 1997). "
            "Studies on vegetable prices (Zhang et al., 2024; Okere and Balyan, 2025) report strong LSTM results, so it was the "
            "deep-learning model we most expected to do well. Ours used the last 3 months plus the calendar month, averaged over 3 seeds.",
            f"Best for tomatoes (R{hv('Tomatoes','LSTM'):.3f}) and a near-tie with Random Forest for potatoes "
            f"(R{hv('Potatoes','LSTM'):.3f}), even though it ranked last on validation for potatoes. Onions were weak "
            f"(R{hv('Brown onions','LSTM'):.3f}, bias {hv('Brown onions','LSTM','Bias'):+.2f}), and tomato bias was large "
            f"({hv('Tomatoes','LSTM','Bias'):+.2f}).",
            "🟡 **Strong runner-up.** If the group were to use a different model per commodity, LSTM would be the tomato model. "
            "We did not, because it is harder to explain and to retrain, and the published LSTM studies used far longer series than our 25 months."),
        "ANN": (
            "A small feed-forward neural network (MLP) can approximate non-linear relationships between the lag prices and next "
            "month's price, but it is prone to overfitting on short series (Zhang, Patuwo and Hu, 1998). We kept it tiny (2 or 4 "
            "hidden neurons, L2 penalty) for that reason.",
            f"Never first. Under the common protocol (all 25 months) potatoes improved from R1.524 to R{hv('Potatoes','ANN'):.3f}, "
            f"which still did not beat the last-month baseline. Onions R{hv('Brown onions','ANN'):.3f}, tomatoes R{hv('Tomatoes','ANN'):.3f}.",
            "❌ **Not selected.** Consistently mid-table. The big change after harmonisation shows how much a short series "
            "depends on every extra month of data."),
    }
    for m in MODELS:
        with st.container(border=True):
            a, b = st.columns([3, 2])
            why, what, verdict = notes[m]
            with a:
                st.markdown(f"#### <span style='color:{COLORS[m]}'>●</span> {m}", unsafe_allow_html=True)
                st.markdown(f"**Why we tried it.** {why}")
                st.markdown(f"**What happened.** {what}")
                st.markdown(verdict)
            with b:
                st.plotly_chart(mini(m), use_container_width=True, config=PLOT_CFG, key=f"mini_{m}")

    st.markdown("#### Why we did not just pick the best model per commodity")
    st.markdown(
        "Choosing LSTM for tomatoes and Random Forest for the rest would look better on paper, but that choice is made *after* "
        "seeing four test months, so it would overfit the test set. One model is also simpler to maintain and explain. "
        "Makridakis, Spiliotis and Assimakopoulos (2018) also warn that complex machine-learning methods often fail to beat "
        "simple benchmarks, which our onion results confirm. We therefore chose one model and kept both simple baselines in "
        "every comparison.")

# ================================================================== ⑤ method + data
with tab_method:
    st.subheader("How the models were trained (harmonised protocol)")
    st.markdown("""
- **Data:** approved FINAL v3 monthly sales-weighted prices, Aug 2024–Aug 2026 (25 months); the previous 3 months' prices are the inputs.
- **Split:** 14 training targets (Nov 2024–Dec 2025), 4 validation (Jan–Apr 2026), 4 test (May–Aug 2026), identical for all models. Chronological, never shuffled.
- **Selection:** each member's own candidate grid; the setting with the lowest **validation MAE** was chosen (RMSE breaks ties).
- **Final fit:** the chosen setting was refitted on training + validation, then forecast the test months once (seed 42). Scalers are fitted on the fitting data only.
- **Baselines:** *last month* (repeat the previous month's price) and *same month last year* (seasonal naive).
- **Random Forest** predicts the change from last month's price (chosen on validation). **LSTM** uses a 3-seed ensemble with calendar-month sine/cosine inputs, as submitted.
- **XGBoost, SVR and ANN** were re-run under this protocol. Their originally submitted runs (training-only fit, validation-RMSE selection, ANN 24-month window) can be viewed with *As originally submitted* in the sidebar.
""")
    pr = prices.melt(id_vars=["month", "date"], var_name="series", value_name="R/kg").dropna()
    pr = pr[pr.series.isin(PRICE_COL.values())]
    pr["series"] = pr.series.map({v: k for k, v in PRICE_COL.items()})
    fig = px.line(pr, x="date", y="R/kg", color="series", markers=True, category_orders={"series": COMMODITIES},
                  color_discrete_sequence=["#b45309", "#7c2d12", "#dc2626"])
    fig.update_traces(hovertemplate="%{x|%b %Y}: R%{y:.2f}/kg")
    fig.update_layout(hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True, thickness=0.06), tickformat="%b %Y"), xaxis_title=None)
    st.markdown("**Historical monthly prices**")
    st.plotly_chart(style(fig, 400, top=True), use_container_width=True, config=PLOT_CFG)
    st.caption("Potatoes, brown onions and tomatoes have 25 consecutive months; avocados (23) and beetroot (4) were excluded from the common comparison.")
    st.markdown("**Limitations we are aware of:** four test months per commodity, price lags only (no weather or supply volumes), "
                "the test window has already been inspected, and there are no prediction intervals. A longer series and "
                "rolling-origin backtesting would be needed before any real deployment.")

st.caption("Academic prototype. Rankings describe four test months per commodity and are not evidence of dependable future forecasts.")
