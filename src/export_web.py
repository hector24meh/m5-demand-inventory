"""Exporta el resumen del proyecto M5 a JSON para la página del portafolio.
Uso: uv run python src/export_web.py   (después de ejecutar los notebooks 04 a 11)"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
PROC = ROOT / "data" / "processed"
OUT = Path.home() / "Documents" / "portfolio" / "public" / "data" / "m5"
OUT.mkdir(parents=True, exist_ok=True)


def limpiar_columnas(df):
    return df.rename(columns=lambda c: c.replace("%", "pct").lower())


def registros(df, dec=4):
    return json.loads(limpiar_columnas(df).round(dec).to_json(orient="records"))


# --- Inventario ---
inv = pd.read_csv(REP / "simulacion_inventario_volumen_CA_1.csv")


def inventario_en(modelo, fr):
    c = inv[inv["modelo"] == modelo].sort_values("fill_rate")
    if not (c["fill_rate"].min() <= fr <= c["fill_rate"].max()):
        return None
    return float(np.interp(fr, c["fill_rate"], c["inv_valor_prom"]))


ahorro_por_fr = []
for fr in [0.92, 0.95, 0.97, 0.98]:
    i_lgb, i_tsb = inventario_en("LGB", fr), inventario_en("TSB", fr)
    ahorro_por_fr.append({"fill_rate": fr, "inv_lgb": i_lgb, "inv_tsb": i_tsb,
                          "ahorro_pct": round((1 - i_lgb / i_tsb) * 100, 1) if i_lgb and i_tsb else None})

opt = inv.loc[inv.groupby("modelo")["costo_total"].idxmin()].set_index("modelo")
pf = pd.read_csv(REP / "metricas_prueba_final_CA_1.csv").set_index("modelo")
bt = pd.read_csv(REP / "metricas_lgb_vs_baselines_CA_1.csv")
meta = json.loads((OUT / "meta_simulacion.json").read_text())

kpis = {
    "ahorro_inventario_fr95_pct": next(a["ahorro_pct"] for a in ahorro_por_fr if a["fill_rate"] == 0.95),
    "ahorro_costo_vs_tsb_pct": round((1 - opt.loc["LGB", "costo_total"] / opt.loc["TSB", "costo_total"]) * 100, 1),
    "ahorro_costo_vs_sn_pct": round((1 - opt.loc["LGB", "costo_total"] / opt.loc["SeasonalNaive", "costo_total"]) * 100, 1),
    "fill_rate_optimo_lgb": round(float(opt.loc["LGB", "fill_rate"]), 4),
    "fill_rate_optimo_tsb": round(float(opt.loc["TSB", "fill_rate"]), 4),
    "rmsse_prueba_lgb": round(float(pf.loc["LGB", "RMSSE_limpio"]), 3),
    "rmsse_prueba_autoets": round(float(pf.loc["AutoETS", "RMSSE_limpio"]), 3),
    "sesgo_prueba_lgb_pct": round(float(pf.loc["LGB", "Sesgo_%_limpio"]), 1),
    "sesgo_prueba_autoets_pct": round(float(pf.loc["AutoETS", "Sesgo_%_limpio"]), 1),
    "sesgo_prueba_tsb_pct": round(float(pf.loc["TSB", "Sesgo_%_limpio"]), 1),
    **meta,
}

# --- EDA y censura ---
seg = pl.read_parquet(PROC / "segmentacion_CA_1.parquet")
mascara = pl.read_parquet(PROC / "mascara_CA_1.parquet")
eda = {
    "clases_demanda": dict(seg.group_by("sb_class").len().sort("sb_class").iter_rows()),
    "matriz_abc_xyz": [{"abc": a, "xyz": x, "productos": n}
                       for a, x, n in seg.group_by(["abc", "xyz"]).len().sort(["abc", "xyz"]).iter_rows()],
    "censura_pct": {t: round(n / mascara.height * 100, 2)
                    for t, n in mascara.group_by("tipo").len().sort("tipo").iter_rows()},
}

# --- Ablación (06): el efecto de corregir la censura vs. el efecto del modelo ---
abl = pd.read_csv(REP / "ablacion_censura_CA_1.csv")
abl_res = pd.read_csv(REP / "ablacion_resumen_CA_1.csv", index_col=0).rename_axis("modelo").reset_index()
abl_sens = pd.read_csv(REP / "ablacion_sensibilidad_CA_1.csv").rename(columns={
    "Censura (TSB_imp vs. TSB)": "censura_tsb", "Censura (AutoETS_imp vs. AutoETS)": "censura_autoets",
    "Modelo (LGB vs. AutoETS_imp)": "modelo_vs_autoets_imp", "Modelo (LGB vs. TSB_imp)": "modelo_vs_tsb_imp"})


def inv_abl(modelo, fr):
    c = abl[abl["modelo"] == modelo].sort_values("fill_rate")
    if not (c["fill_rate"].min() <= fr <= c["fill_rate"].max()):
        return None
    return float(np.interp(fr, c["fill_rate"], c["inv_valor_prom"]))


ahorro_censura = []
for fr in [0.92, 0.95, 0.97, 0.98]:
    con, sin = inv_abl("TSB_imp", fr), inv_abl("TSB", fr)
    ahorro_censura.append({"fill_rate": fr, "inv_corregida": con, "inv_ventas": sin,
                           "ahorro_pct": round((1 - con / sin) * 100, 1) if con and sin else None})
opt_abl = abl.loc[abl.groupby("modelo")["costo_total"].idxmin()].set_index("modelo")

# --- Detector (07) e incertidumbre (08) ---
det = pd.read_csv(REP / "detector_sensibilidad_CA_1.csv")
fa = pd.read_csv(REP / "detector_falsas_alarmas_CA_1.csv")
ic = pd.read_csv(REP / "bootstrap_ic_CA_1.csv")
ic["clave"] = ic["comparacion"].map({
    "Censura · TSB (demanda corregida vs. ventas)": "censura_tsb",
    "Censura · AutoETS (demanda corregida vs. ventas)": "censura_autoets",
    "Modelo · LightGBM vs. AutoETS corregido": "modelo_vs_autoets_imp",
    "Modelo · LightGBM vs. TSB corregido": "modelo_vs_tsb_imp",
    "Original · LightGBM vs. TSB con ventas": "lgb_vs_tsb"})
ic["metrica"] = ic["metrica"].map({"Costo total óptimo": "costo", "Inventario a fill rate 95%": "inv95"})

kpis.update({
    "censura_ahorro_inventario_fr95_pct": next(a["ahorro_pct"] for a in ahorro_censura if a["fill_rate"] == 0.95),
    "censura_ahorro_costo_pct": round((1 - opt_abl.loc["TSB_imp", "costo_total"] / opt_abl.loc["TSB", "costo_total"]) * 100, 1),
    "detector_falsas_alarmas_pct": round(float(fa["agotado_sintetico_pct"].mean()), 2),
    "detector_precision_pct": round(float((1 - fa["agotado_sintetico_pct"].mean() / fa["agotado_observado_pct"].mean()) * 100), 0),
})

# --- Extensiones: reentrenamiento (09), 10 tiendas (10) y LightGBM cuantílico (11) ---
reent = pd.read_csv(REP / "reentrenamiento_resumen_CA_1.csv", index_col=0).rename_axis("modelo").reset_index()
tiendas = pd.read_csv(REP / "multitienda_resumen.csv")
prob = pd.read_csv(REP / "probabilistico_resumen_CA_1.csv", index_col=0).rename_axis("politica").reset_index()
prob["clave"] = prob["politica"].map({"LGB + stock empírico (actual)": "actual",
                                      "LGB directo 13 d + stock empírico": "directo",
                                      "LGB cuantílico": "cuantilico",
                                      "LGB cuantílico + conformal": "conformal"})
cob = pd.read_csv(REP / "probabilistico_cobertura_CA_1.csv").rename(columns={
    "LGB + stock empírico (actual)": "actual", "LGB directo 13 d + stock empírico": "directo",
    "LGB cuantílico": "cuantilico", "LGB cuantílico + conformal": "conformal"})
ic_prob = pd.read_csv(REP / "probabilistico_bootstrap_CA_1.csv")
ic_prob["clave"] = ic_prob["comparacion"].map({
    "LGB cuantílico vs. AutoETS corregido": "cuantilico_vs_autoets_imp",
    "LGB cuantílico vs. TSB corregido": "cuantilico_vs_tsb_imp",
    "LGB cuantílico vs. LGB actual": "cuantilico_vs_actual",
    "LGB directo vs. LGB actual": "directo_vs_actual",
    "LGB cuantílico vs. LGB directo": "cuantilico_vs_directo",
    "LGB cuantílico vs. TSB con ventas": "cuantilico_vs_tsb"})


def ic_de(clave, metrica):
    r = ic_prob[(ic_prob["clave"] == clave) & (ic_prob["metrica"] == metrica)].iloc[0]
    return {k: round(float(r[k]), 1) for k in ["cambio_pct", "ic95_inf", "ic95_sup"]}


kpis.update({
    "tiendas_censura_costo_prom_pct": round(float(tiendas["censura_costo_pct"].mean()), 1),
    "tiendas_censura_costo_min_pct": round(float(tiendas["censura_costo_pct"].max()), 1),
    "tiendas_censura_costo_max_pct": round(float(tiendas["censura_costo_pct"].min()), 1),
    "tiendas_censura_inv_prom_pct": round(float(tiendas["censura_inv95_pct"].mean()), 1),
    "cuantilico_vs_autoets_imp": ic_de("cuantilico_vs_autoets_imp", "costo"),
    "cuantilico_vs_autoets_imp_inv": ic_de("cuantilico_vs_autoets_imp", "inv95"),
    "cuantilico_vs_tsb": ic_de("cuantilico_vs_tsb", "costo"),
    "reentreno_costo_pct": round(float(reent.set_index("modelo").loc["LGB_reentrenado", "vs_LGB_%_costo"]), 1),
})

resumen = {
    "kpis": kpis,
    "extensiones": {
        "tiendas": registros(tiendas),
        "probabilistico": registros(prob.drop(columns="politica")),
        "cobertura": registros(cob),
        "ic_probabilistico": registros(ic_prob[["clave", "metrica", "cambio_pct", "ic95_inf", "ic95_sup", "prob_ahorro"]]),
        "reentrenamiento": registros(reent),
    },
    "ablacion": {
        "tradeoff": registros(abl[["modelo", "alpha", "fill_rate", "inv_valor_prom", "dias_cobertura", "costo_total"]]),
        "resumen": registros(abl_res),
        "ahorro_censura_por_fill_rate": ahorro_censura,
        "sensibilidad": registros(abl_sens),
    },
    "detector": {"sensibilidad": registros(det)},
    "incertidumbre": registros(ic[["clave", "metrica", "cambio_pct", "ic95_inf", "ic95_sup", "prob_ahorro"]]),
    "metricas_backtest": registros(bt),
    "prueba_final": registros(pf.reset_index()),
    "prueba_final_abc": registros(pd.read_csv(REP / "prueba_final_por_abc_CA_1.csv")
                                  .rename(columns={"Unnamed: 0": "abc"})),
    "tradeoff": registros(inv[["modelo", "alpha", "fill_rate", "inv_valor_prom",
                               "dias_cobertura", "costo_total"]]),
    "ahorro_por_fill_rate": ahorro_por_fr,
    "sensibilidad": registros(pd.read_csv(REP / "sensibilidad_costos_CA_1.csv")),
    "eda": eda,
}

(OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1))
print(f"resumen.json → {OUT} ({(OUT / 'resumen.json').stat().st_size / 1024:.0f} KB)")
print(json.dumps(kpis, indent=2, ensure_ascii=False))