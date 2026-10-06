"""
PoC de variables candidatas
===========================
Objetivo único: saber si alguna variable adicional, que la usuaria SÍ puede declarar,
mejora de forma real la discriminación del modelo (AUC-ROC) frente a las 8 del diccionario v1.

Regla de decisión (fijada ANTES de correr):
  - Una candidata se considera solo si sube el AUC promedio en >= 0.005 (validación cruzada de 5 folds)
    y la mejora se repite en el subgrupo emprendedoras.
  - Aun cumpliendo, entra al diccionario SOLO si encontramos sustento Q1-Q2 (regla del proyecto).

Además mide:
  - Base sin restricciones monotónicas: cuánto AUC "cuesta" anclar el modelo a la lógica de la SBS.

Datos: Home Credit (application_train, bureau, bureau_balance, previous_application), solo mujeres.
Uso:   python poc_variables.py "C:\\ruta\\home-credit-default-risk"
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RUTA = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\alexa\Escritorio\UPC\TP1\home-credit-default-risk"
SEMILLA = 42


def csv(nombre):
    return os.path.join(RUTA, f"dataset-{nombre}.csv")


EDUCACION = {"Lower secondary": 0, "Secondary / secondary special": 1, "Incomplete higher": 2,
             "Higher education": 3, "Academic degree": 3}
t0 = time.time()

# ---------------------------------------------------------------- 1. Datos
app = pd.read_csv(csv("application_train"), usecols=[
    "SK_ID_CURR", "TARGET", "CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT", "DAYS_BIRTH",
    "NAME_EDUCATION_TYPE", "CNT_CHILDREN", "NAME_INCOME_TYPE", "ORGANIZATION_TYPE", "DAYS_EMPLOYED",
    "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "NAME_FAMILY_STATUS", "CNT_FAM_MEMBERS"])
w = app[app.CODE_GENDER == "F"].reset_index(drop=True)

bur = pd.read_csv(csv("bureau"), usecols=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE",
                                          "AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT"])
en_buro = w.SK_ID_CURR.isin(bur.SK_ID_CURR)
act = (bur[bur.CREDIT_ACTIVE == "Active"].assign(d=lambda x: x.AMT_CREDIT_SUM_DEBT.fillna(0))
       .groupby("SK_ID_CURR").agg(n=("SK_ID_BUREAU", "count"), deuda=("d", "sum")))
hist = bur.groupby("SK_ID_CURR").agg(n_total=("SK_ID_BUREAU", "count"), dias_primero=("DAYS_CREDIT", "min"))
w = w.merge(act, left_on="SK_ID_CURR", right_index=True, how="left") \
     .merge(hist, left_on="SK_ID_CURR", right_index=True, how="left")

bb = pd.read_csv(csv("bureau_balance"), usecols=["SK_ID_BUREAU", "STATUS"],
                 dtype={"SK_ID_BUREAU": "int64", "STATUS": "category"})
ids_bb = bb.SK_ID_BUREAU.unique()
ids_atr = bb.loc[bb.STATUS.isin(["1", "2", "3", "4", "5"]), "SK_ID_BUREAU"].unique()
del bb
c_bb = bur.loc[bur.SK_ID_BUREAU.isin(ids_bb), "SK_ID_CURR"].unique()
c_atr = bur.loc[bur.SK_ID_BUREAU.isin(ids_atr), "SK_ID_CURR"].unique()

prev = pd.read_csv(csv("previous_application"), usecols=["SK_ID_CURR", "NAME_CONTRACT_STATUS"])
rechazos = prev[prev.NAME_CONTRACT_STATUS == "Refused"].groupby("SK_ID_CURR").size()
w["n_rechazos"] = w.SK_ID_CURR.map(rechazos).fillna(0)

# ---------------------------------------------------------------- 2. Variables
X = pd.DataFrame(index=w.index)
X["ratio_cuota_ingreso"] = w.AMT_ANNUITY / w.AMT_INCOME_TOTAL
X["ratio_deuda_ingreso"] = (w.AMT_CREDIT + np.where(en_buro, w.deuda.fillna(0), np.nan)) / w.AMT_INCOME_TOTAL
X["ratio_monto_ingreso"] = w.AMT_CREDIT / w.AMT_INCOME_TOTAL
X["n_otros_creditos"] = np.where(en_buro, w.n.fillna(0), np.nan)
X["atraso_historial"] = np.where(w.SK_ID_CURR.isin(c_atr), 1.0, np.where(w.SK_ID_CURR.isin(c_bb), 0.0, np.nan))
X["edad"] = -w.DAYS_BIRTH / 365.25
X["nivel_educativo"] = w.NAME_EDUCATION_TYPE.map(EDUCACION)
X["n_hijos"] = w.CNT_CHILDREN
BASE = list(X.columns)
MONO_BASE = [1, 1, 1, 1, 1, 0, 0, 0]

# Candidatas declarables (pregunta sugerida al lado)
dias_emp = w.DAYS_EMPLOYED.where(w.DAYS_EMPLOYED != 365243)       # 365243 = pensionista/sin empleo en Home Credit
C = pd.DataFrame(index=w.index)
C["anios_trabajo_negocio"] = -dias_emp / 365.25                    # ¿Hace cuántos años tienes tu negocio/trabajo actual?
C["vivienda_propia"] = (w.FLAG_OWN_REALTY == "Y").astype(float)    # ¿La vivienda donde vives es tuya?
C["tipo_vivienda"] = w.NAME_HOUSING_TYPE.astype("category")        # ¿Propia, alquilada, con familiares...?
C["estado_civil"] = w.NAME_FAMILY_STATUS.astype("category")        # Estado civil
C["miembros_hogar"] = w.CNT_FAM_MEMBERS                            # ¿Cuántas personas viven en tu hogar?
C["n_creditos_historicos"] = np.where(en_buro, w.n_total, np.nan)  # ¿Cuántos créditos has tenido en total?
C["anios_historial_credito"] = np.where(en_buro, -w.dias_primero / 365.25, np.nan)  # ¿Hace cuántos años sacaste tu primer crédito?
C["n_rechazos"] = w.n_rechazos                                      # ¿Cuántas veces te rechazaron un crédito?

y = w.TARGET.values
emp = ((w.NAME_INCOME_TYPE == "Commercial associate") | (w.ORGANIZATION_TYPE == "Self-employed")).values


# ---------------------------------------------------------------- 3. Validación cruzada
def auc_cv(Xc, mono):
    todas, empr = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEMILLA).split(Xc, y):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                               monotone_constraints=mono, random_state=SEMILLA, verbose=-1)
        m.fit(Xc.iloc[tr], y[tr])
        p = m.predict_proba(Xc.iloc[te])[:, 1]
        todas.append(roc_auc_score(y[te], p))
        empr.append(roc_auc_score(y[te][emp[te]], p[emp[te]]))
    return {"auc_todas": float(np.mean(todas)), "sd_todas": float(np.std(todas)),
            "auc_emprendedoras": float(np.mean(empr)), "sd_emprendedoras": float(np.std(empr))}


resultados = {"BASE (8 variables, monotónico)": auc_cv(X, MONO_BASE),
              "BASE sin restricciones monotónicas": auc_cv(X, [0] * 8)}
b = resultados["BASE (8 variables, monotónico)"]
for c in C.columns:
    Xc = pd.concat([X, C[[c]]], axis=1)
    resultados[f"+ {c}"] = auc_cv(Xc, MONO_BASE + [0])
Xall = pd.concat([X, C], axis=1)
resultados["+ TODAS las candidatas"] = auc_cv(Xall, MONO_BASE + [0] * C.shape[1])

for r in resultados.values():
    r["mejora_todas"] = r["auc_todas"] - b["auc_todas"]
    r["mejora_emprendedoras"] = r["auc_emprendedoras"] - b["auc_emprendedoras"]
    r["pasa_regla"] = bool(r["mejora_todas"] >= 0.005 and r["mejora_emprendedoras"] >= 0.005)

# ---------------------------------------------------------------- 4. Reporte
print(f"\nMujeres: {len(X):,} | emprendedoras: {emp.sum():,} | TARGET=1: {y.mean()*100:.2f} % | {time.time()-t0:.0f} s")
print(f"\n{'Configuración':38s} {'AUC todas':>10s} {'mejora':>8s} {'AUC empr.':>10s} {'mejora':>8s}  regla")
for nombre, r in resultados.items():
    print(f"{nombre:38s} {r['auc_todas']:10.4f} {r['mejora_todas']:+8.4f} {r['auc_emprendedoras']:10.4f} "
          f"{r['mejora_emprendedoras']:+8.4f}  {'PASA' if r['pasa_regla'] else '-'}")
faltantes = {c: round(float(C[c].isna().mean() * 100), 2) for c in C.columns}
print("\n% de faltantes en las candidatas:", faltantes)
with open("resultado_poc_variables.json", "w", encoding="utf-8") as fh:
    json.dump({"resultados": resultados, "faltantes_candidatas_pct": faltantes,
               "regla": "mejora >= 0.005 en todas y en emprendedoras (CV 5 folds)"}, fh, indent=2, ensure_ascii=False)
print("\nGuardado: resultado_poc_variables.json")
