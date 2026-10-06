# PoC 08: variables candidatas y costo de las restricciones monotónicas

Parte del proyecto de tesis "Plataforma web para la detección temprana del riesgo de sobreendeudamiento en mujeres emprendedoras" (UPC, Taller de Proyecto I, 2026-20).

## Objetivo único
Saber si alguna variable adicional que la usuaria puede declarar mejora de forma real la discriminación del modelo frente a las 8 variables base. También mide cuánto AUC-ROC cuesta anclar el modelo a la lógica de la SBS mediante restricciones monotónicas.

## Datos
Home Credit Default Risk (Kaggle, 2018): `application_train`, `bureau` y `bureau_balance`. Los CSV no se incluyen en el repositorio porque las reglas de la competencia no permiten redistribuirlos. Se descargan de https://www.kaggle.com/competitions/home-credit-default-risk/data.

Población: mujeres (`CODE_GENDER == "F"`). El subgrupo emprendedoras son las clientas con `NAME_INCOME_TYPE == "Commercial associate"` u `ORGANIZATION_TYPE == "Self-employed"`, y se reporta por separado.

También se usa `previous_application` para las candidatas que dependen del historial de solicitudes.

## Método
Validación cruzada estratificada de 5 particiones con semilla 42, y LGBMClassifier con:
- 600 árboles;
- tasa de aprendizaje de 0.03;
- 31 hojas;
- 200 registros mínimos por hoja;
- muestreo de 0.8 de las filas y 0.9 de las variables.

**Regla, fijada antes de correr:** una candidata se considera solo si sube el AUC-ROC promedio en al menos 0.005 tanto en todas las mujeres como en las emprendedoras. Aun así, entra solo si tiene sustento en un estudio Q1-Q2.

## Cómo correrla
```
pip install -r requirements.txt
python poc_variables.py "C:\ruta\home-credit-default-risk"
```

## Resultado
| Configuración | AUC-ROC todas | DE | AUC-ROC emprendedoras | DE | Mejora todas | Mejora emprendedoras | Pasa la regla |
|---|---|---|---|---|---|---|---|
| BASE (8 variables, monotónico) | 0.6422 | 0.0038 | 0.6291 | 0.0038 | +0.0000 | +0.0000 | No |
| BASE sin restricciones monotónicas | 0.6562 | 0.0043 | 0.6471 | 0.0031 | +0.0140 | +0.0180 | Sí |
| + anios_trabajo_negocio | 0.6529 | 0.0022 | 0.6409 | 0.0070 | +0.0107 | +0.0118 | Sí |
| + vivienda_propia | 0.6421 | 0.0030 | 0.6302 | 0.0040 | -0.0002 | +0.0011 | No |
| + tipo_vivienda | 0.6417 | 0.0034 | 0.6286 | 0.0038 | -0.0006 | -0.0005 | No |
| + estado_civil | 0.6442 | 0.0034 | 0.6314 | 0.0038 | +0.0020 | +0.0023 | No |
| + miembros_hogar | 0.6431 | 0.0034 | 0.6301 | 0.0033 | +0.0009 | +0.0010 | No |
| + n_creditos_historicos | 0.6505 | 0.0027 | 0.6372 | 0.0025 | +0.0083 | +0.0081 | Sí |
| + anios_historial_credito | 0.6551 | 0.0034 | 0.6404 | 0.0022 | +0.0129 | +0.0113 | Sí |
| + n_rechazos | 0.6532 | 0.0038 | 0.6391 | 0.0062 | +0.0109 | +0.0100 | Sí |
| + TODAS las candidatas | 0.6776 | 0.0027 | 0.6627 | 0.0031 | +0.0353 | +0.0336 | Sí |

Mejora = diferencia de AUC-ROC promedio frente a la base de 8 variables con restricciones monotónicas. DE = desviación estándar entre las 5 particiones.

## Decisión de diseño
- Entran los años de trabajo o negocio.
- Las variables de vivienda, estado civil y personas en el hogar no aportan.
- Los años de historial crediticio, los créditos históricos y los rechazos previos se descartan por falta de sustento o porque la usuaria no puede declararlos de forma confiable.
- Las restricciones monotónicas se mantienen como decisión de diseño.

## Archivos
- `poc_variables.py`: script.
- `resultado_poc_variables.json`: salida estructurada.
- `salida_variables.txt`: salida completa de la consola.
