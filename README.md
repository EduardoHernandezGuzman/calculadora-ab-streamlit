# Calculadora CRO 3.0 — A/B Testing

Aplicación interactiva en **Streamlit** para analizar experimentos A/B orientados a **CRO (Conversion Rate Optimization)** con enfoques **bayesiano** y **frecuentista**.

## Motores disponibles

| Motor | Distribución | Session ID |
|---|---|---|
| Bayesiano [0,1] | Beta-Binomial | ❌ / ✅ |
| Bayesiano [0,∞] | Gamma-Poisson | ❌ / ✅ |
| Frecuentista | Bootstrap | ❌ / ✅ |

## Arquitectura

```
app.py                  → UI Streamlit + wizard de configuración
adapter/engine_router.py → enrutador que selecciona el motor según configuración
pablo_code/              → 6 motores de cálculo (implementaciones independientes)
```

Cada motor devuelve una estructura homogénea: `summary`, `figures`, `pdf_bytes`, `log_text`, `comparisons`.

## Funcionalidades

- Wizard guiado paso a paso para configurar el análisis
- Interpretación con IA vía OpenAI (opcional)
- Exportación a PDF (opcional)
- CSS personalizado con identidad visual de VML THE COCKTAIL

## Requisitos

```
streamlit pandas matplotlib seaborn numpy openai pymc
```

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```
