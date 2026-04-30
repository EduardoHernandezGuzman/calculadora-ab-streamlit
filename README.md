# Calculadora A/B (CRO)

Aplicación interactiva en Streamlit para analizar experimentos A/B orientados a CRO (Conversion Rate Optimization), con soporte para enfoques bayesianos y frecuentistas.

## Qué hace

Permite subir datos de experimentos (CSV) y comparar múltiples variantes para entender cuál funciona mejor, con distintas metodologías estadísticas:

- Bayesiano con priors [0,1]
- Bayesiano con priors [0, ∞]
- Frecuentista (bootstrap)
- Soporte con y sin `session_id`

El sistema selecciona el motor adecuado mediante un router interno (`adapter/engine_router.py`).

## Arquitectura

- `app.py`: UI completa en Streamlit (inputs, visualización, flujo de usuario)
- `adapter/engine_router.py`: capa de abstracción que enruta al motor estadístico
- `pablo_code/`: implementación de los distintos motores de cálculo

Cada motor devuelve una estructura homogénea:

- `summary`: resultados agregados
- `figures`: visualizaciones
- `pdf_bytes`: reporte exportable
- `log_text`: trazas
- `comparisons`: comparativas entre variantes

## Flujo de uso

1. Subir un CSV con los datos del experimento
2. Configurar:
   - Tipo de motor (bayesiano / frecuentista)
   - Uso de session_id
3. Ejecutar el análisis
4. Revisar:
   - Resultados resumidos
   - Gráficas
   - Comparativas entre variantes
5. Descargar reporte (si aplica)

## Requisitos

Listado en `requirements.txt`. Principales:

- streamlit
- pandas
- numpy
- matplotlib
- seaborn

## Cómo ejecutar

Desde la raíz del proyecto:

```
pip install -r requirements.txt
streamlit run app.py
```

## Notas

- El proyecto está preparado para trabajar con múltiples definiciones estadísticas de A/B testing.
- La separación por motores permite extender fácilmente nuevos métodos.
- La UI está fuertemente personalizada vía CSS dentro de Streamlit.

## Posibles mejoras

- Tests automáticos de los motores
- Validación más estricta del input CSV
- Persistencia de experimentos
- API para ejecución sin UI