from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from contextlib import redirect_stdout
import os

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------
LOGO_URL = "https://vml-thecocktail.com/img/logo_vml_thecocktail.png"

# -----------------------------------------------------------------------------
# FIX imports
# -----------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path[:0] = [p for p in (str(_HERE), str(_PARENT)) if p not in sys.path]

# -----------------------------------------------------------------------------
# Router engines
# -----------------------------------------------------------------------------
try:
    from adapter.engine_router import (
        run_engine,
        get_engine_label,
        ENGINE_0_1_NO_SID,
        ENGINE_0_1_SID,
        ENGINE_0_INF_NO_SID,
        ENGINE_0_INF_SID,
        ENGINE_FREQ_NO_SID,
        ENGINE_FREQ_SID,
    )
except Exception as e:
    st.error(
        "No se pudo importar `adapter.engine_router`.\n\n"
        "Revisa:\n"
        "- Que exista la carpeta `adapter/`.\n"
        "- Que `adapter/` tenga `__init__.py`.\n"
        "- Que exista `adapter/engine_router.py`.\n"
        "- Que estás ejecutando Streamlit desde la raíz correcta del proyecto.\n"
    )
    st.exception(e)
    st.stop()

# -----------------------------------------------------------------------------
# Streamlit config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Calculadora A/B",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Ensure OPENAI_API_KEY is available for downstream modules
# - Supports different key casings
# - Bridges st.secrets -> os.environ for libraries using getenv
# -----------------------------------------------------------------------------
try:
    key = None
    # Common variants
    for k in ("OPENAI_API_KEY", "openai_api_key", "OPENAI-API-KEY"):
        if k in st.secrets and st.secrets[k]:
            key = st.secrets[k]
            break

    # Fallback using .get (safer in some Streamlit contexts)
    if not key:
        key = st.secrets.get("OPENAI_API_KEY")

    if key:
        os.environ["OPENAI_API_KEY"] = key
    else:
        # Fallback manual input (useful if Streamlit secrets fail in cloud)
        user_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key
except Exception:
    # If secrets are not available, rely on existing environment variables
    pass

# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
--primary-color: #6366f1;
--secondary-color: #8b5cf6;
--success-color: #10b981;
--warning-color: #f59e0b;
--error-color: #ef4444;
--info-color: #3b82f6;
--background-light: #f8fafc;
--background-card: #ffffff;
--text-primary: #1e293b;
--text-secondary: #64748b;
--border-color: #e2e8f0;
}
.main-header {
font-size: 2.5rem;
color: var(--primary-color);
font-weight: 700;
margin-bottom: 2rem;
text-align: center;
}
.sub-header {
font-size: 1.5rem;
color: var(--text-primary);
font-weight: 600;
margin: 1.5rem 0;
}
.success-box {
background: linear-gradient(135deg, #10b981, #059669);
color: white;
padding: 1.25rem;
border-radius: 12px;
margin: 1.5rem 0;
box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.1);
}
.info-box {
background: linear-gradient(135deg, var(--info-color), var(--primary-color));
color: white;
padding: 1.25rem;
border-radius: 12px;
margin: 1.5rem 0;
box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.1);
}
.warning-box {
background: linear-gradient(135deg, var(--warning-color), #d97706);
color: white;
padding: 1.25rem;
border-radius: 12px;
margin: 1.5rem 0;
box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.1);
}
.error-box {
background: linear-gradient(135deg, var(--error-color), #dc2626);
color: white;
padding: 1.25rem;
border-radius: 12px;
margin: 1.5rem 0;
box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.1);
}
.section-spacer { margin: 3rem 0; }
.subsection-spacer { margin: 2rem 0; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
height: 50px;
padding-left: 20px;
padding-right: 20px;
background-color: var(--background-light);
border-radius: 8px;
color: var(--text-secondary);
font-weight: 500;
}
.stTabs [aria-selected="true"] {
background-color: #10b981 !important;
color: white !important;
}
.stButton > button {
background: linear-gradient(135deg, #64748b, #475569);
color: white;
border: none;
border-radius: 8px;
padding: 0.6rem 1.5rem;
font-weight: 600;
transition: all 0.3s ease;
}
.stButton > button:hover {
background: linear-gradient(135deg, #475569, #334155);
transform: translateY(-2px);
box-shadow: 0 4px 12px rgba(100, 116, 139, 0.3);
}
.stButton > button[kind="primary"] {
background: linear-gradient(135deg, #10b981, #059669);
}
.stButton > button[kind="primary"]:hover {
background: linear-gradient(135deg, #059669, #047857);
box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}
.result-card {
background: var(--background-card);
padding: 2rem;
border-radius: 16px;
border: 1px solid var(--border-color);
box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
margin: 1.5rem 0;
}
.choice-title {
font-size: 1.25rem;
font-weight: 700;
color: var(--text-primary);
margin-bottom: 0.5rem;
}
.choice-text {
color: var(--text-secondary);
line-height: 1.5;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
background: var(--background-card) !important;
border-radius: 16px !important;
border: 1px solid var(--border-color) !important;
box-shadow: 0 4px 10px rgba(0,0,0,0.06) !important;
padding: 0.4rem 0.4rem 1rem 0.4rem !important;
height: 100%;
}
div[data-testid="stVerticalBlockBorderWrapper"]
> div[data-testid="stVerticalBlock"] {
height: 100%;
display: flex;
flex-direction: column;
justify-content: space-between;
}
div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button {
width: auto !important;
}
div[data-testid="stColumn"] .stButton > button[kind="primary"] {
width: max-content;
}
.center-wrap {
max-width: 1050px;
margin: 0 auto;
}
.step-block { margin: 0 0 2rem 0; }
.step-content {
transition: filter .2s ease, opacity .2s ease, transform .2s ease;
}
.step-done .step-content {
filter: blur(2.5px);
opacity: 0.65;
transform: scale(0.997);
}
.step-active .step-content {
filter: none;
opacity: 1;
transform: none;
}
.logo-header {
display: flex;
justify-content: center;
align-items: center;
margin: 0.25rem 0 1.75rem 0;
}
.logo-header img {
max-width: 260px;
width: 100%;
height: auto;
}
.sidebar-title {
font-size: 2.05rem;
font-weight: 900;
color: var(--text-primary);
margin: 0.35rem 0 0.75rem 0;
}
.sidebar-section-title {
font-size: 1.35rem;
font-weight: 900;
color: var(--text-primary);
margin: 1.15rem 0 0.75rem 0;
}
details[data-testid="stExpander"] {
background: #f8fafc;
border: 1px solid var(--border-color);
border-radius: 14px;
padding: 0.35rem 0.55rem;
margin-bottom: 0.85rem;
}
details[data-testid="stExpander"] summary {
font-weight: 800;
color: var(--text-primary);
}
details[data-testid="stExpander"] summary p { margin: 0; }
details[data-testid="stExpander"] div[role="region"] { padding-top: 0.25rem; }
.sidebar-static-card {
background: #f8fafc;
border: 1px solid var(--border-color);
border-radius: 14px;
padding: 0.85rem 0.85rem;
margin: 0 0 0.85rem 0;
font-weight: 900;
color: var(--text-primary);
font-size: 1.1rem;
}
.dd-value {
color: var(--text-secondary);
font-weight: 700;
margin: 0.1rem 0 0.35rem 0;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-child(1) {
font-size: 0 !important;
line-height: 0 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-child(1)::before {
content: "Arrastra y suelta tu archivo aquí";
font-size: 14px !important;
line-height: 1.6 !important;
color: var(--text-primary);
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-child(2) {
font-size: 0 !important;
line-height: 0 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-child(2)::before {
content: "Límite 200 MB por archivo  •  CSV";
font-size: 12px !important;
line-height: 1.4 !important;
color: var(--text-secondary);
}
[data-testid="stFileUploaderDropzone"] button {
font-size: 0 !important;
color: transparent !important;
position: relative;
}
[data-testid="stFileUploaderDropzone"] button * {
color: transparent !important;
font-size: 0 !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
content: "📎  Adjuntar";
visibility: visible !important;
font-size: 14px !important;
font-weight: 600;
color: #727c92 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
def reset_wizard():
    st.session_state.wizard_step = 1
    st.session_state.enfoque = None
    st.session_state.session_id = None
    st.session_state.tipo_valores = None
    st.session_state.freq_interval_type = None
    st.session_state.ruta_ok = False
    st.session_state.selected_engine_key = None
    st.session_state.show_app = False
    st.session_state.pending_scroll_to = None
    st.session_state.outputs = None
    st.session_state.datos_procesados = False


def init_wizard_state():
    if "wizard_step" not in st.session_state:
        reset_wizard()


def set_engine_from_selected_model():
    st.session_state.outputs = None
    st.session_state.datos_procesados = False


def is_bayes_engine(engine_key: Optional[str]) -> bool:
    if not engine_key:
        return False
    return engine_key in {
        ENGINE_0_1_NO_SID,
        ENGINE_0_1_SID,
        ENGINE_0_INF_NO_SID,
        ENGINE_0_INF_SID,
    }


def is_freq_engine(engine_key: Optional[str]) -> bool:
    if not engine_key:
        return False
    return engine_key in {ENGINE_FREQ_NO_SID, ENGINE_FREQ_SID}


def check_route_and_set_model():
    enfoque = st.session_state.get("enfoque")
    session_id = st.session_state.get("session_id")
    tipo_valores = st.session_state.get("tipo_valores")
    freq_interval_type = st.session_state.get("freq_interval_type")

    if enfoque == "frecuentista":
        if session_id not in (True, False):
            st.session_state.ruta_ok = False
            st.session_state.selected_engine_key = None
            return
        if freq_interval_type not in ("centrado", "derecha", "izquierda"):
            st.session_state.ruta_ok = False
            st.session_state.selected_engine_key = None
            return
        st.session_state.ruta_ok = True
        st.session_state.selected_engine_key = (
            ENGINE_FREQ_SID if session_id else ENGINE_FREQ_NO_SID
        )
        return

    if enfoque == "bayesiano":
        if tipo_valores not in ("0_1", "0_inf") or session_id not in (True, False):
            st.session_state.ruta_ok = False
            st.session_state.selected_engine_key = None
            return
        st.session_state.ruta_ok = True
        if tipo_valores == "0_1" and session_id is False:
            st.session_state.selected_engine_key = ENGINE_0_1_NO_SID
        elif tipo_valores == "0_1" and session_id is True:
            st.session_state.selected_engine_key = ENGINE_0_1_SID
        elif tipo_valores == "0_inf" and session_id is False:
            st.session_state.selected_engine_key = ENGINE_0_INF_NO_SID
        else:
            st.session_state.selected_engine_key = ENGINE_0_INF_SID
        return

    st.session_state.ruta_ok = False
    st.session_state.selected_engine_key = None


def scroll_to_anchor(anchor_id: str):
    components.html(
        f"""
<script>
const el = window.parent.document.getElementById("{anchor_id}");
if (el) {{
  el.scrollIntoView({{ behavior: "smooth", block: "center" }});
}}
</script>
""",
        height=0,
    )


def go_to_step(step: int):
    st.session_state.wizard_step = step
    st.session_state.pending_scroll_to = f"step-{step}"
    st.rerun()


def step_open(step_num: int):
    current = st.session_state.wizard_step
    cls = "step-active" if current == step_num else "step-done"
    st.markdown(
        f'<div id="step-{step_num}" class="step-block {cls}"><div class="step-content">',
        unsafe_allow_html=True,
    )


def step_close():
    st.markdown("</div></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Modal — formato CSV
# -----------------------------------------------------------------------------
@st.dialog(" ", width="large")
def modal_formato_csv():
    st.markdown(
        """
      <h3 style="
          text-align: center;
          font-size: 1.4rem;
          font-weight: 700;
          color: #1e293b;
          margin: 0 0 1.25rem 0;
          padding-bottom: 0.75rem;
          border-bottom: 1px solid #e2e8f0;
      ">
          ¿Cómo debe ser el formato del CSV?
      </h3>
      <div style="text-align: center; margin-bottom: 1.5rem; color: #1e293b;
           font-size: 15px; line-height: 1.75;">
          El archivo CSV debe contener los datos del experimento agregados por día.
          Incluye las siguientes columnas:<br>
          <strong>día, visitas A, visitas B, conversiones A</strong> y
          <strong>conversiones B</strong>.<br><br>
          La columna <strong>día</strong> debe usar valores numéricos consecutivos
          (1, 2, 3…), representando el orden de los días del experimento
          (no fechas reales).
      </div>
      """,
        unsafe_allow_html=True,
    )
    df_ejemplo = pd.DataFrame(
        {
            "Día": list(range(1, 11)),
            "Visitas A": [43, 42, 22, 15, 59, 50, 42, 27, 51, 25],
            "Visitas B": [61, 45, 21, 16, 46, 55, 31, 26, 51, 57],
            "Conversiones A": [5, 7, 6, 7, 14, 3, 3, 6, 3, 7],
            "Conversiones B": [6, 7, 2, 8, 15, 7, 10, 8, 5, 9],
        }
    )
    col_left, col_table, col_right = st.columns([1, 4, 1])
    with col_table:
        st.dataframe(df_ejemplo, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Sidebar helpers
# -----------------------------------------------------------------------------
def _freq_hypothesis_from_interval(interval_type: Optional[str]) -> dict:
    if interval_type == "centrado":
        return {"tipo": "Two-Tailed", "direccion": None}
    if interval_type == "derecha":
        return {"tipo": "One-Tailed", "direccion": "Mejora (cola derecha)"}
    if interval_type == "izquierda":
        return {"tipo": "One-Tailed", "direccion": "Empeora (cola izquierda)"}
    return {"tipo": "—", "direccion": None}


def _render_sidebar_dropdowns(engine_key: Optional[str]):
    enfoque = st.session_state.get("enfoque")

    if enfoque == "bayesiano":
        st.markdown(
            '<div class="sidebar-title">Calculadora Bayesiana</div>',
            unsafe_allow_html=True,
        )
    elif enfoque == "frecuentista":
        st.markdown(
            '<div class="sidebar-title">Calculadora Frecuentista</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sidebar-title">Calculadora</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sidebar-section-title">Configuración del test</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Modelo Estadístico", expanded=False):
        if enfoque == "bayesiano":
            modelo_txt = "Bayesiano"
        elif enfoque == "frecuentista":
            modelo_txt = "Frecuentista"
        else:
            modelo_txt = "—"

        st.markdown(
            f'<div class="dd-value">{modelo_txt}</div>',
            unsafe_allow_html=True,
        )

        if enfoque == "bayesiano":
            st.markdown(
                "El enfoque bayesiano interpreta los resultados en términos de "
                "probabilidad directa. En lugar de preguntarse si el resultado es "
                "estadísticamente significativo, responde: ¿cuál es la probabilidad "
                "de que la variante B sea mejor que la A?\n\n"
                "- No necesitas un tamaño de muestra fijo.\n"
                "- Análisis de resultados basado en probabilidad.\n"
                "- Decisión más rápida: puedes parar el test cuando desees."
            )
        elif enfoque == "frecuentista":
            st.markdown(
                "El enfoque frecuentista comprueba si la diferencia observada podría "
                "deberse al azar, respondiendo: ¿la diferencia entre A y B es "
                "estadísticamente significativa? ¿podemos rechazar la hipótesis nula?\n\n"
                "- Debes calcular previamente la muestra y esperar hasta alcanzarla.\n"
                "- Análisis de resultados basado en p-value."
            )
        else:
            st.caption("Selecciona un modelo en el wizard.")

    st.markdown(
        '<div class="sidebar-section-title">Opciones de ejecución</div>',
        unsafe_allow_html=True,
    )

    if is_bayes_engine(engine_key):
        st.session_state.num_samples = 20000
        generate_pdf = st.checkbox(
            "Generar PDF",
            value=bool(st.session_state.get("generate_pdf", False)),
            help="Genera un reporte PDF con tablas y gráficos (si el motor lo soporta).",
        )
        include_ai = st.checkbox(
            "Interpretación IA (OpenAI)",
            value=bool(st.session_state.get("include_ai", True)),
            help="Solo si tienes OPENAI_API_KEY configurada en el entorno.",
        )
        st.session_state.generate_pdf = bool(generate_pdf)
        st.session_state.include_ai = bool(include_ai)

    elif is_freq_engine(engine_key):
        st.session_state.n_iteraciones = 10000
        generate_pdf = st.checkbox(
            "Generar PDF",
            value=bool(st.session_state.get("generate_pdf", False)),
            help="Genera un reporte PDF con tablas y gráficos (si el motor lo soporta).",
        )
        include_ai = st.checkbox(
            "Interpretación IA (OpenAI)",
            value=bool(st.session_state.get("include_ai", True)),
            help="Solo si tienes OPENAI_API_KEY configurada en el entorno.",
        )
        st.session_state.generate_pdf = bool(generate_pdf)
        st.session_state.include_ai = bool(include_ai)

    else:
        st.checkbox("Generar PDF", value=False, disabled=True)

    if enfoque == "bayesiano":
        tv = st.session_state.get("tipo_valores")

        if tv == "0_1":
            tv_txt = "Conversión única (Beta-Binomial)"
            tv_copy = (
                "Los usuarios pueden convertir sólo una vez. "
                "Se analizará mediante la distribución previa **Beta**, "
                "ideal para conversiones donde **0** = no convierte "
                "y **1** = convierte en la sesión."
            )
        elif tv == "0_inf":
            tv_txt = "Conversiones múltiples (Gamma-Poisson)"
            tv_copy = (
                "Se analizará mediante la distribución previa **Gamma-Poisson**, "
                "adecuada para conteos de métricas donde un usuario "
                "puede convertir más de una vez."
            )
        else:
            tv_txt = "—"
            tv_copy = ""

        with st.expander("Tipo de conversiones", expanded=False):
            st.markdown(
                f'<div class="dd-value">{tv_txt}</div>',
                unsafe_allow_html=True,
            )
            if tv_copy:
                st.markdown(tv_copy)

        with st.expander("Nivel de confianza", expanded=False):
            st.markdown(
                '<div class="dd-value">95% (Default)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "Es el umbral de probabilidad de que la variante sea mejor que el control. "
                "Un intervalo de credibilidad del **95%** es el más común y significa que, "
                "dados los datos y el modelo, existe un 95% de probabilidad de que el "
                "verdadero parámetro esté dentro de ese rango."
            )

        ua_txt = (
            "Con Session ID" if st.session_state.get("session_id") else "Sin Session ID"
        )

        with st.expander("Unidad de análisis", expanded=False):
            st.markdown(
                f'<div class="dd-value">{ua_txt}</div>',
                unsafe_allow_html=True,
            )
            if st.session_state.get("session_id"):
                st.markdown(
                    "Analizarás tu test A/B **con Session ID**. "
                    "El CSV de tu test A/B deberá contener una columna "
                    "con los Session ID de cada sesión."
                )
            else:
                st.markdown(
                    "Analizarás tu test A/B **sin Session ID**. "
                    "El análisis se realizará utilizando eventos y sesiones agregados."
                )

    elif enfoque == "frecuentista":
        hyp = _freq_hypothesis_from_interval(st.session_state.get("freq_interval_type"))
        tipo = hyp["tipo"]
        direccion = hyp["direccion"]

        with st.expander("Tipo de hipótesis", expanded=False):
            st.markdown(
                f'<div class="dd-value">{tipo}</div>',
                unsafe_allow_html=True,
            )
            if tipo == "Two-Tailed":
                st.markdown(
                    "Se analiza cualquier diferencia, tanto mejora como empeoramiento. "
                    "Este enfoque evalúa si existe un efecto estadísticamente significativo "
                    "sin asumir de antemano el sentido del cambio."
                )
            elif tipo == "One-Tailed":
                st.markdown(
                    "Se analiza únicamente una diferencia en una dirección específica, "
                    "ya sea mejorar o empeorar la métrica objetivo. En caso de seleccionar "
                    "One-Tailed, en el siguiente paso se indicará si el análisis debe "
                    "detectar una mejora o un empeoramiento de la métrica."
                )

        if direccion:
            with st.expander("Dirección de hipótesis", expanded=False):
                st.markdown(
                    f'<div class="dd-value">{direccion}</div>',
                    unsafe_allow_html=True,
                )
                if "derecha" in direccion.lower():
                    st.markdown(
                        "Se evalúa si el valor de la métrica en la variante es mayor "
                        "que en el control, según el criterio definido para el experimento."
                    )
                elif "izquierda" in direccion.lower():
                    st.markdown(
                        "Se evalúa si el valor de la métrica en la variante es menor "
                        "que en el control, según el criterio definido para el experimento."
                    )

        with st.expander("Nivel de confianza", expanded=False):
            st.markdown(
                '<div class="dd-value">95% (Default)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "Es el umbral que determina si el resultado es estadísticamente "
                "significativo. Con un nivel de confianza del **95%** (α = 0.05), "
                "rechazamos la hipótesis nula cuando el p-value es inferior a 0.05, "
                "lo que indica que la diferencia observada es poco probable que se "
                "deba al azar."
            )

        ua_txt = (
            "Con Session ID" if st.session_state.get("session_id") else "Sin Session ID"
        )

        with st.expander("Unidad de análisis", expanded=False):
            st.markdown(
                f'<div class="dd-value">{ua_txt}</div>',
                unsafe_allow_html=True,
            )
            if st.session_state.get("session_id"):
                st.markdown(
                    "Analizarás tu test A/B **con Session ID**. "
                    "El CSV de tu test A/B deberá contener una columna "
                    "con los Session ID de cada sesión."
                )
            else:
                st.markdown(
                    "Analizarás tu test A/B **sin Session ID**. "
                    "El análisis se realizará utilizando eventos y sesiones agregados."
                )

    else:
        with st.expander("Configuración", expanded=False):
            st.caption("Completa el wizard para ver la configuración.")


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def _parse_interval_value(val):
    if val is None:
        return None

    if isinstance(val, (list, tuple)):
        try:
            return [float(val[0]), float(val[1])]
        except Exception:
            return None

    if hasattr(val, "__len__") and not isinstance(val, str):
        try:
            return [float(val[0]), float(val[1])]
        except Exception:
            pass

    if isinstance(val, str):
        txt = val.strip()
        txt = (
            txt.replace("array(", "").replace(")", "").replace("[", "").replace("]", "")
        )
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        if len(parts) >= 2:
            try:
                return [float(parts[0]), float(parts[1])]
            except Exception:
                return None

    return None


def _get_bayes_comparison_lines(out, dia, g1, g2):
    comparisons = getattr(out, "comparisons", None)
    if not comparisons:
        return []

    paso_objetivo = None
    for paso in comparisons:
        if str(paso.get("dia")) == str(dia):
            paso_objetivo = paso
            break

    if paso_objetivo is None:
        return []

    clave = f"{g1}_vs_{g2}"
    stats = paso_objetivo.get(clave)
    if not isinstance(stats, dict):
        return []

    try:
        media_estim = float(stats.get("uplift_media", 0)) * 100
    except Exception:
        media_estim = 0.0

    try:
        prob_mejor = float(stats.get("prob_mejor", 0)) * 100
    except Exception:
        prob_mejor = 0.0

    ci_centered = _parse_interval_value(stats.get("ci_centered"))
    ci_right = _parse_interval_value(stats.get("ci_right"))
    ci_left = _parse_interval_value(stats.get("ci_left"))

    if ci_centered is None or ci_right is None or ci_left is None:
        return []

    str_cent = f"[{ci_centered[0]*100:.2f}%, {ci_centered[1]*100:.2f}%]"
    str_right = f"> {ci_right[0]*100:.2f}%"
    str_left = f"< {ci_left[1]*100:.2f}%"

    return [
        "",
        f"📈 Uplift (relativo {g1} vs {g2}):",
        f"  Media estimada: {media_estim:.2f}%",
        f"  ---------------------------------------------",
        f"  1. IC Centrado:   {str_cent} (Estándar)",
        f"  2. IC Suelo:      {str_right} (Mínimo asegurado 95%)",
        f"  3. IC Techo:      {str_left} (Máximo riesgo 95%)",
        f"  ---------------------------------------------",
        f"  Probabilidad de que {g1} > {g2}: {prob_mejor:.2f}%",
    ]


def _format_console_blocks(out, engine_key: Optional[str]) -> List[str]:
    if out is None or getattr(out, "summary", None) is None:
        return []
    df = out.summary.copy()
    if df.empty:
        return []

    def _pct(x):
        try:
            return f"{float(x)*100:.2f}%"
        except Exception:
            return str(x)

    blocks = []

    if is_bayes_engine(engine_key):
        if "dia" not in df.columns or "grupo" not in df.columns:
            return []

        for dia, ddf in df.groupby("dia", sort=False):
            ddf_sorted = ddf.sort_values("grupo")
            lines = [f"🗓️  {dia}"]
            grupos_en_dia = []

            for _, r in ddf_sorted.iterrows():
                g = str(r.get("grupo", ""))
                grupos_en_dia.append(g)
                visitas = r.get("visitas", "")
                conv = r.get("conversiones", "")
                acum_v = r.get("acum_visitas", "")
                acum_c = r.get("acum_conversiones", "")
                media = r.get("media", "")
                ci_low = r.get("ci_low", "")
                ci_high = r.get("ci_high", "")

                lines.append(f"Grupo {g}:")
                lines.append(
                    f"  📊 Acumulado: {acum_v} visitas | {acum_c} conversiones"
                )
                lines.append(f"  Visitas día: {visitas} | Conversiones día: {conv}")
                lines.append(f"  Media: {_pct(media)}")
                lines.append(f"  IC 95%: [{_pct(ci_low)}, {_pct(ci_high)}]")

            if len(grupos_en_dia) >= 2:
                g1 = grupos_en_dia[0]
                g2 = grupos_en_dia[1]
                lines.extend(_get_bayes_comparison_lines(out, dia, g1, g2))
                lines.extend(_get_bayes_comparison_lines(out, dia, g2, g1))

            blocks.append("\n".join(lines))
        return blocks

    if is_freq_engine(engine_key):
        r0 = df.iloc[0].to_dict()
        interval_type = st.session_state.get("freq_interval_type", "centrado")

        lines = [
            "==================================================",
            "           ANÁLISIS DE PRECISIÓN B vs A           ",
            "==================================================",
        ]

        if "n_visitas_A" in r0 and "n_visitas_B" in r0:
            lines.append(
                f"Diseño A             | Visitas: {int(r0.get('n_visitas_A', 0)):>8} | Convs: {int(r0.get('conv_A', 0)):>6}"
            )
            lines.append(
                f"Diseño B             | Visitas: {int(r0.get('n_visitas_B', 0)):>8} | Convs: {int(r0.get('conv_B', 0)):>6}"
            )
        else:
            ga = str(r0.get("grupo_A_col", "A"))
            gb = str(r0.get("grupo_B_col", "B"))
            lines.append(
                f"{ga} (A): {int(r0.get('n_A', 0))} filas | {float(r0.get('conv_A', 0))} convs | Media: {float(r0.get('media_A', 0)):.4f}"
            )
            lines.append(
                f"{gb} (B): {int(r0.get('n_B', 0))} filas | {float(r0.get('conv_B', 0))} convs | Media: {float(r0.get('media_B', 0)):.4f}"
            )

        lines.append("--------------------------------------------------")

        try:
            significancia = float(r0.get("precision_B_mejor", 0)) * 100
        except Exception:
            significancia = 0.0

        lines.append(f"NIVEL DE SIGNIFICANCIA DE QUE B > A: {significancia:.2f}%")

        if interval_type == "centrado":
            center_low = r0.get("ci_uplift_center_low", "")
            center_high = r0.get("ci_uplift_center_high", "")
            try:
                lines.append(
                    f"IC CENTRADO (UPLIFT): [{float(center_low):.2f}%, {float(center_high):.2f}%]"
                )
            except Exception:
                lines.append(f"IC CENTRADO (UPLIFT): [{center_low}%, {center_high}%]")

        elif interval_type == "derecha":
            right_left = r0.get("ci_right_95_left", "")
            try:
                lines.append(
                    f"COLA DERECHA (IC 95% IZQUIERDA): > {float(right_left):.2f}%"
                )
            except Exception:
                lines.append(f"COLA DERECHA (IC 95% IZQUIERDA): > {right_left}%")

        elif interval_type == "izquierda":
            left_right = r0.get("ci_left_95_right", "")
            try:
                lines.append(
                    f"COLA IZQUIERDA (IC 95% DERECHA): < {float(left_right):.2f}%"
                )
            except Exception:
                lines.append(f"COLA IZQUIERDA (IC 95% DERECHA): < {left_right}%")

        lines.append("==================================================")
        blocks.append("\n".join(lines))
        return blocks

    return []


# -----------------------------------------------------------------------------
# Wizard
# -----------------------------------------------------------------------------
def render_wizard():
    st.markdown('<div class="center-wrap">', unsafe_allow_html=True)

    st.markdown(
        f'<div class="logo-header"><img src="{LOGO_URL}" alt="VML THE COCKTAIL"></div>',
        unsafe_allow_html=True,
    )

    step = st.session_state.wizard_step
    pending = st.session_state.get("pending_scroll_to")
    if pending:
        scroll_to_anchor(pending)
        st.session_state.pending_scroll_to = None

    # ─────────────────────────────────────────────────────────────
    # PASO 1
    # ─────────────────────────────────────────────────────────────
    step_open(1)

    st.markdown(
        """
<div class="result-card">
<div class="choice-title">¡Bienvenido a la calculadora de tests A/B!</div>
<div class="choice-text">
    Esta calculadora te ayudará a tomar decisiones basadas en datos
    eligiendo entre el enfoque bayesiano o frecuentista.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="result-card">
<div class="choice-title">Elige el modelo que deseas utilizar para analizar tu test A/B</div>
<div class="choice-text">¿No sabes cuál elegir? No te preocupes: te explico cada uno de forma sencilla.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if step == 1:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            with st.container(border=True):
                st.markdown(
                    """
      <div style="min-height: 240px;">
          <div class="choice-title">Modelo Bayesiano</div>
          <div class="choice-text">
              El enfoque bayesiano interpreta los resultados en términos de probabilidad directa.
              <br><br>
              En lugar de preguntarse "¿es este resultado estadísticamente significativo?",
              responde preguntas como: "¿cuál es la probabilidad de que la variante B sea mejor que la A?"
              <br><br>
              <ul>
                  <li>No necesitas un tamaño de muestra fijo.</li>
                  <li>Análisis de resultados basado en probabilidad.</li>
                  <li>Decisión más rápida: puedes parar el test cuando desees.</li>
              </ul>
          </div>
      </div>
      """,
                    unsafe_allow_html=True,
                )
                _, col_btn, _ = st.columns([1, 2, 1])
                with col_btn:
                    st.markdown(
                        '<div style="height:20px;"></div>', unsafe_allow_html=True
                    )
                    if st.button(
                        "Modelo Bayesiano",
                        key="btn_bayesiano",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state.enfoque = "bayesiano"
                        st.session_state.session_id = None
                        st.session_state.tipo_valores = None
                        st.session_state.freq_interval_type = None
                        go_to_step(2)

        with col2:
            with st.container(border=True):
                st.markdown(
                    """
      <div style="min-height: 240px;">
          <div class="choice-title">Modelo Frecuentista</div>
          <div class="choice-text">
              El enfoque frecuentista se centra en comprobar si la diferencia observada
              podría deberse al azar, respondiendo preguntas como:
              "¿la diferencia entre A y B es estadísticamente significativa?"
              o "¿podemos rechazar la hipótesis nula?".
              <br><br>
              <ul>
                  <li>Debes calcular previamente la muestra y esperar hasta alcanzarla.</li>
                  <li>Análisis de resultados basado en p-value.</li>
              </ul>
          </div>
      </div>
      """,
                    unsafe_allow_html=True,
                )
                _, col_btn, _ = st.columns([1, 2, 1])
                with col_btn:
                    st.markdown(
                        '<div style="height:20px;"></div>', unsafe_allow_html=True
                    )
                    if st.button(
                        "Modelo Frecuentista",
                        key="btn_frecuentista",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state.enfoque = "frecuentista"
                        st.session_state.session_id = None
                        st.session_state.tipo_valores = None
                        st.session_state.freq_interval_type = None
                        go_to_step(2)

    else:
        enfoque_txt = (
            "Bayesiano" if st.session_state.enfoque == "bayesiano" else "Frecuentista"
        )
        st.markdown(
            f'<div class="success-box">✅ Seleccionado: <b>{enfoque_txt}</b></div>',
            unsafe_allow_html=True,
        )
        if st.button("Editar paso 1", key="edit_step_1"):
            st.session_state.wizard_step = 1
            st.session_state.session_id = None
            st.session_state.tipo_valores = None
            st.session_state.freq_interval_type = None
            st.session_state.ruta_ok = False
            st.session_state.selected_engine_key = None
            st.session_state.show_app = False
            st.session_state.pending_scroll_to = "step-1"
            st.session_state.outputs = None
            st.session_state.datos_procesados = False
            st.rerun()

    step_close()

    # ─────────────────────────────────────────────────────────────
    # PASO 2
    # ─────────────────────────────────────────────────────────────
    if step >= 2:
        step_open(2)
        st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

        enfoque_txt = (
            "Bayesiano" if st.session_state.enfoque == "bayesiano" else "Frecuentista"
        )
        st.markdown(
            f"""
<div class="result-card">
  <div class="choice-title">¡Buena elección!</div>
  <div class="choice-text">Has seleccionado analizar tu test A/B con el modelo {enfoque_txt}.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="result-card">
  <div class="choice-title">¿Puedes analizar tu test A/B con "Session ID" de cada sesión que ha formado parte del experimento?</div>
  <div class="choice-text">
      Para analizar correctamente un experimento A/B es necesario definir la unidad de análisis.
      <br><br>
      En entornos web, el uso del "Session ID" de Google Analytics 4 (GA4) permite identificar
      exposiciones y conversiones a nivel de sesión, evitando duplicidades y sesgos en el cálculo
      de resultados. Si dispones de Session ID, deberás añadir al CSV una columna con dichos datos.
      <br><br>
      ¿Es posible analizar este experimento utilizando el Session ID de GA4?
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if step == 2:
            c1, c2 = st.columns(2, gap="large")

            with c1:
                with st.container(border=True):
                    st.markdown(
                        """
          <div style="min-height: 130px;">
              <div class="choice-title">✅ Con Session ID</div>
              <div class="choice-text">
                  Dispondrás de mayor precisión en el análisis al identificar
                  exposiciones y conversiones a nivel de sesión, evitando duplicidades.
                  <br><br>
                  El CSV deberá incluir una columna con los Session ID de GA4.
              </div>
          </div>
          """,
                        unsafe_allow_html=True,
                    )
                    _, col_btn, _ = st.columns([1, 2, 1])
                    with col_btn:
                        st.markdown(
                            '<div style="height:20px;"></div>', unsafe_allow_html=True
                        )
                        if st.button(
                            "Tengo Session ID",
                            key="btn_sid_yes",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state.session_id = True
                            go_to_step(3)

            with c2:
                with st.container(border=True):
                    st.markdown(
                        """
          <div style="min-height: 130px;">
              <div class="choice-title">❌ Sin Session ID</div>
              <div class="choice-text">
                  El análisis se realizará utilizando eventos y sesiones agregados.
                  <br><br>
                  El CSV deberá contener datos de visitas y conversiones
                  agregados por día para cada variante.
              </div>
          </div>
          """,
                        unsafe_allow_html=True,
                    )
                    _, col_btn, _ = st.columns([1, 2, 1])
                    with col_btn:
                        st.markdown(
                            '<div style="height:20px;"></div>', unsafe_allow_html=True
                        )
                        if st.button(
                            "Sin Session ID",
                            key="btn_sid_no",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state.session_id = False
                            go_to_step(3)

            if st.button("⬅️ Volver", key="back_2"):
                go_to_step(1)

        else:
            sid_txt = (
                "con Session ID" if st.session_state.session_id else "sin Session ID"
            )
            st.markdown(
                f'<div class="success-box">✅ Seleccionado: <b>{sid_txt}</b></div>',
                unsafe_allow_html=True,
            )
            if st.button("Editar paso 2", key="edit_step_2"):
                st.session_state.wizard_step = 2
                st.session_state.tipo_valores = None
                st.session_state.freq_interval_type = None
                st.session_state.ruta_ok = False
                st.session_state.selected_engine_key = None
                st.session_state.show_app = False
                st.session_state.pending_scroll_to = "step-2"
                st.session_state.outputs = None
                st.session_state.datos_procesados = False
                st.rerun()

        step_close()

    # ─────────────────────────────────────────────────────────────
    # PASO 3 — BAYESIANO
    # ─────────────────────────────────────────────────────────────
    if step >= 3 and st.session_state.get("enfoque") == "bayesiano":
        step_open(3)
        st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

        perfect_text = (
            "Analizarás tu test A/B con Session ID. De esta manera, el CSV de tu test A/B "
            "deberá contener una columna con los Session ID."
            if st.session_state.session_id
            else "Analizarás tu test A/B sin Session ID. El análisis se realizará utilizando eventos y sesiones agregados."
        )

        st.markdown(
            f"""
<div class="result-card">
  <div class="choice-title">¡Perfecto!</div>
  <div class="choice-text">{perfect_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="result-card">
  <div class="choice-title">¿Los usuarios pueden convertir una sola vez o más de una vez la métrica a analizar?</div>
  <div class="choice-text">Selecciona el tipo de conversión que mejor describe tu métrica objetivo.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        if step == 3:
            c1, c2 = st.columns(2, gap="large")

            with c1:
                with st.container(border=True):
                    st.markdown(
                        """
          <div style="min-height: 110px;">
              <div class="choice-title">Conversiones únicas</div>
              <div class="choice-text">
                  El usuario puede convertir una sola vez. Se analiza mediante
                  la distribución previa <b>Beta</b>, ideal para conversiones
                  donde 0 = no convierte y 1 = convierte en la sesión.
              </div>
          </div>
          """,
                        unsafe_allow_html=True,
                    )
                    _, col_btn, _ = st.columns([1, 2, 1])
                    with col_btn:
                        st.markdown(
                            '<div style="height:20px;"></div>', unsafe_allow_html=True
                        )
                        if st.button(
                            "Conversiones únicas",
                            key="btn_01",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state.tipo_valores = "0_1"
                            go_to_step(4)

            with c2:
                with st.container(border=True):
                    st.markdown(
                        """
          <div style="min-height: 110px;">
              <div class="choice-title">Conversiones múltiples</div>
              <div class="choice-text">
                  Se analiza mediante la distribución previa <b>Gamma-Poisson</b>,
                  adecuada para conteos de métricas donde un usuario puede
                  convertir más de una vez.
              </div>
          </div>
          """,
                        unsafe_allow_html=True,
                    )
                    _, col_btn, _ = st.columns([1, 2, 1])
                    with col_btn:
                        st.markdown(
                            '<div style="height:20px;"></div>', unsafe_allow_html=True
                        )
                        if st.button(
                            "Conversiones múltiples",
                            key="btn_0inf",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state.tipo_valores = "0_inf"
                            go_to_step(4)

            if st.button("⬅️ Volver", key="back_3_bayes"):
                go_to_step(2)

        else:
            tipo_txt = (
                "Conversiones únicas"
                if st.session_state.tipo_valores == "0_1"
                else "Conversiones múltiples"
            )
            st.markdown(
                f'<div class="success-box">✅ Seleccionado: <b>{tipo_txt}</b></div>',
                unsafe_allow_html=True,
            )
            if st.button("Editar paso 3", key="edit_step_3_bayes"):
                st.session_state.wizard_step = 3
                st.session_state.ruta_ok = False
                st.session_state.selected_engine_key = None
                st.session_state.show_app = False
                st.session_state.pending_scroll_to = "step-3"
                st.session_state.outputs = None
                st.session_state.datos_procesados = False
                st.rerun()

        step_close()

    # ─────────────────────────────────────────────────────────────
    # PASO 3 — FRECUENTISTA
    # ─────────────────────────────────────────────────────────────
    if step >= 3 and st.session_state.get("enfoque") == "frecuentista":
        step_open(3)
        st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

        perfect_text = (
            "Analizarás tu test A/B con Session ID. De esta manera, el CSV de tu test A/B "
            "deberá contener una columna con los Session ID."
            if st.session_state.session_id
            else "Analizarás tu test A/B sin Session ID. El análisis se realizará utilizando eventos y sesiones agregados."
        )

        st.markdown(
            f"""
<div class="result-card">
  <div class="choice-title">¡Perfecto!</div>
  <div class="choice-text">{perfect_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="result-card">
  <div class="choice-title">¿Qué tipo de intervalo quieres utilizar para interpretar el uplift relativo?</div>
  <div class="choice-text">
      Selecciona el tipo de intervalo de confianza que mejor se adapta a tu hipótesis de test.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if step == 3:
            c1, c2, c3 = st.columns(3, gap="large")

            with c1:
                with st.container(border=True):
                    st.markdown(
                        """
          <div style="min-height: 110px;">
              <div class="choice-title">IC Centrado</div>
              <div class="choice-text">
                  Muestra el intervalo de confianza centrado del uplift relativo.
                  Opción estándar y más habitual para tests bidireccionales.
              </div>
          </div>
          """,
                        unsafe_allow_html=True,
                    )
                    _, col_btn, _ = st.columns([1, 3, 1])
                    with col_btn:
                        st.markdown(
                            '<div style="height:20px;"></div>', unsafe_allow_html=True
                        )
                        if st.button(
                            "IC centrado",
                            key="btn_freq_centrado",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state.freq_interval_type = "centrado"
                            go_to_step(4)

            with c2:
                with st.container(border=True):
                    st.markdown(
                        """
          <div style="min-height: 110px;">
              <div class="choice-title">Cola derecha</div>
              <div class="choice-text">
                  Utiliza el IC 95% izquierda. Indica el mínimo uplift
                  que podemos esperar con un 95% de confianza.
              </div>
          </div>
          """,
                        unsafe_allow_html=True,
                    )
                    _, col_btn, _ = st.columns([1, 3, 1])
                    with col_btn:
                        st.markdown(
                            '<div style="height:20px;"></div>', unsafe_allow_html=True
                        )
                        if st.button(
                            "Cola derecha",
                            key="btn_freq_derecha",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state.freq_interval_type = "derecha"
                            go_to_step(4)

            with c3:
                with st.container(border=True):
                    st.markdown(
                        """
          <div style="min-height: 110px;">
              <div class="choice-title">Cola izquierda</div>
              <div class="choice-text">
                  Utiliza el IC 95% derecha. Indica el máximo riesgo
                  de pérdida con un 95% de confianza.
              </div>
          </div>
          """,
                        unsafe_allow_html=True,
                    )
                    _, col_btn, _ = st.columns([1, 3, 1])
                    with col_btn:
                        st.markdown(
                            '<div style="height:20px;"></div>', unsafe_allow_html=True
                        )
                        if st.button(
                            "Cola izquierda",
                            key="btn_freq_izquierda",
                            type="primary",
                            use_container_width=True,
                        ):
                            st.session_state.freq_interval_type = "izquierda"
                            go_to_step(4)

            if st.button("⬅️ Volver", key="back_3_freq"):
                go_to_step(2)

        else:
            interval_txt = {
                "centrado": "IC centrado",
                "derecha": "Cola derecha",
                "izquierda": "Cola izquierda",
            }.get(st.session_state.freq_interval_type, "")
            st.markdown(
                f'<div class="success-box">✅ Seleccionado: <b>{interval_txt}</b></div>',
                unsafe_allow_html=True,
            )
            if st.button("Editar paso 3", key="edit_step_3_freq"):
                st.session_state.wizard_step = 3
                st.session_state.ruta_ok = False
                st.session_state.selected_engine_key = None
                st.session_state.show_app = False
                st.session_state.pending_scroll_to = "step-3"
                st.session_state.outputs = None
                st.session_state.datos_procesados = False
                st.rerun()

        step_close()

    # ─────────────────────────────────────────────────────────────
    # PASO 4
    # ─────────────────────────────────────────────────────────────
    if step >= 4:
        step_open(4)
        st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

        check_route_and_set_model()

        if st.session_state.ruta_ok:
            engine_key = st.session_state.selected_engine_key

            if is_bayes_engine(engine_key):
                extra = (
                    "El CSV deberá contener datos agregados por día (Conversiones X / Visitas X)."
                    if st.session_state.session_id is False
                    else "El CSV deberá contener SessionID y conversiones por sesión."
                )
            else:
                interval_txt = {
                    "centrado": "IC centrado",
                    "derecha": "Cola derecha (IC 95% izquierda)",
                    "izquierda": "Cola izquierda (IC 95% derecha)",
                }.get(st.session_state.freq_interval_type, "")

                extra = (
                    "Frecuentista sin Session ID: CSV agregado con Visitas/Conversiones A y B."
                    if st.session_state.session_id is False
                    else "Frecuentista con Session ID: CSV con columnas A y B (valores por sesión), NaN cuando no aplica."
                )
                extra += f"<br><br>Intervalo seleccionado: <b>{interval_txt}</b>"

            st.markdown(
                f"""
  <div class="result-card">
      <div class="choice-title">¡Listo!</div>
      <div class="choice-text">
          Ruta disponible ✅<br>
          Motor seleccionado: <b>{get_engine_label(engine_key)}</b><br><br>
          {extra}
      </div>
  </div>
  """,
                unsafe_allow_html=True,
            )

            col1, _ = st.columns([1, 5])
            with col1:
                if st.button(
                    "Analizar test A/B",
                    key="btn_go_app",
                    type="primary",
                ):
                    set_engine_from_selected_model()
                    st.session_state.show_app = True
                    st.rerun()

        else:
            st.markdown(
                """
  <div class="warning-box">
      <b>Todavía no disponible</b><br><br>
      Selecciona un enfoque y completa los pasos.
  </div>
  """,
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if st.button(
                    "Volver al inicio",
                    key="btn_back_home",
                    type="primary",
                ):
                    reset_wizard()
                    st.rerun()

        if st.button("⬅️ Volver al paso anterior", key="back_4"):
            go_to_step(3)

        step_close()

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Calculadora
# -----------------------------------------------------------------------------
def render_calculadora_actual():
    st.markdown(
        '<h2 class="main-header">Calculadora para Tests A/B</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="info-box">
Esta herramienta te permite analizar los resultados de tus tests A/B usando modelos estadísticos bayesianos o frecuentistas. Además, te ayudaremos a la interpretación de los resultados mediante Inteligencia Artificial. Sube un archivo CSV con el formato indicado y analiza tu test A/B.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    engine_key = st.session_state.get("selected_engine_key")

    with st.sidebar:
        _render_sidebar_dropdowns(engine_key)

        st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

        if st.button("Empezar nuevo análisis"):
            reset_wizard()
            st.rerun()

        if st.button("Cargar nuevos datos"):
            st.session_state.outputs = None
            st.session_state.datos_procesados = False
            st.success("Reiniciado correctamente")
            st.rerun()

    st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="sub-header">Cargar datos desde CSV</p>',
        unsafe_allow_html=True,
    )

    # ── Info hint sin fondo + botón link pegado a la derecha del texto ──
    # col_hint (5) → texto, col_link (2) → botón link, _ (3) → espacio vacío
    col_hint, col_link, _ = st.columns([2.5, 2, 5], vertical_alignment="center")

    with col_hint:
        st.markdown(
            """
          <div style="
              display: flex;
              align-items: center;
              gap: 0.65rem;
              font-size: 14px;
              color: #1e293b;
              line-height: 1.5;
              padding: 0.45rem 0;
          ">
              <span style="font-size: 18px; flex-shrink: 0;">💡</span>
              <span>Sube un CSV con el formato requerido.</span>
          </div>
          """,
            unsafe_allow_html=True,
        )

    with col_link:
        st.markdown(
            """
          <style>
          div[data-testid="stHorizontalBlock"]:has(#csv-link-anchor) {
            gap: 0 !important;
          }
          div[data-testid="element-container"]:has(#csv-link-anchor) {
              display: none !important;
              height: 0 !important;
              margin: 0 !important;
              padding: 0 !important;
          }
          div[data-testid="element-container"]:has(#csv-link-anchor)
            + div[data-testid="element-container"]
            button {
              background: none !important;
              border: none !important;
              color: #3b82f6 !important;
              text-decoration: underline !important;
              box-shadow: none !important;
              padding: 0 !important;
              font-size: 14px !important;
              font-weight: 500 !important;
              min-height: unset !important;
              height: auto !important;
          }
          div[data-testid="element-container"]:has(#csv-link-anchor)
            + div[data-testid="element-container"]
            button:hover {
              background: none !important;
              transform: none !important;
              box-shadow: none !important;
              color: #1d4ed8 !important;
              text-decoration: underline !important;
          }
          </style>
          <span id="csv-link-anchor"></span>
          """,
            unsafe_allow_html=True,
        )
        if st.button("Ver ejemplo de formato", key="btn_ver_formato"):
            modal_formato_csv()

    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Selecciona tu archivo CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ ¡Archivo cargado correctamente!")
            st.subheader("Vista previa de tus datos:")
            st.dataframe(df, use_container_width=True)

            if st.button("Analizar experimento", type="primary"):
                if not engine_key:
                    st.error("No hay motor seleccionado. Vuelve al wizard.")
                else:
                    if is_bayes_engine(engine_key):
                        config = {
                            "num_samples": st.session_state.get("num_samples", 20000),
                            "generate_pdf": st.session_state.get("generate_pdf", False),
                            "include_ai": st.session_state.get("include_ai", False),
                        }
                    else:
                        config = {
                            "n_iteraciones": st.session_state.get(
                                "n_iteraciones", 10000
                            ),
                            "generate_pdf": st.session_state.get("generate_pdf", False),
                            "include_ai": st.session_state.get("include_ai", False),
                        }

                    try:
                        with st.spinner("Ejecutando motor..."):
                            out = run_engine(engine_key, df, config)
                        st.session_state.outputs = out
                        st.session_state.datos_procesados = True
                        st.success("Motor ejecutado.")
                    except NotImplementedError as e:
                        st.warning(str(e))
                    except Exception as e:
                        st.error(f"❌ Error ejecutando el motor: {e}")

        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    if st.session_state.get("datos_procesados", False):
        st.markdown("---")
        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
        st.markdown(
            '<h2 class="main-header">Resultados</h2>',
            unsafe_allow_html=True,
        )

        out = st.session_state.get("outputs")
        if out is None:
            st.info("No hay outputs disponibles.")
            return

        if getattr(out, "summary", None) is not None:
            st.subheader("Resumen")
            summary_df = out.summary.copy()
            summary_df = summary_df[
                [c for c in summary_df.columns if not c.startswith("ci_")]
            ]
            st.dataframe(summary_df, use_container_width=True)

            console_blocks = _format_console_blocks(out, engine_key)
            if console_blocks:
                st.subheader("Salida tipo consola")
                for b in console_blocks:
                    st.code(b)

        if getattr(out, "log_text", None):
            if st.session_state.get("include_ai", False):
                st.subheader("Interpretación IA")
            else:
                st.subheader("Interpretación / Log")
            st.code(out.log_text)

        if getattr(out, "figures", None):
            st.subheader("Gráficos")
            for fig in out.figures:
                try:
                    st.pyplot(fig)
                except Exception:
                    pass

        if getattr(out, "pdf_bytes", None):
            st.subheader("Reporte")
            st.download_button(
                "📄 Descargar PDF",
                data=out.pdf_bytes,
                file_name="reporte.pdf",
                mime="application/pdf",
            )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.markdown("---")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
init_wizard_state()

if st.session_state.get("show_app", False) and not st.session_state.get(
    "selected_engine_key"
):
    check_route_and_set_model()

if st.session_state.get("show_app", False):
    render_calculadora_actual()
else:
    render_wizard()
