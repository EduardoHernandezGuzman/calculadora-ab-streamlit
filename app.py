import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from contextlib import redirect_stdout

from adapter.engine_router import (
    run_engine,
    get_engine_label,
    ENGINE_0_1_NO_SID,
    ENGINE_0_1_SID,
    ENGINE_0_INF_NO_SID,
    ENGINE_0_INF_SID,
)

# =========================
# Configuración de la página
# =========================
st.set_page_config(
    page_title="Calculadora A/B",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS (manteniendo tu estilo + blur steps)
# =========================
st.markdown("""
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

.choice-card {
background: var(--background-card);
padding: 1.6rem;
border-radius: 16px;
border: 1px solid var(--border-color);
box-shadow: 0 4px 10px rgba(0,0,0,0.06);
height: 100%;
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

.center-wrap {
max-width: 1050px;
margin: 0 auto;
}

/* ---- Wizard blur sections ---- */
.step-block {
margin: 0 0 2rem 0;
}

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

</style>
""", unsafe_allow_html=True)

# =========================
# Helpers de estado
# =========================
def reset_wizard():
    st.session_state.wizard_step = 1
    st.session_state.enfoque = None           # "bayesiano" | "frecuentista"
    st.session_state.session_id = None        # True | False
    st.session_state.tipo_valores = None      # "0_1" | "0_inf"
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
    """
    Inicializa el motor elegido por el wizard.
    (El motor real se conectará en adapter/engine_router.py)
    """
    st.session_state.outputs = None
    st.session_state.datos_procesados = False


def check_route_and_set_model():
    enfoque = st.session_state.get("enfoque")
    session_id = st.session_state.get("session_id")
    tipo_valores = st.session_state.get("tipo_valores")

    # Habilitamos Bayesiano (4 combinaciones). Frecuentista queda como no disponible.
    if enfoque != "bayesiano":
        st.session_state.ruta_ok = False
        st.session_state.selected_engine_key = None
        return

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
        unsafe_allow_html=True
    )


def step_close():
    st.markdown("</div></div>", unsafe_allow_html=True)

# =========================
# Wizard UI (ACUMULATIVO + BLUR + AUTO-SCROLL)
# =========================
def render_wizard():
    st.markdown('<div class="center-wrap">', unsafe_allow_html=True)

    st.markdown('<h2 class="main-header">VML THE COCKTAIL</h2>', unsafe_allow_html=True)

    step = st.session_state.wizard_step

    pending = st.session_state.get("pending_scroll_to")
    if pending:
        scroll_to_anchor(pending)
        st.session_state.pending_scroll_to = None

    # STEP 1
    step_open(1)

    st.markdown("""
    <div class="result-card">
        <div class="choice-title">¡Bienvenido a la calculadora de tests A/B!</div>
        <div class="choice-text">
            Esta calculadora te ayudará a tomar decisiones basadas en datos eligiendo entre el enfoque bayesiano o frecuentista, los dos modelos estadísticos más comunes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="result-card">
        <div class="choice-title">Elige el modelo que deseas utilizar para analizar tu test A/B</div>
        <div class="choice-text">¿No sabes cuál elegir? No te preocupes: te explico cada uno de forma sencilla.</div>
    </div>
    """, unsafe_allow_html=True)

    if step == 1:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("""
            <div class="choice-card">
                <div class="choice-title">Modelo Bayesiano</div>
                <div class="choice-text">
                    El enfoque bayesiano interpreta los resultados en términos de probabilidad directa.
                    <br><br>
                    En lugar de preguntarse “¿es este resultado estadísticamente significativo?”, responde preguntas como:
                    “¿cuál es la probabilidad de que la variante B sea mejor que la A?”
                    <ul>
                        <li>No necesitas un tamaño de muestra fijo.</li>
                        <li>Análisis de resultados vasado en probabilidad.</li>
                        <li>Decisión más rápida: puedes parar cuando desees.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Elegir modelo Bayesiano", key="btn_bayesiano", type="primary"):
                st.session_state.enfoque = "bayesiano"
                go_to_step(2)

        with col2:
            st.markdown("""
            <div class="choice-card">
                <div class="choice-title">Modelo Frecuentista</div>
                <div class="choice-text">
                    El enfoque frecuentista se centra en comprobar si la diferencia observada podría deberse al azar,
                    respondiendo preguntas como: “¿la diferencia A vs B es estadísticamente significativa?” o “¿podemos rechazar la hipótesis nula?”.
                    <ul>
                        <li>Debes calcular previamente la muestra y esperar hasta alcanzarla.</li>
                        <li>Análisis de resultados basado en p-value.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Elegir modelo Frecuentista", key="btn_frecuentista", type="primary"):
                st.session_state.enfoque = "frecuentista"
                go_to_step(2)
    else:
        enfoque_txt = "Bayesiano" if st.session_state.enfoque == "bayesiano" else "Frecuentista"
        st.markdown(f"""
        <div class="success-box">
            ✅ Seleccionado: <b>{enfoque_txt}</b>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Editar paso 1", key="edit_step_1"):
            st.session_state.wizard_step = 1
            st.session_state.session_id = None
            st.session_state.tipo_valores = None
            st.session_state.ruta_ok = False
            st.session_state.selected_engine_key = None
            st.session_state.show_app = False
            st.session_state.pending_scroll_to = "step-1"
            st.session_state.outputs = None
            st.session_state.datos_procesados = False
            st.rerun()

    step_close()

    # STEP 2
    if step >= 2:
        step_open(2)

        st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

        enfoque_txt = "Bayesiano" if st.session_state.enfoque == "bayesiano" else "Frecuentista"
        st.markdown(f"""
        <div class="result-card">
            <div class="choice-title">¡Buena elección!</div>
            <div class="choice-text">Has seleccionado analizar tu test A/B con el modelo {enfoque_txt}.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="result-card">
            <div class="choice-title">¿Puedes analizar tu test A/B con “Session ID” de cada sesión que ha formado parte del experimento?</div>
            <div class="choice-text">
                Para analizar correctamente un experimento A/B es necesario definir la unidad de análisis.
                En entornos web, el uso del “Session ID” de GA4 permite identificar exposiciones y conversiones a nivel de sesión, evitando duplicidades y sesgos en el cáculo de resultados.
                <br><br>
                ¿Es posible analizar este experimento utilizando el Session ID de GA4?
            </div>
        </div>
        """, unsafe_allow_html=True)

        if step == 2:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                if st.button("Tengo Session ID", key="btn_sid_yes", type="primary"):
                    st.session_state.session_id = True
                    go_to_step(3)
            with c2:
                if st.button("No tengo Session ID", key="btn_sid_no", type="primary"):
                    st.session_state.session_id = False
                    go_to_step(3)

            if st.button("⬅️ Volver", key="back_2"):
                go_to_step(1)
        else:
            sid_txt = "con Session ID" if st.session_state.session_id else "sin Session ID"
            st.markdown(f"""
            <div class="success-box">
                ✅ Seleccionado: <b>{sid_txt}</b>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Editar paso 2", key="edit_step_2"):
                st.session_state.wizard_step = 2
                st.session_state.tipo_valores = None
                st.session_state.ruta_ok = False
                st.session_state.selected_engine_key = None
                st.session_state.show_app = False
                st.session_state.pending_scroll_to = "step-2"
                st.session_state.outputs = None
                st.session_state.datos_procesados = False
                st.rerun()

        step_close()

    # STEP 3
    if step >= 3:
        step_open(3)

        st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

        sid_txt = "con Session ID" if st.session_state.session_id else "sin Session ID"
        st.markdown(f"""
        <div class="result-card">
            <div class="choice-title">¡Perfecto!</div>
            <div class="choice-text">Analizarás tu test A/B {sid_txt}.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="result-card">
            <div class="choice-title">¿Los valores de tu test van de 0 a 1 o van desde 0 a infinito?</div>
            <div class="choice-text">
                <ul>
                    <li><b>Valores entre 0 y 1</b>: de esta manera se analizará mediante la distribución previa Beta, ideal para conversiones (siendo 0 la no conversión y 1 si el usuario ha convertido en la sesión).</li>
                    <li><b>Valores de 0 a infinito</b>: con esta opción se analizará mediante la distrubución previa Gamma-Poisson, es adecuada para conteos de métricas.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if step == 3:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                if st.button("Valores entre 0 y 1", key="btn_01", type="primary"):
                    st.session_state.tipo_valores = "0_1"
                    go_to_step(4)
            with c2:
                if st.button("Valores de 0 a infinito", key="btn_0inf", type="primary"):
                    st.session_state.tipo_valores = "0_inf"
                    go_to_step(4)

            if st.button("⬅️ Volver", key="back_3"):
                go_to_step(2)
        else:
            tipo_txt = "Valores entre 0 y 1" if st.session_state.tipo_valores == "0_1" else "Valores de 0 a infinito"
            st.markdown(f"""
            <div class="success-box">
                ✅ Seleccionado: <b>{tipo_txt}</b>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Editar paso 3", key="edit_step_3"):
                st.session_state.wizard_step = 3
                st.session_state.ruta_ok = False
                st.session_state.selected_engine_key = None
                st.session_state.show_app = False
                st.session_state.pending_scroll_to = "step-3"
                st.session_state.outputs = None
                st.session_state.datos_procesados = False
                st.rerun()

        step_close()

    # STEP 4
    if step >= 4:
        step_open(4)

        st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

        check_route_and_set_model()

        if st.session_state.ruta_ok:
            extra = (
                "De esta manera, el CSV de tu test A/B deberá contener eventos y sesiones agregados."
                if st.session_state.session_id is False else
                "De esta manera, el CSV deberá contener una columna con los Session ID."
            )
            st.markdown(f"""
            <div class="result-card">
                <div class="choice-title">¡Perfecto!</div>
                <div class="choice-text">
                    Ruta disponible ✅<br>
                    Motor seleccionado: <b>{get_engine_label(st.session_state.selected_engine_key)}</b><br><br>
                    {extra}
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if st.button("Analizar test A/B", key="btn_go_app", type="primary"):
                    set_engine_from_selected_model()
                    st.session_state.show_app = True
                    st.rerun()

        else:
            st.markdown("""
            <div class="warning-box">
                <b>Todavía no disponible</b><br><br>
                Ahora mismo la capa visual solo habilita rutas bayesianas.
                Si elegiste “Frecuentista”, vuelve atrás y selecciona “Bayesiano”.
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if st.button("Volver al inicio", key="btn_back_home", type="primary"):
                    reset_wizard()
                    st.rerun()

        if st.button("⬅️ Volver al paso anterior", key="back_4"):
            go_to_step(3)

        step_close()

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# App actual (capa visual para motor)
# =========================
def render_calculadora_actual():
    st.markdown('<h2 class="main-header">Calculadora Bayesiana para Tests A/B</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Esta herramienta te permite analizar los resultados de tus pruebas A/B utilizando estadística bayesiana.
    Sube un archivo CSV con tus datos o ingresa la información manualmente.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="sub-header">Motor seleccionado</p>', unsafe_allow_html=True)
        engine_key = st.session_state.get("selected_engine_key")
        st.info(f"**{get_engine_label(engine_key)}**")

        if st.button("⬅️ Volver al inicio (Wizard)"):
            reset_wizard()
            st.rerun()

        st.markdown('<p class="sub-header">Configuración</p>', unsafe_allow_html=True)

        umbral_prob = st.slider(
            "Umbral de probabilidad para decisión",
            min_value=0.8,
            max_value=0.99,
            value=0.95,
            step=0.01,
            format="%.2f",
            key="umbral_prob"
        )

        umbral_mejora = st.slider(
            "Umbral de mejora mínima",
            min_value=0.01,
            max_value=0.20,
            value=0.01,
            step=0.01,
            format="%.2f",
            key="umbral_mejora"
        )

        if st.button("Reiniciar"):
            st.session_state.outputs = None
            st.session_state.datos_procesados = False
            st.success("Reiniciado correctamente")
            st.rerun()

    # Tabs
    st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Cargar CSV", "✏️ Entrada manual", "📋 Formato CSV"])

    # TAB 1 CSV
    with tab1:
        st.markdown('<p class="sub-header">Cargar datos desde CSV</p>', unsafe_allow_html=True)
        st.info("💡 Si no sabes cómo preparar tu archivo CSV, revisa la pestaña **'Formato CSV'**.")

        uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # TODO: cuando conectemos los motores, validaremos columnas según engine_key
                st.success("✅ ¡Archivo cargado correctamente!")
                st.subheader("Vista previa de tus datos:")
                st.dataframe(df, use_container_width=True)

                if st.button("🚀 Procesar datos del CSV", type="primary"):
                    engine_key = st.session_state.get("selected_engine_key")
                    if not engine_key:
                        st.error("No hay motor seleccionado. Vuelve al wizard.")
                    else:
                        config = {
                            "umbral_prob": st.session_state.get("umbral_prob", 0.95),
                            "umbral_mejora": st.session_state.get("umbral_mejora", 0.01),
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

    # TAB 2 Manual (placeholder)
    with tab2:
        st.markdown('<p class="sub-header">Entrada manual de datos</p>', unsafe_allow_html=True)
        st.info("Entrada manual: la conectaremos cuando tengamos definidos los inputs exactos para cada motor de Pablo.")

    # TAB 3 Formato CSV (placeholder genérico)
    with tab3:
        st.markdown('<p class="sub-header">Cómo preparar tu archivo CSV</p>', unsafe_allow_html=True)
        st.info("El formato exacto dependerá del motor elegido ([0,1] vs [0,∞] y con/sin Session ID). Lo dejaremos definido al integrar los 4 scripts de Pablo.")

    # Resultados
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    if st.session_state.get("datos_procesados", False):
        st.markdown("---")
        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
        st.markdown('<h2 class="main-header">Resultados</h2>', unsafe_allow_html=True)

        out = st.session_state.get("outputs")
        if out is None:
            st.info("No hay outputs disponibles.")
            return

        # Summary
        if getattr(out, "summary", None) is not None:
            st.subheader("Resumen")
            st.dataframe(out.summary, use_container_width=True)

        # Logs
        if getattr(out, "log_text", None):
            st.subheader("Log")
            st.code(out.log_text)

        # Figures
        if getattr(out, "figures", None):
            st.subheader("Gráficos")
            for fig in out.figures:
                try:
                    st.pyplot(fig)
                except Exception:
                    pass

        # PDF
        if getattr(out, "pdf_bytes", None):
            st.subheader("Reporte")
            st.download_button(
                "📄 Descargar PDF",
                data=out.pdf_bytes,
                file_name="reporte.pdf",
                mime="application/pdf",
            )

    # Footer
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 20px;">
    <p style="margin: 0; color: #555;">Idea y concepto: <strong>Claudia de la Cruz</strong> &nbsp;|&nbsp; Desarrollo: <strong>Pablo González</strong> &nbsp;|&nbsp; Desarrollo visual: <strong>Eduardo Hernández</strong></p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# MAIN
# =========================
init_wizard_state()

if st.session_state.get("show_app", False):
    render_calculadora_actual()
else:
    render_wizard()