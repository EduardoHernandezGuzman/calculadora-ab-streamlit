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
  ENGINE_FREQ_NO_SID,
  ENGINE_FREQ_SID,
)

st.set_page_config(
  page_title="Calculadora A/B",
  page_icon="📊",
  layout="wide",
  initial_sidebar_state="expanded"
)

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

def reset_wizard():
  st.session_state.wizard_step = 1
  st.session_state.enfoque = None
  st.session_state.session_id = None
  st.session_state.tipo_valores = None
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

def is_bayes_engine(engine_key: str | None) -> bool:
  if not engine_key:
      return False
  return engine_key in {
      ENGINE_0_1_NO_SID, ENGINE_0_1_SID, ENGINE_0_INF_NO_SID, ENGINE_0_INF_SID
  }

def is_freq_engine(engine_key: str | None) -> bool:
  if not engine_key:
      return False
  return engine_key in {ENGINE_FREQ_NO_SID, ENGINE_FREQ_SID}

def check_route_and_set_model():
  enfoque = st.session_state.get("enfoque")
  session_id = st.session_state.get("session_id")
  tipo_valores = st.session_state.get("tipo_valores")

  if enfoque == "frecuentista":
      if session_id not in (True, False):
          st.session_state.ruta_ok = False
          st.session_state.selected_engine_key = None
          return
      st.session_state.ruta_ok = True
      st.session_state.selected_engine_key = ENGINE_FREQ_SID if session_id else ENGINE_FREQ_NO_SID
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
      unsafe_allow_html=True
  )

def step_close():
  st.markdown("</div></div>", unsafe_allow_html=True)

def render_wizard():
  st.markdown('<div class="center-wrap">', unsafe_allow_html=True)
  st.markdown('<h2 class="main-header">VML THE COCKTAIL</h2>', unsafe_allow_html=True)

  step = st.session_state.wizard_step

  pending = st.session_state.get("pending_scroll_to")
  if pending:
      scroll_to_anchor(pending)
      st.session_state.pending_scroll_to = None

  step_open(1)

  st.markdown("""
  <div class="result-card">
      <div class="choice-title">¡Bienvenido a la calculadora de tests A/B!</div>
      <div class="choice-text">
          Esta calculadora te ayudará a tomar decisiones basadas en datos eligiendo entre el enfoque bayesiano o frecuentista.
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
                  <ul>
                      <li>No necesitas un tamaño de muestra fijo.</li>
                      <li>Análisis basado en probabilidad.</li>
                      <li>Decisión más rápida: puedes parar cuando desees.</li>
                  </ul>
              </div>
          </div>
          """, unsafe_allow_html=True)
          if st.button("Elegir modelo Bayesiano", key="btn_bayesiano", type="primary"):
              st.session_state.enfoque = "bayesiano"
              st.session_state.session_id = None
              st.session_state.tipo_valores = None
              go_to_step(2)

      with col2:
          st.markdown("""
          <div class="choice-card">
              <div class="choice-title">Modelo Frecuentista</div>
              <div class="choice-text">
                  El enfoque frecuentista estima precisión/IC y permite contrastar si B supera a A (bootstrap).
                  <ul>
                      <li>Análisis basado en remuestreo (bootstrap).</li>
                      <li>Salida: precisión + intervalo de confianza.</li>
                  </ul>
              </div>
          </div>
          """, unsafe_allow_html=True)
          if st.button("Elegir modelo Frecuentista", key="btn_frecuentista", type="primary"):
              st.session_state.enfoque = "frecuentista"
              st.session_state.session_id = None
              st.session_state.tipo_valores = None
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

  if step >= 2:
      step_open(2)

      st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

      enfoque_txt = "Bayesiano" if st.session_state.enfoque == "bayesiano" else "Frecuentista"
      st.markdown(f"""
      <div class="result-card">
          <div class="choice-title">¡Perfecto!</div>
          <div class="choice-text">Has seleccionado analizar tu test A/B con el modelo <b>{enfoque_txt}</b>.</div>
      </div>
      """, unsafe_allow_html=True)

      st.markdown("""
      <div class="result-card">
          <div class="choice-title">¿Puedes analizar tu test A/B con "Session ID"?</div>
          <div class="choice-text">
              En entornos web, el uso del "Session ID" de GA4 permite identificar exposiciones y conversiones a nivel de sesión.
              <br><br>
              ¿Es posible analizar este experimento utilizando Session ID?
          </div>
      </div>
      """, unsafe_allow_html=True)

      if step == 2:
          c1, c2 = st.columns(2, gap="large")
          with c1:
              if st.button("Tengo Session ID", key="btn_sid_yes", type="primary"):
                  st.session_state.session_id = True
                  if st.session_state.enfoque == "frecuentista":
                      go_to_step(4)
                  else:
                      go_to_step(3)
          with c2:
              if st.button("No tengo Session ID", key="btn_sid_no", type="primary"):
                  st.session_state.session_id = False
                  if st.session_state.enfoque == "frecuentista":
                      go_to_step(4)
                  else:
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

  if step >= 3 and st.session_state.get("enfoque") == "bayesiano":
      step_open(3)

      st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

      sid_txt = "con Session ID" if st.session_state.session_id else "sin Session ID"
      st.markdown(f"""
      <div class="result-card">
          <div class="choice-title">Tipo de métrica</div>
          <div class="choice-text">Analizarás tu test A/B {sid_txt}. Ahora elige el tipo de valores.</div>
      </div>
      """, unsafe_allow_html=True)

      st.markdown("""
      <div class="result-card">
          <div class="choice-title">¿Los valores van de 0 a 1 o van de 0 a infinito?</div>
          <div class="choice-text">
              <ul>
                  <li><b>0 a 1</b>: prior Beta (conversiones 0/1).</li>
                  <li><b>0 a ∞</b>: prior Gamma (conteos por visita/sesión).</li>
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

  if step >= 4:
      step_open(4)
      st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)

      check_route_and_set_model()

      if st.session_state.ruta_ok:
          engine_key = st.session_state.selected_engine_key

          if is_bayes_engine(engine_key):
              extra = (
                  "El CSV deberá contener datos agregados por día (Conversiones X / Visitas X)."
                  if st.session_state.session_id is False else
                  "El CSV deberá contener SessionID y conversiones por sesión (según el motor de Pablo)."
              )
          else:
              extra = (
                  "Frecuentista sin Session ID: CSV agregado con Visitas/Conversiones A y B."
                  if st.session_state.session_id is False else
                  "Frecuentista con Session ID: CSV con columnas A y B (valores por sesión), NaN cuando no aplica."
              )

          st.markdown(f"""
          <div class="result-card">
              <div class="choice-title">¡Listo!</div>
              <div class="choice-text">
                  Ruta disponible ✅<br>
                  Motor seleccionado: <b>{get_engine_label(engine_key)}</b><br><br>
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
              Selecciona un enfoque y completa los pasos.
          </div>
          """, unsafe_allow_html=True)

          c1, c2, c3 = st.columns([1, 2, 1])
          with c2:
              if st.button("Volver al inicio", key="btn_back_home", type="primary"):
                  reset_wizard()
                  st.rerun()

      if st.button("⬅️ Volver al paso anterior", key="back_4"):
          if st.session_state.get("enfoque") == "bayesiano":
              go_to_step(3)
          else:
              go_to_step(2)

      step_close()

  st.markdown('</div>', unsafe_allow_html=True)

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
      txt = txt.replace("array(", "").replace(")", "").replace("[", "").replace("]", "")
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
      f"  Probabilidad de que {g1} > {g2}: {prob_mejor:.2f}%"
  ]

def _format_console_blocks(out, engine_key: str | None) -> list[str]:
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
              lines.append(f"  📊 Acumulado: {acum_v} visitas | {acum_c} conversiones")
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
      lines = ["==================================================", "           ANÁLISIS DE PRECISIÓN B vs A           ", "=================================================="]
      if "n_visitas_A" in r0 and "n_visitas_B" in r0:
          lines.append(f"Diseño A             | Visitas: {int(r0.get('n_visitas_A', 0)):>8} | Convs: {int(r0.get('conv_A', 0)):>6}")
          lines.append(f"Diseño B             | Visitas: {int(r0.get('n_visitas_B', 0)):>8} | Convs: {int(r0.get('conv_B', 0)):>6}")
      else:
          ga = str(r0.get("grupo_A_col", "A"))
          gb = str(r0.get("grupo_B_col", "B"))
          lines.append(f"{ga} (A): {int(r0.get('n_A', 0))} filas | {float(r0.get('conv_A', 0))} convs | Media: {float(r0.get('media_A', 0)):.4f}")
          lines.append(f"{gb} (B): {int(r0.get('n_B', 0))} filas | {float(r0.get('conv_B', 0))} convs | Media: {float(r0.get('media_B', 0)):.4f}")

      lines.append("--------------------------------------------------")
      prec = float(r0.get("precision_B_mejor", 0))
      lines.append(f"          Precisión de que B > A: {prec*100:.2f}%          ")
      cd_low = r0.get("ci_diff_low", "")
      cd_high = r0.get("ci_diff_high", "")
      try:
          lines.append(f"            IC 95% Izquierda: {float(cd_low):.5f}            ")
          lines.append(f"            IC 95% Derecha:   {float(cd_high):.5f}            ")
      except Exception:
          lines.append(f"            IC 95% Izquierda: {cd_low}            ")
          lines.append(f"            IC 95% Derecha:   {cd_high}            ")
      lines.append("==================================================")
      blocks.append("\n".join(lines))
      return blocks

  return []

def render_calculadora_actual():
  st.markdown('<h2 class="main-header">Calculadora para Tests A/B</h2>', unsafe_allow_html=True)
  st.markdown("""
  <div class="info-box">
  Esta herramienta ejecuta los motores (bayesianos y frecuentistas) programados por Pablo y muestra sus resultados de forma visual.
  Sube un archivo CSV con tus datos.
  </div>
  """, unsafe_allow_html=True)

  st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

  engine_key = st.session_state.get("selected_engine_key")

  with st.sidebar:
      st.markdown('<p class="sub-header">Motor seleccionado</p>', unsafe_allow_html=True)
      st.info(f"**{get_engine_label(engine_key)}**")

      if st.button("⬅️ Volver al inicio (Wizard)"):
          reset_wizard()
          st.rerun()

      st.markdown('<p class="sub-header">Opciones de ejecución</p>', unsafe_allow_html=True)

      if is_bayes_engine(engine_key):
          num_samples = st.number_input(
              "Número de muestras (simulación)",
              min_value=5000,
              max_value=200000,
              value=20000,
              step=5000,
              help="A más muestras, más estable el resultado (y más lento).",
          )
          generate_pdf = st.checkbox(
              "Generar PDF",
              value=False,
              help="Genera un reporte PDF con tablas y gráficos (si el motor lo soporta)."
          )
          include_ai = st.checkbox(
              "Interpretación IA (OpenAI)",
              value=False,
              help="Solo si tienes OPENAI_API_KEY configurada en el entorno."
          )

          st.session_state.num_samples = int(num_samples)
          st.session_state.generate_pdf = bool(generate_pdf)
          st.session_state.include_ai = bool(include_ai)

      elif is_freq_engine(engine_key):
          n_iteraciones = st.number_input(
              "Iteraciones (bootstrap)",
              min_value=1000,
              max_value=200000,
              value=10000,
              step=1000,
              help="A más iteraciones, más estable el resultado (y más lento).",
          )
          generate_pdf = st.checkbox(
              "Generar PDF",
              value=False,
              help="Genera un reporte PDF con tablas y gráficos (si el motor lo soporta)."
          )
          include_ai = st.checkbox(
              "Interpretación IA (OpenAI)",
              value=False,
              help="Solo si tienes OPENAI_API_KEY configurada en el entorno."
          )

          st.session_state.n_iteraciones = int(n_iteraciones)
          st.session_state.generate_pdf = bool(generate_pdf)
          st.session_state.include_ai = bool(include_ai)

      else:
          st.warning("No hay motor seleccionado. Vuelve al wizard.")

      if st.button("Reiniciar"):
          st.session_state.outputs = None
          st.session_state.datos_procesados = False
          st.success("Reiniciado correctamente")
          st.rerun()

  st.markdown('<div class="subsection-spacer"></div>', unsafe_allow_html=True)
  tab1, tab3 = st.tabs(["📊 Cargar CSV", "📋 Formato CSV"])

  with tab1:
      st.markdown('<p class="sub-header">Cargar datos desde CSV</p>', unsafe_allow_html=True)
      st.info("💡 Si no sabes cómo preparar tu archivo CSV, revisa la pestaña **'Formato CSV'**.")

      uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type=["csv"])

      if uploaded_file is not None:
          try:
              df = pd.read_csv(uploaded_file)

              st.success("✅ ¡Archivo cargado correctamente!")
              st.subheader("Vista previa de tus datos:")
              st.dataframe(df, use_container_width=True)

              if st.button("🚀 Procesar datos del CSV", type="primary"):
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
                              "n_iteraciones": st.session_state.get("n_iteraciones", 10000),
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

  with tab3:
      st.markdown('<p class="sub-header">Cómo preparar tu archivo CSV</p>', unsafe_allow_html=True)

      if is_bayes_engine(engine_key):
          st.markdown("""
- **Bayesiana [0,1] sin Session ID**: columnas `Día`, `Conversiones A`, `Visitas A`, `Conversiones B`, `Visitas B` (y más variantes si aplica).
- **Bayesiana [0,∞] sin Session ID**: mismo formato agregado, pero interpretado como conteos por visita.
- **Bayesiana con Session ID**: columnas `Día`, `SessionID`, y columnas de conversiones por variante (NaN cuando no aplica).
""")
      elif is_freq_engine(engine_key):
          if engine_key == ENGINE_FREQ_NO_SID:
              st.markdown("""
- **Frecuentista sin Session ID (agregado)**: columnas:
- `Día`, `Visitas A`, `Visitas B`, `Conversiones A`, `Conversiones B`
- El motor suma todo el periodo y calcula precisión + IC por bootstrap.
""")
          else:
              st.markdown("""
- **Frecuentista con Session ID (por sesión)**:
- El motor toma **las 2 primeras columnas** como A y B (según el script).
- Cada fila es una sesión/observación.
- Usa `NaN` cuando la sesión no pertenece a esa variante.
- Los valores pueden ser 0/1 o conteos (según lo que mida Pablo en ese script).
""")
      else:
          st.info("Selecciona un motor para ver su formato.")

  st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

  if st.session_state.get("datos_procesados", False):
      st.markdown("---")
      st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
      st.markdown('<h2 class="main-header">Resultados</h2>', unsafe_allow_html=True)

      out = st.session_state.get("outputs")
      if out is None:
          st.info("No hay outputs disponibles.")
          return

      if getattr(out, "summary", None) is not None:
          st.subheader("Resumen")
          summary_df = out.summary.copy()
          summary_df = summary_df[[c for c in summary_df.columns if not c.startswith("ci_")]]
          st.dataframe(summary_df, use_container_width=True)

          console_blocks = _format_console_blocks(out, engine_key)
          if console_blocks:
              st.subheader("Salida tipo consola")
              for b in console_blocks:
                  st.code(b)

      if getattr(out, "log_text", None):
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
  st.markdown("""
  <div style="text-align: center; background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 20px;">
    <p style="margin: 0; color: #555;">
      Idea y concepto: <strong>Claudia de la Cruz</strong>, <strong>Alex García</strong>
      &nbsp;|&nbsp; Desarrollo estadístico: <strong>Pablo González</strong>
      &nbsp;|&nbsp; Arquitectura de aplicación y UX: <strong>Eduardo Hernández</strong>
    </p>
  </div>
  """, unsafe_allow_html=True)

init_wizard_state()

if st.session_state.get("show_app", False):
  render_calculadora_actual()
else:
  render_wizard()