# -*- coding: utf-8 -*-
"""
Varios_diseno_frecuentista.py

Adaptación desde Colab a módulo reutilizable (Streamlit).
- Mantiene la lógica: Bootstrap sobre datos agregados + IC + gráfico + IA opcional.
- Elimina dependencias de Google Drive / rutas fijas.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

import warnings

warnings.filterwarnings("ignore", "Glyph .* missing from font")
sns.set(style="whitegrid")


def _get_openai_api_key() -> str:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY", "").strip()


def interpretar_resultados_con_ia(resultados: Dict[str, Any]) -> str:
    if OpenAI is None:
        return "❌ La librería 'openai' no está instalada en este entorno."

    api_key = _get_openai_api_key()
    if not api_key:
        return "❌ OPENAI_API_KEY no está configurada en secrets o entorno."

    client = OpenAI(api_key=api_key)

    g1, g2 = "Control (A)", "Variante (B)"
    m1, m2 = resultados["media_real_g1"], resultados["media_real_g2"]

    uplift_pct = ((m2 - m1) / m1) * 100 if m1 != 0 else 0
    precision_b_mejor = resultados["precision_b_mejor"] * 100
    ci_rel_low, ci_rel_high = resultados["ci_relativo_centrado"]
    cola_derecha_izq = resultados["ci_relativo_derecha_izq"]
    cola_izquierda_der = resultados["ci_relativo_izquierda_der"]

    prompt = f"""
Eres un Director de CRO. Analiza los resultados de un test A/B y proporciona una recomendación clara de negocio basada en inferencia frecuentista.

IMPORTANTE:
No uses la palabra "probabilidad". Usa siempre "NIVEL DE SIGNIFICANCIA".
Lenguaje claro, ejecutivo y sin fórmulas.

DATOS DEL TEST:

Grupo Control (A):
Visitas Acumuladas = {resultados["n_g1"]}
Conversiones Acumuladas = {int(resultados["conv_g1"])}
Tasa Media = {m1:.4f}

Grupo Variante (B):
Visitas Acumuladas = {resultados["n_g2"]}
Conversiones Acumuladas = {int(resultados["conv_g2"])}
Tasa Media = {m2:.4f}

TASA DE CONVERSIÓN MEDIA:
{g1}: {m1:.4f}
{g2}: {m2:.4f}

UPLIFT (MEJORA) ESTIMADO:
La variante B mejora un {uplift_pct:.2f}% respecto al control A.

NIVEL DE SIGNIFICANCIA DE QUE B > A:
{precision_b_mejor:.2f}%

INTERVALOS DEL UPLIFT RELATIVO:
IC centrado 95%: [{ci_rel_low:.2f}%, {ci_rel_high:.2f}%]
Límite inferior (escenario conservador): > {cola_derecha_izq:.2f}%
Límite superior (escenario optimista): < {cola_izquierda_der:.2f}%

REGLAS DE DECISIÓN:

Significancia estadística (regla del cero):
Si el intervalo de confianza incluye el 0% → el resultado no es concluyente.

Nivel de significancia:
Si el nivel de significancia de superioridad es > 95% → considerar ganador sólido.

Gestión de riesgo:
Traduce el peor y mejor escenario del intervalo a impacto real de negocio.

ESTRUCTURA DE RESPUESTA:

DICTAMEN
Conclusión clara: ¿B gana, pierde o no hay evidencia suficiente?

ANÁLISIS DE RIESGO
Qué puede pasar en el peor y mejor escenario (impacto negocio).

ACCIÓN RECOMENDADA
¿Implementar variante, mantener control o seguir testeando?
Justificación breve y directa.
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un Director de CRO experto en experimentación y análisis frecuentista.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"❌ Nota: Error de conexión con la API de OpenAI: {e}"


class AnalisisBootstrapAgregado:
    def __init__(self, n_iteraciones: int = 10000):
        self.n_iter = int(n_iteraciones)
        self.resultados: Dict[str, Any] = {}
        self.distribuciones_medias: Dict[str, np.ndarray] = {}
        self.distribuciones_uplift_rel: Optional[np.ndarray] = None

    def analizar(self, n_a: int, conv_a: int, n_b: int, conv_b: int):
        data_a = np.array([1] * int(conv_a) + [0] * int(n_a - conv_a))
        data_b = np.array([1] * int(conv_b) + [0] * int(n_b - conv_b))

        print(f"🔄 Iniciando Bootstrap con {self.n_iter} iteraciones...")

        medias_a = np.zeros(self.n_iter)
        medias_b = np.zeros(self.n_iter)
        diferencias_ba = np.zeros(self.n_iter)

        for i in range(self.n_iter):
            m_a = np.mean(np.random.choice(data_a, size=len(data_a), replace=True))
            m_b = np.mean(np.random.choice(data_b, size=len(data_b), replace=True))

            medias_a[i] = m_a
            medias_b[i] = m_b
            diferencias_ba[i] = m_b - m_a

        self.distribuciones_medias["A"] = medias_a
        self.distribuciones_medias["B"] = medias_b
        self.distribuciones_medias["diferencia"] = diferencias_ba

        precision_b_mejor = float(np.mean(diferencias_ba > 0))

        ci_low = float(np.percentile(diferencias_ba, 2.5))
        ci_high = float(np.percentile(diferencias_ba, 97.5))

        m_a_obs = conv_a / n_a if n_a != 0 else 0.0
        m_b_obs = conv_b / n_b if n_b != 0 else 0.0

        if m_a_obs != 0:
            uplift_rel = (diferencias_ba / m_a_obs) * 100
            ci_rel_centrado = np.percentile(uplift_rel, [2.5, 97.5]).astype(float)
            ci_rel_derecha_izq = float(np.percentile(uplift_rel, 5.0))
            ci_rel_izquierda_der = float(np.percentile(uplift_rel, 95.0))
        else:
            uplift_rel = np.zeros_like(diferencias_ba)
            ci_rel_centrado = np.array([0.0, 0.0], dtype=float)
            ci_rel_derecha_izq = 0.0
            ci_rel_izquierda_der = 0.0

        self.distribuciones_uplift_rel = uplift_rel

        self.resultados = {
            "n_g1": int(n_a),
            "n_g2": int(n_b),
            "conv_g1": int(conv_a),
            "conv_g2": int(conv_b),
            "media_real_g1": float(m_a_obs),
            "media_real_g2": float(m_b_obs),
            "precision_b_mejor": precision_b_mejor,
            "ci_diferencia": (ci_low, ci_high),
            "ci_relativo_centrado": (
                float(ci_rel_centrado[0]),
                float(ci_rel_centrado[1]),
            ),
            "ci_relativo_derecha_izq": float(ci_rel_derecha_izq),
            "ci_relativo_izquierda_der": float(ci_rel_izquierda_der),
        }

    def generar_reporte(self, pdf: Optional[PdfPages] = None):
        r = self.resultados

        print("\n" + "=" * 50)
        print(f"{'ANÁLISIS DE PRECISIÓN B vs A':^50}")
        print("=" * 50)
        print(
            f"{'Diseño A':<20} | Visitas: {r['n_g1']:>8} | Convs: {int(r['conv_g1']):>6}"
        )
        print(
            f"{'Diseño B':<20} | Visitas: {r['n_g2']:>8} | Convs: {int(r['conv_g2']):>6}"
        )
        print("-" * 50)
        print(
            f"NIVEL DE SIGNIFICANCIA DE QUE B > A: {r['precision_b_mejor'] * 100:.2f}%"
        )
        print(
            f"IC CENTRADO (UPLIFT): [{r['ci_relativo_centrado'][0]:.2f}%, {r['ci_relativo_centrado'][1]:.2f}%]"
        )
        print(
            f"COLA DERECHA (IC 95% IZQUIERDA): > {r['ci_relativo_derecha_izq']:.2f}%"
        )
        print(
            f"COLA IZQUIERDA (IC 95% DERECHA): < {r['ci_relativo_izquierda_der']:.2f}%"
        )
        print("=" * 50)

        if pdf:
            fig_t = plt.figure(figsize=(8, 6))
            txt = (
                f"REPORTE DE NIVEL DE SIGNIFICANCIA (DATOS AGREGADOS)\n\n"
                f"Métricas de Control (A):\n"
                f"Visitas: {r['n_g1']} | Conversiones: {int(r['conv_g1'])}.\n\n"
                f"Métricas de Variante (B):\n"
                f"Visitas: {r['n_g2']} | Conversiones: {int(r['conv_g2'])}.\n\n"
                f"--------------------------------------------\n"
                f"Tasa Conv. A: {r['media_real_g1']:.4%}\n"
                f"Tasa Conv. B: {r['media_real_g2']:.4%}\n\n"
                f"NIVEL DE SIGNIFICANCIA DE QUE B > A: {r['precision_b_mejor'] * 100:.2f}%\n\n"
                f"INTERVALOS DEL UPLIFT RELATIVO:\n"
                f"IC Centrado: [{r['ci_relativo_centrado'][0]:.2f}%, {r['ci_relativo_centrado'][1]:.2f}%]\n"
                f"Cola derecha (IC 95% izquierda): > {r['ci_relativo_derecha_izq']:.2f}%\n"
                f"Cola izquierda (IC 95% derecha): < {r['ci_relativo_izquierda_der']:.2f}%\n"
            )
            fig_t.text(
                0.5,
                0.5,
                txt,
                family="monospace",
                fontsize=11,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black", pad=15),
            )
            pdf.savefig(fig_t)
            plt.close(fig_t)

        fig = plt.figure(figsize=(10, 6))
        sns.histplot(
            self.distribuciones_medias["diferencia"],
            color="skyblue",
            kde=True,
            element="step",
        )
        plt.axvline(0, color="red", linestyle="--", label="Sin diferencia")
        plt.axvline(
            r["ci_diferencia"][0],
            color="green",
            linestyle=":",
            label=f"Lím. Izq: {r['ci_diferencia'][0]:.4f}",
        )
        plt.axvline(
            r["ci_diferencia"][1],
            color="green",
            linestyle=":",
            label=f"Lím. Der: {r['ci_diferencia'][1]:.4f}",
        )
        plt.title("Precisión del Uplift: Distribución de la diferencia (B - A)")
        plt.xlabel("Diferencia de Tasas de Conversión")
        plt.legend(loc="upper right")

        if pdf:
            pdf.savefig(fig)

        return fig


def run(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = config or {}
    n_iteraciones = int(config.get("n_iteraciones", 10000))
    generate_pdf = bool(config.get("generate_pdf", False))
    include_ai = bool(config.get("include_ai", False))

    required = ["Visitas A", "Visitas B", "Conversiones A", "Conversiones B"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {missing}")

    total_v_a = int(df["Visitas A"].sum())
    total_v_b = int(df["Visitas B"].sum())
    total_c_a = int(df["Conversiones A"].sum())
    total_c_b = int(df["Conversiones B"].sum())

    analisis = AnalisisBootstrapAgregado(n_iteraciones=n_iteraciones)
    analisis.analizar(total_v_a, total_c_a, total_v_b, total_c_b)

    figures: List[Any] = []
    pdf_bytes: Optional[bytes] = None

    if generate_pdf:
        bio = io.BytesIO()
        with PdfPages(bio) as pdf:
            fig_diff = analisis.generar_reporte(pdf)
            if fig_diff is not None:
                figures.append(fig_diff)
        pdf_bytes = bio.getvalue()
    else:
        fig_diff = analisis.generar_reporte(pdf=None)
        if fig_diff is not None:
            figures.append(fig_diff)

    log_text = ""
    if include_ai:
        log_text = interpretar_resultados_con_ia(analisis.resultados)

    r = analisis.resultados
    uplift_pct = (
        ((r["media_real_g2"] - r["media_real_g1"]) / r["media_real_g1"] * 100)
        if r["media_real_g1"] != 0
        else 0.0
    )

    summary = pd.DataFrame(
        [
            {
                "n_visitas_A": r["n_g1"],
                "n_visitas_B": r["n_g2"],
                "conv_A": r["conv_g1"],
                "conv_B": r["conv_g2"],
                "tasa_A": r["media_real_g1"],
                "tasa_B": r["media_real_g2"],
                "uplift_%": uplift_pct,
                "precision_B_mejor": r["precision_b_mejor"],
                "ci_diff_low": r["ci_diferencia"][0],
                "ci_diff_high": r["ci_diferencia"][1],
                "ci_uplift_center_low": r["ci_relativo_centrado"][0],
                "ci_uplift_center_high": r["ci_relativo_centrado"][1],
                "ci_right_95_left": r["ci_relativo_derecha_izq"],
                "ci_left_95_right": r["ci_relativo_izquierda_der"],
            }
        ]
    )

    return {
        "summary": summary,
        "figures": figures,
        "pdf_bytes": pdf_bytes,
        "log_text": log_text,
    }