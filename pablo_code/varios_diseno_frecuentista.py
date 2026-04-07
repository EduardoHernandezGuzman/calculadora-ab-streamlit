# -*- coding: utf-8 -*-
"""
Varios_diseno_frecuentista.py

Adaptación desde Colab a módulo reutilizable (Streamlit).
- Mantiene la lógica: Bootstrap sobre datos agregados + IC + gráfico + (opcional) IA Fisher.
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


def interpretar_resultados_con_ia(resultados: Dict[str, Any]) -> str:
    if OpenAI is None:
        return "❌ La librería 'openai' no está instalada en este entorno."

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    if not api_key:
        return "❌ OPENAI_API_KEY no está configurada en secrets o entorno."

    client = OpenAI(api_key=api_key)
    ASSISTANT_ID = "asst_XBKGUebN9P5Zzt81kVmzMLxt"

    g1, g2 = "Control (A)", "Variante (B)"
    m1, m2 = resultados["media_real_g1"], resultados["media_real_g2"]

    uplift_pct = ((m2 - m1) / m1) * 100 if m1 != 0 else 0
    precision_b_gana = resultados["precision_b_mejor"] * 100
    ci_rel_low, ci_rel_high = resultados["ci_relativo_centrado"]
    cola_derecha_izq = resultados["ci_relativo_derecha_izq"]
    cola_izquierda_der = resultados["ci_relativo_izquierda_der"]

    resumen_datos = (
        f"DATOS DEL TEST PARA ANALIZAR:\n"
        f"TEST A/B: {g1} vs {g2}\n"
        f"MUESTRAS (Visitas): A: {resultados['n_g1']} | B: {resultados['n_g2']}\n"
        f"CONVERSIONES TOTALES: A: {int(resultados['conv_g1'])} | B: {int(resultados['conv_g2'])}\n"
        f"----------------------------------------------------\n"
        f"1. TASA DE CONVERSIÓN MEDIA:\n"
        f"   - {g1}: {m1:.4%}\n"
        f"   - {g2}: {m2:.4%}\n\n"
        f"2. UPLIFT (MEJORA) ESTIMADO:\n"
        f"   - La variante B mejora un {uplift_pct:.2f}% respecto al control A.\n\n"
        f"3. NIVEL DE SIGNIFICANCIA DE QUE B > A:\n"
        f"   - {precision_b_gana:.2f}%\n\n"
        f"4. INTERVALOS DEL UPLIFT RELATIVO:\n"
        f"   - IC centrado: [{ci_rel_low:.2f}%, {ci_rel_high:.2f}%]\n"
        f"   - Cola derecha (IC 95% izquierda): > {cola_derecha_izq:.2f}%\n"
        f"   - Cola izquierda (IC 95% derecha): < {cola_izquierda_der:.2f}%"
    )

    try:
        run = client.beta.threads.create_and_run_poll(
            assistant_id=ASSISTANT_ID,
            thread={"messages": [{"role": "user", "content": resumen_datos}]},
        )

        if run.status == "completed":
            mensajes = client.beta.threads.messages.list(thread_id=run.thread_id)
            return mensajes.data[0].content[0].text.value

        return f"❌ Error en la ejecución del agente Fisher. Estado final: {run.status}"

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
            "ci_relativo_centrado": (float(ci_rel_centrado[0]), float(ci_rel_centrado[1])),
            "ci_relativo_derecha_izq": float(ci_rel_derecha_izq),
            "ci_relativo_izquierda_der": float(ci_rel_izquierda_der),
        }

    def generar_reporte(self, pdf: Optional[PdfPages] = None):
        r = self.resultados

        print("\n" + "=" * 50)
        print(f"{'ANÁLISIS DE PRECISIÓN B vs A':^50}")
        print("=" * 50)
        print(f"{'Diseño A':<20} | Visitas: {r['n_g1']:>8} | Convs: {int(r['conv_g1']):>6}")
        print(f"{'Diseño B':<20} | Visitas: {r['n_g2']:>8} | Convs: {int(r['conv_g2']):>6}")
        print("-" * 50)
        print(f"{'NIVEL DE SIGNIFICANCIA DE QUE B > A: ' + f'{r['precision_b_mejor']*100:.2f}%':^50}")
        print(f"{'IC CENTRADO (UPLIFT): ' + f'[{r['ci_relativo_centrado'][0]:.2f}%, {r['ci_relativo_centrado'][1]:.2f}%]':^50}")
        print(f"{'COLA DERECHA (IC 95% IZQUIERDA): ' + f'> {r['ci_relativo_derecha_izq']:.2f}%':^50}")
        print(f"{'COLA IZQUIERDA (IC 95% DERECHA): ' + f'< {r['ci_relativo_izquierda_der']:.2f}%':^50}")
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
                f"NIVEL DE SIGNIFICANCIA DE QUE B > A: {r['precision_b_mejor']*100:.2f}%\n\n"
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
        sns.histplot(self.distribuciones_medias["diferencia"], color="skyblue", kde=True, element="step")
        plt.axvline(0, color="red", linestyle="--", label="Sin diferencia")
        plt.axvline(r["ci_diferencia"][0], color="green", linestyle=":", label=f"Lím. Izq: {r['ci_diferencia'][0]:.4f}")
        plt.axvline(r["ci_diferencia"][1], color="green", linestyle=":", label=f"Lím. Der: {r['ci_diferencia'][1]:.4f}")
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
    uplift_pct = ((r["media_real_g2"] - r["media_real_g1"]) / r["media_real_g1"] * 100) if r["media_real_g1"] != 0 else 0.0

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