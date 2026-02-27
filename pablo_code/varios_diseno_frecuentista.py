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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# OpenAI es opcional (solo si include_ai=True)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# Silenciar warnings de glifos faltantes (igual que el original)
import warnings
warnings.filterwarnings("ignore", "Glyph .* missing from font")

sns.set(style="whitegrid")


# ==========================================
# 1. FUNCIÓN DE INTERPRETACIÓN IA (FISHER)
# ==========================================
def interpretar_resultados_con_ia(resultados: Dict[str, Any]) -> str:
    """
    Interpretación ejecutiva usando Asistente Fisher (Frecuentista).
    Mantiene el flujo original, pero ahora:
      - usa OPENAI_API_KEY del entorno
      - devuelve texto (no print como salida principal)
    """
    if OpenAI is None:
        return "❌ La librería 'openai' no está instalada en este entorno."

    if not os.getenv("OPENAI_API_KEY"):
        return "❌ OPENAI_API_KEY no está configurada en el entorno."

    client = OpenAI()

    # ID del Asistente Fisher (se mantiene igual)
    ASSISTANT_ID = "asst_XBKGUebN9P5Zzt81kVmzMLxt"

    g1, g2 = "Control (A)", "Variante (B)"
    m1, m2 = resultados["media_real_g1"], resultados["media_real_g2"]

    uplift_pct = ((m2 - m1) / m1) * 100 if m1 != 0 else 0
    precision_b_gana = resultados["precision_b_mejor"] * 100
    ci_rel_low, ci_rel_high = resultados["ci_relativo"]

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
        f"3. PRECISIÓN DE SUPERIORIDAD:\n"
        f"   - Precisión de que B es superior a A: {precision_b_gana:.2f}%\n\n"
        f"4. INTERVALO DE CONFIANZA DEL UPLIFT (95%):\n"
        f"   - Rango de precisión de la mejora: [{ci_rel_low:.2f}%, {ci_rel_high:.2f}%]"
    )

    try:
        # Mismo patrón que el original: create_and_run_poll
        run = client.beta.threads.create_and_run_poll(
            assistant_id=ASSISTANT_ID,
            thread={"messages": [{"role": "user", "content": resumen_datos}]},
        )

        if run.status == "completed":
            mensajes = client.beta.threads.messages.list(thread_id=run.thread_id)
            respuesta = mensajes.data[0].content[0].text.value
            return respuesta

        return f"❌ Error en la ejecución del agente Fisher. Estado final: {run.status}"

    except Exception as e:
        return f"❌ Nota: Error de conexión con la API de OpenAI: {e}"


# ==========================================
# 2. CLASE DE ANÁLISIS BOOTSTRAP (MISMA LÓGICA)
# ==========================================
class AnalisisBootstrapAgregado:
    def __init__(self, n_iteraciones: int = 10000):
        self.n_iter = n_iteraciones
        self.resultados: Dict[str, Any] = {}
        self.distribuciones_medias: Dict[str, np.ndarray] = {}

    def analizar(self, n_a: int, conv_a: int, n_b: int, conv_b: int):
        # Reconstruimos los datos (0s y 1s) para que el Bootstrap pueda remuestrear
        data_a = np.array([1] * int(conv_a) + [0] * int(n_a - conv_a))
        data_b = np.array([1] * int(conv_b) + [0] * int(n_b - conv_b))

        print(f"🔄 Iniciando Bootstrap con {self.n_iter} iteraciones...")

        medias_a = np.zeros(self.n_iter)
        medias_b = np.zeros(self.n_iter)
        diferencias_ba = np.zeros(self.n_iter)  # Contraste B - A

        for i in range(self.n_iter):
            m_a = np.mean(np.random.choice(data_a, size=len(data_a), replace=True))
            m_b = np.mean(np.random.choice(data_b, size=len(data_b), replace=True))

            medias_a[i] = m_a
            medias_b[i] = m_b
            diferencias_ba[i] = m_b - m_a

        self.distribuciones_medias["A"] = medias_a
        self.distribuciones_medias["B"] = medias_b
        self.distribuciones_medias["diferencia"] = diferencias_ba

        precision_b_mejor = np.mean(diferencias_ba > 0)

        ci_low = np.percentile(diferencias_ba, 2.5)
        ci_high = np.percentile(diferencias_ba, 97.5)

        m_a_obs = conv_a / n_a if n_a != 0 else 0.0
        ci_rel_low = (ci_low / m_a_obs) * 100 if m_a_obs != 0 else 0.0
        ci_rel_high = (ci_high / m_a_obs) * 100 if m_a_obs != 0 else 0.0

        self.resultados = {
            "n_g1": int(n_a),
            "n_g2": int(n_b),
            "conv_g1": int(conv_a),
            "conv_g2": int(conv_b),
            "media_real_g1": float(m_a_obs),
            "media_real_g2": float(conv_b / n_b) if n_b != 0 else 0.0,
            "precision_b_mejor": float(precision_b_mejor),
            "ci_diferencia": (float(ci_low), float(ci_high)),
            "ci_relativo": (float(ci_rel_low), float(ci_rel_high)),
        }

    def generar_reporte(self, pdf: Optional[PdfPages] = None):
        r = self.resultados

        # Consola (igual que el original)
        print("\n" + "=" * 50)
        print(f"{'ANÁLISIS DE PRECISIÓN B vs A':^50}")
        print("=" * 50)
        print(f"{'Diseño A':<20} | Visitas: {r['n_g1']:>8} | Convs: {int(r['conv_g1']):>6}")
        print(f"{'Diseño B':<20} | Visitas: {r['n_g2']:>8} | Convs: {int(r['conv_g2']):>6}")
        print("-" * 50)
        print(f"{'Precisión de que B > A: ' + f'{r['precision_b_mejor']*100:.2f}%':^50}")
        print(f"{'IC 95% Izquierda: ' + f'{r['ci_diferencia'][0]:.5f}':^50}")
        print(f"{'IC 95% Derecha:   ' + f'{r['ci_diferencia'][1]:.5f}':^50}")
        print("=" * 50)

        # Página de texto en PDF (igual idea)
        if pdf:
            fig_t = plt.figure(figsize=(8, 6))
            txt = (
                f"REPORTE DE PRECISIÓN (DATOS AGREGADOS)\n\n"
                f"Métricas de Control (A):\n"
                f"Visitas: {r['n_g1']} | Conversiones: {int(r['conv_g1'])}.\n\n"
                f"Métricas de Variante (B):\n"
                f"Visitas: {r['n_g2']} | Conversiones: {int(r['conv_g2'])}.\n\n"
                f"--------------------------------------------\n"
                f"Tasa Conv. A: {r['media_real_g1']:.4%}\n"
                f"Tasa Conv. B: {r['media_real_g2']:.4%}\n\n"
                f"PRECISIÓN DE QUE B > A: {r['precision_b_mejor']*100:.2f}%\n\n"
                f"INTERVALO DE CONFIANZA (95%):\n"
                f"Límite Izquierdo: {r['ci_diferencia'][0]:.5f}\n"
                f"Límite Derecho: {r['ci_diferencia'][1]:.5f}\n"
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

        # Gráfico de Diferencia
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

        # OJO: no hacemos plt.show() aquí en módulo; devolvemos la figura para Streamlit
        return fig


# ==========================================
# 3. FUNCIÓN RUN (entrada/salida para Streamlit)
# ==========================================
def run(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ejecuta el análisis frecuentista (sin Session ID) con datos agregados.

    Espera columnas (como el original):
      - 'Visitas A', 'Visitas B', 'Conversiones A', 'Conversiones B'
    Suma todo el periodo (como el original) y lanza bootstrap.

    config:
      - n_iteraciones: int (default 10000)
      - generate_pdf: bool (default False)
      - include_ai: bool (default False)
    """
    config = config or {}
    n_iteraciones = int(config.get("n_iteraciones", 10000))
    generate_pdf = bool(config.get("generate_pdf", False))
    include_ai = bool(config.get("include_ai", False))

    required = ["Visitas A", "Visitas B", "Conversiones A", "Conversiones B"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {missing}")

    # Sumarizar para obtener el total del periodo (igual que el original)
    total_v_a = int(df["Visitas A"].sum())
    total_v_b = int(df["Visitas B"].sum())
    total_c_a = int(df["Conversiones A"].sum())
    total_c_b = int(df["Conversiones B"].sum())

    analisis = AnalisisBootstrapAgregado(n_iteraciones=n_iteraciones)
    analisis.analizar(total_v_a, total_c_a, total_v_b, total_c_b)

    # Generar figura (y opcional PDF)
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

    # IA Fisher (opcional)
    log_text = ""
    if include_ai:
        log_text = interpretar_resultados_con_ia(analisis.resultados)

    # Summary para UI
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
                "ci_uplift_%_low": r["ci_relativo"][0],
                "ci_uplift_%_high": r["ci_relativo"][1],
            }
        ]
    )

    return {
        "summary": summary,
        "figures": figures,
        "pdf_bytes": pdf_bytes,
        "log_text": log_text,
    }