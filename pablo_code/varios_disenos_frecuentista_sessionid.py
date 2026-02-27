# -*- coding: utf-8 -*-
"""
Frecuentista (Bootstrap) con Session ID
Adaptación del notebook de Colab para integrarlo en el proyecto:

- Elimina dependencias de Colab/Drive y rutas hardcodeadas.
- Expone run(df, config) -> dict con: summary, figures, pdf_bytes, log_text
- Mantiene la lógica de bootstrap + gráficos + (opcional) interpretación IA.
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# OpenAI es opcional: solo se usa si include_ai=True y hay API key
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


def _safe_openai_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        return OpenAI()
    except Exception:
        return None


def interpretar_resultados_con_ia(resultados: Dict[str, Any]) -> str:
    """
    Interpretación ejecutiva centrada en Precisión y contraste B > A.
    Nota: en el Colab original se usaban prompts; aquí mantenemos el espíritu:
    - Solo corre si hay OPENAI_API_KEY y el usuario lo activa.
    """
    client = _safe_openai_client()
    if client is None:
        return ""

    # Extraemos datos (mismo contenido que el script original)
    g1, g2 = resultados["g1"], resultados["g2"]
    m1, m2 = resultados["media_real_g1"], resultados["media_real_g2"]

    uplift_pct = ((m2 - m1) / m1) * 100 if m1 != 0 else 0
    precision_b_gana = resultados["precision_b_mejor"] * 100
    ci_rel_low, ci_rel_high = resultados["ci_relativo"]

    resumen_datos = (
        f"TEST A/B: Control ({g1}) vs Variante ({g2})\n"
        f"MUESTRAS: {g1}: {resultados['n_g1']} filas | {g2}: {resultados['n_g2']} filas\n"
        f"CONVERSIONES: {g1}: {resultados['conv_g1']} | {g2}: {resultados['conv_g2']}\n"
        f"----------------------------------------------------\n"
        f"1. TASA DE CONVERSIÓN MEDIA:\n"
        f"   - {g1} (A): {m1:.4f}\n"
        f"   - {g2} (B): {m2:.4f}\n\n"
        f"2. UPLIFT (MEJORA) ESTIMADO:\n"
        f"   - La variante B mejora un {uplift_pct:.2f}% respecto al control A.\n\n"
        f"3. PRECISIÓN DE SUPERIORIDAD:\n"
        f"   - Precisión de que {g2} es superior a {g1}: {precision_b_gana:.2f}%\n\n"
        f"4. INTERVALO DE CONFIANZA DEL UPLIFT (95%):\n"
        f"   - Rango de precisión de la mejora: [{ci_rel_low:.2f}%, {ci_rel_high:.2f}%]"
    )

    prompt = f"""
Eres un Director de CRO. Analiza estos resultados de un test A/B.
IMPORTANTE: No uses la palabra "probabilidad", usa siempre "PRECISIÓN".

DATOS DEL TEST:
{resumen_datos}

TU MISIÓN:
Interpreta si B es mejor que A para un directivo.

REGLAS DE DECISIÓN:
1) Significancia: Si el intervalo de confianza cruza el 0%, no es concluyente.
2) Precisión: Si la precisión de superioridad es > 95%, es un ganador sólido.

ESTRUCTURA:
🎯 DICTAMEN
📊 ANÁLISIS DE RIESGO
🚀 ACCIÓN RECOMENDADA
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un experto en A/B testing que habla de precisión."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


class AnalisisBootstrap:
    def __init__(self, n_iteraciones: int = 10000):
        self.n_iter = int(n_iteraciones)
        self.resultados: Dict[str, Any] = {}
        self.distribuciones_medias: Dict[str, np.ndarray] = {}

    def analizar(self, datos_raw: Dict[str, np.ndarray]) -> None:
        grupos = list(datos_raw.keys())
        if len(grupos) < 2:
            raise ValueError("Se necesitan al menos 2 columnas/grupos para analizar (A y B).")

        g1_name, g2_name = grupos[0], grupos[1]

        data1 = np.array(datos_raw[g1_name], dtype=float)
        data2 = np.array(datos_raw[g2_name], dtype=float)

        n_g1, n_g2 = len(data1), len(data2)
        conv_g1, conv_g2 = np.sum(data1), np.sum(data2)

        print(f"🔄 Bootstrapping en curso ({self.n_iter} iteraciones)...")

        medias_g1 = np.zeros(self.n_iter)
        medias_g2 = np.zeros(self.n_iter)
        diferencias_ba = np.zeros(self.n_iter)  # B - A

        for i in range(self.n_iter):
            muestra_1 = np.random.choice(data1, size=n_g1, replace=True)
            muestra_2 = np.random.choice(data2, size=n_g2, replace=True)

            m1, m2 = np.mean(muestra_1), np.mean(muestra_2)
            medias_g1[i], medias_g2[i] = m1, m2
            diferencias_ba[i] = m2 - m1

        self.distribuciones_medias[g1_name] = medias_g1
        self.distribuciones_medias[g2_name] = medias_g2
        self.distribuciones_medias["diferencia"] = diferencias_ba

        precision_b_mejor = np.mean(diferencias_ba > 0)

        ci_low = np.percentile(diferencias_ba, 2.5)
        ci_high = np.percentile(diferencias_ba, 97.5)

        m1_obs = float(np.mean(data1))
        ci_rel_low = (ci_low / m1_obs) * 100 if m1_obs != 0 else 0
        ci_rel_high = (ci_high / m1_obs) * 100 if m1_obs != 0 else 0

        ganador = (
            g2_name
            if (precision_b_mejor > 0.95 and ci_low > 0)
            else (g1_name if (precision_b_mejor < 0.05 and ci_high < 0) else None)
        )

        self.resultados = {
            "g1": g1_name,
            "g2": g2_name,
            "n_g1": n_g1,
            "n_g2": n_g2,
            "conv_g1": conv_g1,
            "conv_g2": conv_g2,
            "media_real_g1": m1_obs,
            "media_real_g2": float(np.mean(data2)),
            "precision_b_mejor": float(precision_b_mejor),
            "ci_diferencia": (float(ci_low), float(ci_high)),
            "ci_relativo": (float(ci_rel_low), float(ci_rel_high)),
            "ganador": ganador,
        }

    def generar_reporte(self, pdf: Optional[PdfPages] = None) -> List[plt.Figure]:
        if not self.resultados:
            return []

        r = self.resultados
        figs: List[plt.Figure] = []

        # --- Página de texto (si PDF) ---
        if pdf is not None:
            fig_t = plt.figure(figsize=(8, 6))
            txt = (
                f"REPORTE DE PRECISIÓN BOOTSTRAP (SESSION ID)\n\n"
                f"{r['g1']} (Control): {r['n_g1']} filas, {int(r['conv_g1'])} conversiones\n"
                f"{r['g2']} (Variante): {r['n_g2']} filas, {int(r['conv_g2'])} conversiones\n\n"
                f"Tasa media A: {r['media_real_g1']:.4f}\n"
                f"Tasa media B: {r['media_real_g2']:.4f}\n\n"
                f"PRECISIÓN DE QUE B > A: {r['precision_b_mejor']*100:.2f}%\n"
                f"INTERVALO DE CONFIANZA (95%):\n"
                f"Límite Izquierdo: {r['ci_diferencia'][0]:.4f}\n"
                f"Límite Derecho:  {r['ci_diferencia'][1]:.4f}\n\n"
                f"RESULTADO: {r['ganador'] if r['ganador'] else 'Sin diferencia estadísticamente precisa'}"
            )
            fig_t.text(
                0.5,
                0.5,
                txt,
                family="monospace",
                fontsize=11,
                ha="center",
                va="center",
                bbox=dict(facecolor="none", edgecolor="black", pad=10),
            )
            pdf.savefig(fig_t)
            figs.append(fig_t)

        # --- FIG 1: distribuciones de medias ---
        fig1 = plt.figure(figsize=(10, 5))
        sns.kdeplot(self.distribuciones_medias[r["g1"]], fill=True, color="gray", label=f"A: {r['g1']}")
        sns.kdeplot(self.distribuciones_medias[r["g2"]], fill=True, color="blue", label=f"B: {r['g2']}")
        plt.title("Precisión de Distribución de Medias")
        plt.xlabel("Media")
        plt.legend()
        if pdf is not None:
            pdf.savefig(fig1)
        figs.append(fig1)

        # --- FIG 2: diferencia B - A ---
        fig2 = plt.figure(figsize=(10, 5))
        sns.histplot(self.distribuciones_medias["diferencia"], color="skyblue", kde=True)
        plt.axvline(0, color="red", linestyle="--", label="Punto de No Diferencia")

        ci_izq, ci_der = r["ci_diferencia"]
        plt.axvline(ci_izq, color="green", linestyle=":", label=f"Lím. Izq: {ci_izq:.4f}")
        plt.axvline(ci_der, color="green", linestyle=":", label=f"Lím. Der: {ci_der:.4f}")

        plt.title("Contraste de Precisión: Diferencia (B - A)")
        plt.legend()
        if pdf is not None:
            pdf.savefig(fig2)
        figs.append(fig2)

        return figs


def run(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Entrada:
      - df: DataFrame con los datos en formato “session-level”.
            Para mantener la compatibilidad con el notebook original:
            se toman las DOS PRIMERAS columnas como grupos A y B (valores numéricos),
            y se hace dropna() para formar las muestras.
      - config:
          - n_iteraciones (int) default 10000
          - generate_pdf (bool) default False
          - include_ai (bool) default False

    Salida:
      dict con keys: summary (DataFrame), figures (list[fig]), pdf_bytes (bytes|None), log_text (str|None)
    """
    config = config or {}
    n_iteraciones = int(config.get("n_iteraciones", 10000))
    generate_pdf = bool(config.get("generate_pdf", False))
    include_ai = bool(config.get("include_ai", False))

    # Mantener “lo de Pablo”: usar las 2 primeras columnas como grupos
    if df.shape[1] < 2:
        raise ValueError("El CSV debe tener al menos 2 columnas (grupo A y grupo B).")

    cols = list(df.columns[:2])
    datos = {c: df[c].dropna().values for c in cols}

    analisis = AnalisisBootstrap(n_iteraciones=n_iteraciones)
    analisis.analizar(datos)

    pdf_bytes: Optional[bytes] = None
    figs: List[plt.Figure] = []

    if generate_pdf:
        buffer = io.BytesIO()
        with PdfPages(buffer) as pdf:
            figs = analisis.generar_reporte(pdf)
            # Interpretación IA al final del PDF (si se activa)
            if include_ai:
                texto_ia = interpretar_resultados_con_ia(analisis.resultados)
                if texto_ia:
                    fig_ia = plt.figure(figsize=(8.27, 11.69))
                    fig_ia.clf()
                    fig_ia.text(0.05, 0.95, texto_ia, va="top", family="monospace", fontsize=10)
                    plt.axis("off")
                    pdf.savefig(fig_ia)
                    figs.append(fig_ia)

        pdf_bytes = buffer.getvalue()
        buffer.close()
    else:
        # Sin PDF: solo figuras en memoria
        figs = analisis.generar_reporte(pdf=None)

    # Summary (tabla)
    r = analisis.resultados
    uplift_pct = ((r["media_real_g2"] - r["media_real_g1"]) / r["media_real_g1"]) * 100 if r["media_real_g1"] else 0.0

    summary = pd.DataFrame(
        [
            {
                "grupo_A_col": r["g1"],
                "grupo_B_col": r["g2"],
                "n_A": r["n_g1"],
                "n_B": r["n_g2"],
                "conv_A": float(r["conv_g1"]),
                "conv_B": float(r["conv_g2"]),
                "media_A": float(r["media_real_g1"]),
                "media_B": float(r["media_real_g2"]),
                "uplift_%": float(uplift_pct),
                "precision_B_mejor": float(r["precision_b_mejor"]),
                "ci_diff_low": float(r["ci_diferencia"][0]),
                "ci_diff_high": float(r["ci_diferencia"][1]),
                "ci_uplift_%_low": float(r["ci_relativo"][0]),
                "ci_uplift_%_high": float(r["ci_relativo"][1]),
                "ganador": r["ganador"] or "",
            }
        ]
    )

    log_text = ""
    if include_ai:
        log_text = interpretar_resultados_con_ia(analisis.resultados)

    return {
        "summary": summary,
        "figures": figs,
        "pdf_bytes": pdf_bytes,
        "log_text": log_text,
    }