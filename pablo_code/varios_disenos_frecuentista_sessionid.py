# -*- coding: utf-8 -*-
"""
Frecuentista (Bootstrap) con Session ID
Adaptación del notebook de Colab para integrarlo en el proyecto.

- Expone run(df, config) -> dict con: summary, figures, pdf_bytes, log_text
- Mantiene la lógica de bootstrap + gráficos + (opcional) interpretación IA.
"""

from __future__ import annotations

import io
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

try:
    from openai import OpenAI
except Exception:
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
    client = _safe_openai_client()
    if client is None:
        return ""

    g1, g2 = resultados["g1"], resultados["g2"]
    m1, m2 = resultados["media_real_g1"], resultados["media_real_g2"]

    uplift_pct = ((m2 - m1) / m1) * 100 if m1 != 0 else 0
    precision_b_gana = resultados["precision_b_mejor"] * 100
    ci_rel_low, ci_rel_high = resultados["ci_relativo_centrado"]
    cola_derecha_izq = resultados["ci_relativo_derecha_izq"]
    cola_izquierda_der = resultados["ci_relativo_izquierda_der"]

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
        f"3. NIVEL DE SIGNIFICANCIA DE QUE B > A:\n"
        f"   - {precision_b_gana:.2f}%\n\n"
        f"4. INTERVALOS DEL UPLIFT RELATIVO:\n"
        f"   - IC centrado: [{ci_rel_low:.2f}%, {ci_rel_high:.2f}%]\n"
        f"   - Cola derecha (IC 95% izquierda): > {cola_derecha_izq:.2f}%\n"
        f"   - Cola izquierda (IC 95% derecha): < {cola_izquierda_der:.2f}%"
    )

    prompt = f"""
Eres un Director de CRO. Analiza estos resultados de un test A/B.
IMPORTANTE: No uses la palabra "probabilidad", usa siempre "NIVEL DE SIGNIFICANCIA".

DATOS DEL TEST:
{resumen_datos}

TU MISIÓN:
Interpreta si B es mejor que A para un directivo.

REGLAS DE DECISIÓN:
1) Significancia: Si el intervalo de confianza cruza el 0%, no es concluyente.
2) Nivel de significancia: Si el nivel de significancia de superioridad es > 95%, es un ganador sólido.

ESTRUCTURA:
🎯 DICTAMEN
📊 ANÁLISIS DE RIESGO
🚀 ACCIÓN RECOMENDADA
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un experto en A/B testing que habla de nivel de significancia."},
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
        self.distribucion_uplift_rel: Optional[np.ndarray] = None

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
        diferencias_ba = np.zeros(self.n_iter)

        for i in range(self.n_iter):
            muestra_1 = np.random.choice(data1, size=n_g1, replace=True)
            muestra_2 = np.random.choice(data2, size=n_g2, replace=True)

            m1, m2 = np.mean(muestra_1), np.mean(muestra_2)
            medias_g1[i], medias_g2[i] = m1, m2
            diferencias_ba[i] = m2 - m1

        self.distribuciones_medias[g1_name] = medias_g1
        self.distribuciones_medias[g2_name] = medias_g2
        self.distribuciones_medias["diferencia"] = diferencias_ba

        precision_b_mejor = float(np.mean(diferencias_ba > 0))

        ci_low = float(np.percentile(diferencias_ba, 2.5))
        ci_high = float(np.percentile(diferencias_ba, 97.5))

        m1_obs = float(np.mean(data1))
        m2_obs = float(np.mean(data2))

        if m1_obs != 0:
            uplift_rel = (diferencias_ba / m1_obs) * 100
            ci_rel_centrado = np.percentile(uplift_rel, [2.5, 97.5]).astype(float)
            ci_rel_derecha_izq = float(np.percentile(uplift_rel, 5.0))
            ci_rel_izquierda_der = float(np.percentile(uplift_rel, 95.0))
        else:
            uplift_rel = np.zeros_like(diferencias_ba)
            ci_rel_centrado = np.array([0.0, 0.0], dtype=float)
            ci_rel_derecha_izq = 0.0
            ci_rel_izquierda_der = 0.0

        self.distribucion_uplift_rel = uplift_rel

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
            "media_real_g2": m2_obs,
            "precision_b_mejor": precision_b_mejor,
            "ci_diferencia": (ci_low, ci_high),
            "ci_relativo_centrado": (float(ci_rel_centrado[0]), float(ci_rel_centrado[1])),
            "ci_relativo_derecha_izq": float(ci_rel_derecha_izq),
            "ci_relativo_izquierda_der": float(ci_rel_izquierda_der),
            "ganador": ganador,
        }

    def generar_reporte(self, pdf: Optional[PdfPages] = None) -> List[plt.Figure]:
        if not self.resultados:
            return []

        r = self.resultados
        figs: List[plt.Figure] = []

        print("\n" + "=" * 40)
        print(f"{'MÉTRICAS DEL TEST':^40}")
        print("=" * 40)
        print(f"{r['g1']} (A): {r['n_g1']} filas | {int(r['conv_g1'])} convs | Media: {r['media_real_g1']:.4f}")
        print(f"{r['g2']} (B): {r['n_g2']} filas | {int(r['conv_g2'])} convs | Media: {r['media_real_g2']:.4f}")
        print("-" * 40)
        print(f"{'NIVEL DE SIGNIFICANCIA DE QUE B > A: ' + f'{r['precision_b_mejor']*100:.2f}%':^40}")
        print(f"{'IC CENTRADO (UPLIFT): ' + f'[{r['ci_relativo_centrado'][0]:.2f}%, {r['ci_relativo_centrado'][1]:.2f}%]':^40}")
        print(f"{'COLA DERECHA (IC 95% IZQUIERDA): ' + f'> {r['ci_relativo_derecha_izq']:.2f}%':^40}")
        print(f"{'COLA IZQUIERDA (IC 95% DERECHA): ' + f'< {r['ci_relativo_izquierda_der']:.2f}%':^40}")
        print("=" * 40)

        if pdf:
            fig_t = plt.figure(figsize=(8, 6))
            txt = (
                f"REPORTE DE NIVEL DE SIGNIFICANCIA BOOTSTRAP\n\n"
                f"{r['g1']} (Control): {r['n_g1']} filas, {int(r['conv_g1'])} conversiones\n"
                f"{r['g2']} (Variante): {r['n_g2']} filas, {int(r['conv_g2'])} conversiones\n\n"
                f"Tasa de conversión A: {r['media_real_g1']:.4f}\n"
                f"Tasa de conversión B: {r['media_real_g2']:.4f}\n\n"
                f"NIVEL DE SIGNIFICANCIA DE QUE B > A: {r['precision_b_mejor']*100:.2f}%\n"
                f"IC centrado: [{r['ci_relativo_centrado'][0]:.2f}%, {r['ci_relativo_centrado'][1]:.2f}%]\n"
                f"Cola derecha (IC 95% izquierda): > {r['ci_relativo_derecha_izq']:.2f}%\n"
                f"Cola izquierda (IC 95% derecha): < {r['ci_relativo_izquierda_der']:.2f}%\n\n"
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
            plt.close(fig_t)

        fig1 = plt.figure(figsize=(10, 5))
        sns.kdeplot(self.distribuciones_medias[r["g1"]], fill=True, color="gray", label=f"A: {r['g1']}")
        sns.kdeplot(self.distribuciones_medias[r["g2"]], fill=True, color="blue", label=f"B: {r['g2']}")
        plt.title("Precisión de Distribución de Medias")
        plt.xlabel("Conversión Media")
        plt.legend()
        if pdf is not None:
            pdf.savefig(fig1)
        figs.append(fig1)

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
    config = config or {}
    n_iteraciones = int(config.get("n_iteraciones", 10000))
    generate_pdf = bool(config.get("generate_pdf", False))
    include_ai = bool(config.get("include_ai", False))

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
        figs = analisis.generar_reporte(pdf=None)

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
                "ci_uplift_center_low": float(r["ci_relativo_centrado"][0]),
                "ci_uplift_center_high": float(r["ci_relativo_centrado"][1]),
                "ci_right_95_left": float(r["ci_relativo_derecha_izq"]),
                "ci_left_95_right": float(r["ci_relativo_izquierda_der"]),
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