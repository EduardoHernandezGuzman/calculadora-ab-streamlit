# -*- coding: utf-8 -*-
"""
Motor Bayesiano [0,1] CON Session ID (datos por sesión, agregación por día).

Entradas esperadas (mínimo):
- Columna "Día"
- Columnas "Conversiones A", "Conversiones B", ... (valores 0/1 o NaN)
  - NaN = sesión sin exposición / sin dato para ese grupo
  - count() cuenta "visitas" (sesiones con dato)
  - sum() cuenta "conversiones" (sesiones con valor 1)
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from io import BytesIO
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", "Glyph .* missing from font")
sns.set(style="whitegrid")


def _interpretar_con_ia(resultados_ultimo_dia: Dict[str, Any]) -> str:
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI()
    except Exception as e:
        return f"[IA] No disponible (no se pudo importar OpenAI): {e}"

    resumen_grupos: List[str] = []
    comparativas: List[str] = []

    for k, v in resultados_ultimo_dia.items():
        if isinstance(v, dict) and "media" in v:
            visitas_acum = resultados_ultimo_dia.get(f"acum_visitas_{k}", "N/A")
            conv_acum = resultados_ultimo_dia.get(f"acum_clicks_{k}", "N/A")
            resumen_grupos.append(
                f"Grupo {k}: Visitas Acumuladas={visitas_acum}, Conversiones Acumuladas={conv_acum}, "
                f"Tasa Media={v['media']:.4f}, IC95%=[{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]"
            )

        if "_vs_" in k and isinstance(v, dict):
            ic_centrado = f"[{v['ci_centered'][0]*100:.2f}%, {v['ci_centered'][1]*100:.2f}%]"
            ic_suelo = f"> {v['ci_right'][0]*100:.2f}%"
            ic_techo = f"< {v['ci_left'][1]*100:.2f}%"
            comparativas.append(
                f"COMPARATIVA {k}:\n"
                f"- Probabilidad de que el primero sea mejor: {v['prob_mejor']*100:.2f}%\n"
                f"- Uplift Medio Estimado: {v['uplift_media']*100:.2f}%\n"
                f"- IC CENTRADO 95%: {ic_centrado}\n"
                f"- IC SUELO: {ic_suelo}\n"
                f"- IC TECHO: {ic_techo}"
            )

    prompt = f"""
Eres un experto Senior en Estadística Bayesiana y Experimentación (A/B Testing). Tu trabajo es interpretar los resultados y dar una recomendación de negocio.

Instrucciones:
- Contempla objetivo de MAXIMIZAR o MINIMIZAR la métrica (da ambas interpretaciones).
- Regla del cero: si el intervalo de uplift incluye 0% => no hay diferencia concluyente.
- Gestión de riesgo: traduce el peor caso (suelo/techo) a lenguaje de negocio.
- Recomendación: detener el test o continuar.

DATOS DEL ÚLTIMO DÍA:
{chr(10).join(resumen_grupos)}

COMPARATIVAS:
{chr(10).join(comparativas)}

Lenguaje claro, ejecutivo y sin fórmulas.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un analista senior de CRO."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[IA] Error llamando a OpenAI: {e}"


class ConversionBayesBeta:
    def __init__(self, priors: Dict[str, Tuple[float, float]]):
        self.priors = priors.copy()
        self.historial: List[Dict[str, Any]] = []
        self.acumulados = defaultdict(lambda: {"clicks": 0.0, "visitas": 0.0})

    def actualizar_con_datos(
        self,
        datos: Dict[str, Tuple[float, float]],
        raw_data: Optional[pd.DataFrame] = None,
        dia: Optional[str] = None,
        num_samples: int = 100000,
    ) -> None:
        resultados: Dict[str, Any] = {
            "dia": dia or f"Día {len(self.historial)+1}",
            "raw_data": raw_data,
        }

        muestras: Dict[str, np.ndarray] = {}
        grupos = list(datos.keys())

        for grupo in grupos:
            alpha0, beta0 = self.priors.get(grupo, (1.0, 1.0))
            visitas_dia, clicks_dia = datos[grupo]

            self.acumulados[grupo]["clicks"] += float(clicks_dia)
            self.acumulados[grupo]["visitas"] += float(visitas_dia)

            total_visitas = self.acumulados[grupo]["visitas"]
            total_clicks = self.acumulados[grupo]["clicks"]
            total_fracasos = total_visitas - total_clicks

            alpha_post = alpha0 + total_clicks
            beta_post = beta0 + total_fracasos

            muestras_array = np.random.beta(a=alpha_post, b=beta_post, size=int(num_samples)).astype(np.float64)
            muestras[grupo] = muestras_array

            mean = float(np.mean(muestras_array))
            std = float(np.std(muestras_array))
            ci = np.percentile(muestras_array, [2.5, 97.5]).astype(np.float64)

            resultados[f"acum_visitas_{grupo}"] = int(total_visitas)
            resultados[f"acum_clicks_{grupo}"] = int(total_clicks)

            resultados[grupo] = {
                "media": mean,
                "std": std,
                "ci": ci,
                "muestras": muestras_array,
            }

        for g1, g2 in combinations(grupos, 2):
            for (a, b) in [(g1, g2), (g2, g1)]:
                tasa_a = muestras[a]
                tasa_b = muestras[b]

                uplift_samples = np.where(tasa_b != 0, (tasa_a - tasa_b) / tasa_b, 0.0)
                diff_samples = tasa_a - tasa_b

                prob_mejor = float(np.mean(diff_samples > 0))
                mean_uplift = float(np.mean(uplift_samples))
                std_uplift = float(np.std(uplift_samples))

                ci_centered = np.percentile(uplift_samples, [2.5, 97.5]).astype(np.float64)
                ci_right = np.percentile(uplift_samples, [5.0, 100.0]).astype(np.float64)
                ci_left = np.percentile(uplift_samples, [0.0, 95.0]).astype(np.float64)

                resultados[f"{a}_vs_{b}"] = {
                    "uplift_media": mean_uplift,
                    "uplift_std": std_uplift,
                    "ci_centered": ci_centered,
                    "ci_right": ci_right,
                    "ci_left": ci_left,
                    "prob_mejor": prob_mejor,
                    "ganador": a if prob_mejor >= 0.95 else None,
                    "diff": diff_samples,
                }

        self.historial.append(resultados)


def _fig_histograma_raw(dia: str, raw_data: pd.DataFrame) -> Optional[plt.Figure]:
    if raw_data is None or raw_data.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    if "Conversiones A" in raw_data.columns:
        s = raw_data["Conversiones A"].dropna()
        if len(s) > 0:
            sns.histplot(
                s, label="Grupo A", kde=False, element="step", alpha=0.4, discrete=True, ax=ax
            )
            plotted = True

    if "Conversiones B" in raw_data.columns:
        s = raw_data["Conversiones B"].dropna()
        if len(s) > 0:
            sns.histplot(
                s, label="Grupo B", kde=False, element="step", alpha=0.4, discrete=True, ax=ax
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(f"{dia} - Distribución Real (0=No conv, 1=Conv)")
    ax.set_xlabel("Valor de Conversión")
    ax.set_ylabel("Frecuencia (Sesiones)")
    ax.legend()
    return fig


def _fig_posteriors_beta(dia: str, paso: Dict[str, Any], grupos: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for grupo in grupos:
        muestras = paso[grupo]["muestras"]
        sns.kdeplot(muestras, label=f"Grupo {grupo}", fill=True, ax=ax)
    ax.set_title(f"{dia} - Incertidumbre Tasa Conversión (Modelo Beta)")
    ax.set_xlabel("Tasa de Conversión Estimada")
    ax.legend()
    return fig


def _fig_diff(dia: str, stats: Dict[str, Any], g1: str, g2: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    diff = stats["diff"]
    sns.histplot(diff, stat="density", element="step", alpha=0.3, label="Histograma", ax=ax)
    sns.kdeplot(diff, fill=True, alpha=0.4, label="Densidad", ax=ax)
    ax.axvline(0, linestyle="--", label="Ref (0)")
    ax.set_title(f"{dia} - Diferencia Absoluta ({g1} - {g2})")
    ax.set_xlabel("Diferencia en Tasa de Conversión")
    ax.legend()
    return fig


def _build_beta_priors(expected_priors: Optional[Dict[str, Tuple[float, float]]], grupos: List[str]) -> Dict[str, Tuple[float, float]]:
    if not expected_priors:
        return {g: (1.0, 1.0) for g in grupos}

    priors: Dict[str, Tuple[float, float]] = {}
    for g in grupos:
        conv, visitas = expected_priors.get(g, (0.0, 0.0))
        alpha0 = float(conv) + 1.0
        beta0 = float(visitas - conv) + 1.0 if visitas >= conv else 1.0
        priors[g] = (alpha0, beta0)

    return priors


def _infer_grupos_from_columns(df: pd.DataFrame) -> List[str]:
    grupos: List[str] = []
    for c in df.columns:
        if isinstance(c, str) and c.startswith("Conversiones "):
            grupos.append(c.replace("Conversiones ", "").strip())
    return sorted(list(dict.fromkeys(grupos)))


def _aggregate_by_day_sessionid(df: pd.DataFrame, grupos: List[str]) -> List[Tuple[Any, pd.DataFrame, Dict[str, Tuple[int, int]]]]:
    if "Día" not in df.columns:
        raise ValueError("Falta la columna 'Día' en el CSV.")

    dias_unicos = sorted(df["Día"].dropna().unique())
    out: List[Tuple[Any, pd.DataFrame, Dict[str, Tuple[int, int]]]] = []

    for dia_val in dias_unicos:
        df_dia = df[df["Día"] == dia_val].copy()
        datos_agregados: Dict[str, Tuple[int, int]] = {}

        for g in grupos:
            col = f"Conversiones {g}"
            if col not in df_dia.columns:
                raise ValueError(f"Falta la columna '{col}' en el CSV.")
            visitas = int(df_dia[col].count())
            conv = float(df_dia[col].sum(skipna=True))
            datos_agregados[g] = (visitas, int(conv))

        out.append((dia_val, df_dia, datos_agregados))

    return out


def run(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = config or {}
    num_samples = int(config.get("num_samples", 20000))
    generate_pdf = bool(config.get("generate_pdf", False))
    include_ai = bool(config.get("include_ai", False))
    expected_priors = config.get("expected_priors")

    grupos = _infer_grupos_from_columns(df)
    if not grupos:
        raise ValueError("No se detectaron columnas 'Conversiones X'. Ej: 'Conversiones A', 'Conversiones B'.")

    priors = _build_beta_priors(expected_priors, grupos)
    modelo = ConversionBayesBeta(priors=priors)
    agregados = _aggregate_by_day_sessionid(df, grupos)

    figures: List[plt.Figure] = []
    log_parts: List[str] = []
    summary_rows: List[Dict[str, Any]] = []

    for dia_val, df_dia_raw, datos_agregados in agregados:
        dia_label = f"Día {int(dia_val)}" if str(dia_val).isdigit() else f"Día {dia_val}"
        modelo.actualizar_con_datos(
            datos=datos_agregados,
            raw_data=df_dia_raw,
            dia=dia_label,
            num_samples=num_samples,
        )

        paso = modelo.historial[-1]
        grupos_stats = [g for g in paso if isinstance(paso.get(g), dict) and "media" in paso[g]]

        for g in grupos_stats:
            total_visitas = int(paso.get(f"acum_visitas_{g}", 0))
            total_conv = int(paso.get(f"acum_clicks_{g}", 0))

            visitas_dia, conv_dia = datos_agregados[g]
            tasa_obs = (conv_dia / visitas_dia) if visitas_dia > 0 else 0.0

            summary_rows.append({
                "dia": dia_label,
                "grupo": g,
                "media": float(paso[g]["media"]),
                "ci_low": float(paso[g]["ci"][0]),
                "ci_high": float(paso[g]["ci"][1]),
                "visitas": int(visitas_dia),
                "conversiones": int(conv_dia),
                "tasa_observada": float(tasa_obs),
                "acum_visitas": total_visitas,
                "acum_conversiones": total_conv,
            })

        f0 = _fig_histograma_raw(dia_label, df_dia_raw)
        if f0 is not None:
            figures.append(f0)

        figures.append(_fig_posteriors_beta(dia_label, paso, grupos_stats))

        comparaciones = [k for k in paso.keys() if isinstance(k, str) and "_vs_" in k]
        for clave in comparaciones:
            stats = paso[clave]
            g1, g2 = clave.split("_vs_")
            figures.append(_fig_diff(dia_label, stats, g1, g2))

    summary_df = pd.DataFrame(summary_rows)

    pdf_bytes: Optional[bytes] = None
    if generate_pdf:
        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            for fig in figures:
                pdf.savefig(fig)
        buffer.seek(0)
        pdf_bytes = buffer.read()

    if include_ai:
        if modelo.historial:
            log_parts.append("🤖 Interpretación IA (último día):")
            log_parts.append(_interpretar_con_ia(modelo.historial[-1]))
        else:
            log_parts.append("[IA] No hay historial para interpretar.")

    return {
        "summary": summary_df,
        "figures": figures,
        "pdf_bytes": pdf_bytes,
        "log_text": "\n\n".join(log_parts) if log_parts else "",
        "comparisons": modelo.historial,
    }