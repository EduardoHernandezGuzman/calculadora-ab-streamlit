# -*- coding: utf-8 -*-
"""
Motor Bayesiano [0,1] SIN session_id (Beta-Binomial) - Multi-grupo

Refactor mínimo para:
- No ejecutar nada al importar (apto para Streamlit)
- Exponer run(df, config) para que la capa visual lo llame
- Mantener ejecución "standalone" vía CLI si se desea
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# pymc estaba importado en el original pero no se usa en este archivo
# lo dejo por compatibilidad si Pablo lo necesita después
try:
    import pymc as pm  # noqa: F401
except Exception:
    pm = None  # type: ignore

warnings.filterwarnings("ignore", "Glyph .* missing from font")
sns.set(style="whitegrid")


# -------------------------
# OpenAI (opcional)
# -------------------------
def interpretar_con_ia(resultados: Dict[str, Any]) -> str:
    """
    Interpretación ejecutiva con OpenAI.
    Si no hay OPENAI_API_KEY en el entorno, devuelve mensaje informativo.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "Interpretación IA no configurada (falta OPENAI_API_KEY)."

    try:
        from openai import OpenAI
    except Exception:
        return "Interpretación IA no disponible (paquete openai no instalado)."

    client = OpenAI(api_key=api_key)

    resumen_grupos = []
    comparativas = []

    for k, v in resultados.items():
        if isinstance(v, dict) and 'media' in v:
            visitas_acum = resultados.get(f"acum_visitas_{k}", "N/A")
            conv_acum = resultados.get(f"acum_clicks_{k}", "N/A")
            resumen_grupos.append(
                f"Grupo {k}: "
                f"Visitas Acumuladas={visitas_acum}, "
                f"Conversiones Acumuladas={conv_acum}, "
                f"Tasa Media={v['media']:.4f}, "
                f"IC95% (Tasa)=[{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]"
            )

        if "_vs_" in k and isinstance(v, dict):
            ic_centrado = f"[{v['ci_centered'][0]*100:.2f}%, {v['ci_centered'][1]*100:.2f}%]"
            ic_suelo = f"> {v['ci_right'][0]*100:.2f}%"
            ic_techo = f"< {v['ci_left'][1]*100:.2f}%"
            comparativas.append(
                f"COMPARATIVA {k}: \n"
                f"   - Probabilidad de que el primero sea mejor: {v['prob_mejor']*100:.2f}%\n"
                f"   - Uplift Medio Estimado: {v['uplift_media']*100:.2f}%\n"
                f"   - Intervalo de Credibilidad (CENTRADO 95%): {ic_centrado}\n"
                f"   - Intervalo Unilateral (SUELO): {ic_suelo}\n"
                f"   - Intervalo Unilateral (TECHO): {ic_techo}"
            )

    prompt = f"""
Eres un experto Senior en Estadística Bayesiana y Experimentación (A/B Testing). Tu trabajo es interpretar los resultados de un experimento y dar una recomendación de negocio.

TUS INSTRUCCIONES CLAVE:

Contexto del Objetivo: Antes de decidir un ganador, contempla que el objetivo puede ser el de MAXIMIZAR la métrica o MINIMIZARLA. Ofrece las dos interpretaciones

Si Minimizar: Buscamos 'Probabilidad A > B' > 95%.
Si Maximizar: Buscamos 'Probabilidad B > A' > 95%.

LA REGLA DEL CERO (Crucial):
Analiza el Intervalo de Credibilidad (IC) del Uplift.
Si el intervalo incluye el 0%: NO hay diferencia concluyente.

Gestión de Riesgo: Traduce el límite 'peor caso' (Suelo/Techo) del intervalo a lenguaje de negocio.

Recomendación: ¿Detener el test o continuar?

DATOS COMPLETOS DEL ÚLTIMO DÍA:
---------------------
{chr(10).join(resumen_grupos)}

COMPARATIVAS DETALLADAS:
{chr(10).join(comparativas)}
---------------------

Lenguaje claro, ejecutivo y sin fórmulas.
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un analista senior de CRO."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# -------------------------
# Núcleo bayesiano (igual que Pablo)
# -------------------------
class ConversionBayesMultiGrupo:
    def __init__(self, priors: Dict[str, Tuple[float, float]]):
        self.priors = priors.copy()
        self.historial: List[Dict[str, Any]] = []
        self.acumulados = {grupo: {'visitas': 0, 'conv': 0} for grupo in priors}

    def actualizar_con_datos(self, datos: Dict[str, Tuple[int, int]], dia: Optional[str] = None, num_samples: int = 100000):
        resultados: Dict[str, Any] = {'dia': dia or f"Día {len(self.historial)+1}"}
        muestras: Dict[str, np.ndarray] = {}
        grupos = list(datos.keys())
        tasas_conversion: Dict[str, float] = {}

        for grupo in grupos:
            alpha0, beta0 = self.priors.get(grupo, (1, 1))
            visitas, conv = datos[grupo]  # (trials, successes)
            alpha_post = alpha0 + conv
            beta_post = beta0 + visitas - conv

            muestras_array = np.random.beta(alpha_post, beta_post, num_samples).astype(np.float64)
            muestras[grupo] = muestras_array

            mean = alpha_post / (alpha_post + beta_post)
            std = np.sqrt((alpha_post * beta_post) / (((alpha_post + beta_post) ** 2) * (alpha_post + beta_post + 1)))
            ci = np.percentile(muestras_array, [2.5, 97.5])

            resultados[grupo] = {
                'media': mean,
                'std': std,
                'ci': ci,
                'muestras': muestras_array
            }

            tasa = conv / visitas if visitas > 0 else 0
            tasas_conversion[grupo] = tasa
            resultados[f"visitas_{grupo}"] = visitas
            resultados[f"conv_{grupo}"] = conv
            resultados[f"tasa_{grupo}"] = tasa

            self.priors[grupo] = (alpha_post, beta_post)
            if grupo not in self.acumulados:
                self.acumulados[grupo] = {'conv': 0, 'visitas': 0}
            self.acumulados[grupo]['conv'] += conv
            self.acumulados[grupo]['visitas'] += visitas

        if len(grupos) >= 2:
            a, b = grupos[0], grupos[1]
            total_a = self.acumulados[a]
            total_b = self.acumulados[b]
            tasa_a1 = total_a['conv'] / total_a['visitas'] if total_a['visitas'] > 0 else 0
            tasa_b1 = total_b['conv'] / total_b['visitas'] if total_b['visitas'] > 0 else 0
            uplift1 = (tasa_b1 - tasa_a1) / tasa_a1 if tasa_a1 > 0 else None
            resultados['uplift1'] = uplift1

        for g1, g2 in combinations(grupos, 2):
            for (a, b) in [(g1, g2), (g2, g1)]:
                tasa_a = muestras[a]
                tasa_b = muestras[b]
                uplift = (tasa_a - tasa_b) / tasa_b
                diff = tasa_a - tasa_b
                prob_mejor = np.mean(diff > 0)
                mean_uplift = np.mean(uplift)
                std_uplift = np.std(uplift)
                ci_uplift = np.percentile(uplift, [2.5, 97.5])

                resultados[f"{a}_vs_{b}"] = {
                    'uplift_media': mean_uplift,
                    'uplift_std': std_uplift,
                    'uplift_ci': ci_uplift,
                    'prob_mejor': prob_mejor,
                    'ganador': a if prob_mejor >= 0.95 else None,
                    'diff': diff,
                    'uplift1': resultados.get('uplift1')
                }

        self.historial.append(resultados)

    def mostrar_resultados_con_graficos(self, pdf: Optional[PdfPages] = None, show: bool = False) -> List[plt.Figure]:
        """
        Devuelve una lista de figuras. En modo Streamlit usaremos show=False y pintaremos con st.pyplot().
        """
        figs: List[plt.Figure] = []

        for paso in self.historial:
            dia = paso['dia']
            try:
                dia_num = int(str(dia).split()[1])
            except Exception:
                dia_num = None

            lines = [f"🗓️  {dia}"]
            grupos = [g for g in paso if isinstance(paso[g], dict) and 'media' in paso[g]]
            for grupo in grupos:
                stats = paso[grupo]
                lines += [
                    f"Grupo {grupo}:",
                    f"  Media esperada: {stats['media']*100:.2f}%",
                    f"  Desviación estándar: {stats['std']*100:.2f}%",
                    f"  IC 95%: [{stats['ci'][0]*100:.2f}%, {stats['ci'][1]*100:.2f}%]"
                ]

            # Página texto (PDF)
            if pdf is not None:
                fig_text = plt.figure(figsize=(8.27, 11.69))
                fig_text.clf()
                fig_text.text(0.01, 0.99, "\n".join(lines), va='top', family='monospace')
                pdf.savefig(fig_text)
                plt.close(fig_text)

            # Fig distribuciones
            fig1 = plt.figure(figsize=(10, 5))
            for grupo in grupos:
                muestras = paso[grupo]['muestras']
                sns.kdeplot(muestras, label=f"Grupo {grupo}", fill=True)
            plt.title(f"{dia} - Distribuciones posteriores (Beta)")
            plt.xlabel("Tasa de conversión")
            plt.legend()

            if pdf is not None:
                pdf.savefig(fig1)
            figs.append(fig1)

            if show:
                plt.show()
            plt.close(fig1)

            # Comparaciones
            comparaciones = [k for k in paso if "_vs_" in k]
            for clave in comparaciones:
                stats = paso[clave]
                g1, g2 = clave.split("_vs_")

                comp_lines = [
                    f"\n📈 Uplift (relativo {g1} vs {g2}):",
                    f"  Media: {stats['uplift1']*100:.2f}%" if stats.get("uplift1") is not None else "  Media: N/A",
                    f"  Desviación estándar: {stats['uplift_std']*100:.2f}%",
                    f"  IC 95%: [{stats['uplift_ci'][0]*100:.2f}%, {stats['uplift_ci'][1]*100:.2f}%]",
                    f"\n📊 Probabilidad de que {g1} > {g2}: {stats['prob_mejor']*100:.2f}%"
                ]

                if stats.get('ganador'):
                    if dia_num is not None and dia_num < 6:
                        comp_lines += [
                            "\n⚠️ *Atención:*",
                            "Aún no has alcanzado el día 6.",
                            "Este resultado podría cambiar con más datos."
                        ]
                    else:
                        comp_lines += [
                            "\n🏆 Resultado final:",
                            f"  Ganador: {g1}",
                            f"  Decisión: Implementar {g1}",
                            f"  Razón: {g1} es mejor con {stats['prob_mejor']*100:.2f}% de probabilidad"
                        ]

                if pdf is not None:
                    fig_cmp = plt.figure(figsize=(8.27, 11.69))
                    fig_cmp.clf()
                    fig_cmp.text(0.01, 0.99, "\n".join(comp_lines), va='top', family='monospace')
                    pdf.savefig(fig_cmp)
                    plt.close(fig_cmp)

                fig2 = plt.figure(figsize=(10, 4))
                sns.kdeplot(stats['diff'], label=f"Diferencia ({g1} - {g2})", fill=True)
                plt.axvline(0, color="black", linestyle="--")
                plt.title(f"{dia} - Diferencia de conversión: {g1} - {g2}")
                plt.xlabel("Diferencia")
                plt.legend()

                if pdf is not None:
                    pdf.savefig(fig2)
                figs.append(fig2)

                if show:
                    plt.show()
                plt.close(fig2)

        return figs


# -------------------------
# Helpers de entrada/salida
# -------------------------
def _build_priors_from_expected(expected: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    priors: Dict[str, Tuple[int, int]] = {}
    for grupo, (conversiones, visitas) in expected.items():
        if not isinstance(conversiones, int) or not isinstance(visitas, int):
            continue
        if conversiones < 0 or visitas < 0:
            continue
        if conversiones > visitas:
            continue
        alfa = conversiones + 1
        beta = (visitas - conversiones) + 1
        priors[grupo] = (alfa, beta)
    return priors


def _extract_daily_data_from_aggregate_row(row: pd.Series) -> Dict[str, Tuple[int, int]]:
    """
    Convierte una fila agregada (un día) en datos por grupo:
      "Conversiones X" + "Visitas X"  ->  (visitas, conversiones)

    Importante: en el original, el script asumía columnas "Conversiones " y "Visitas ".
    """
    datos: Dict[str, Tuple[int, int]] = {}
    for col in row.index:
        if str(col).startswith("Conversiones "):
            grupo = str(col).replace("Conversiones ", "")
            visitas_col = f"Visitas {grupo}"
            if visitas_col in row.index:
                conv = int(row[col])
                visitas = int(row[visitas_col])
                datos[grupo] = (visitas, conv)
    return datos


def _build_summary_from_historial(historial: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Crea un dataframe resumen por día y por grupo con medias e intervalos.
    """
    records = []
    for paso in historial:
        dia = paso.get("dia")
        grupos = [g for g in paso if isinstance(paso[g], dict) and 'media' in paso[g]]
        for g in grupos:
            stats = paso[g]
            records.append({
                "dia": dia,
                "grupo": g,
                "media": stats["media"],
                "ci_low": float(stats["ci"][0]),
                "ci_high": float(stats["ci"][1]),
                "visitas": paso.get(f"visitas_{g}"),
                "conversiones": paso.get(f"conv_{g}"),
                "tasa_observada": paso.get(f"tasa_{g}"),
            })
    return pd.DataFrame.from_records(records)


# -------------------------
# API principal para Streamlit
# -------------------------
def run(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ejecuta el análisis [0,1] sin sessionid (agregado por día).

    config (opcional):
      - expected_priors: dict grupo -> (conv_esperadas, visitas_esperadas)
      - num_samples: int (default 100000)
      - generate_pdf: bool (default False)
      - include_ai: bool (default False)
    """
    config = config or {}
    num_samples = int(config.get("num_samples", 100000))
    generate_pdf = bool(config.get("generate_pdf", False))
    include_ai = bool(config.get("include_ai", False))

    expected_priors = config.get("expected_priors")
    if not isinstance(expected_priors, dict) or not expected_priors:
        # default "suave" (equivalente al original donde ponían A,B,C = (1,1))
        # aquí expected es (conv, visitas)
        expected_priors = {"A": (1, 1), "B": (1, 1)}

    priors = _build_priors_from_expected(expected_priors)
    if not priors:
        priors = {"A": (1, 1), "B": (1, 1)}

    modelo = ConversionBayesMultiGrupo(priors)

    # ejecutar por día (df agregado)
    for _, row in df.iterrows():
        dia = f"Día {int(row['Día'])}" if "Día" in row.index else None
        datos = _extract_daily_data_from_aggregate_row(row)
        if datos:
            modelo.actualizar_con_datos(datos, dia=dia, num_samples=num_samples)

    # Generar PDF en memoria si se pide
    pdf_bytes: Optional[bytes] = None
    figs: List[plt.Figure] = []
    if generate_pdf:
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            figs = modelo.mostrar_resultados_con_graficos(pdf=pdf, show=False)
        pdf_bytes = buf.getvalue()
    else:
        figs = modelo.mostrar_resultados_con_graficos(pdf=None, show=False)

    summary = _build_summary_from_historial(modelo.historial)

    # IA opcional: usar último paso del historial “en bruto”
    ai_text: Optional[str] = None
    if include_ai and modelo.historial:
        try:
            ai_text = interpretar_con_ia(modelo.historial[-1])
        except Exception as e:
            ai_text = f"No se pudo generar interpretación IA: {e}"

    return {
        "summary": summary,
        "figures": figs,
        "pdf_bytes": pdf_bytes,
        "log_text": ai_text,
    }


# -------------------------
# CLI (mantener modo "script")
# -------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Motor Bayesiano [0,1] sin session_id (agregado por día)")
    parser.add_argument("--csv", required=True, help="Ruta al CSV de entrada")
    parser.add_argument("--pdf", default="", help="Ruta de salida PDF (opcional)")
    parser.add_argument("--samples", type=int, default=100000, help="Número de muestras (default: 100000)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    cfg = {
        "num_samples": args.samples,
        "generate_pdf": bool(args.pdf),
        "expected_priors": {"A": (1, 1), "B": (1, 1), "C": (1, 1)},
    }

    out = run(df, cfg)

    if args.pdf and out.get("pdf_bytes"):
        with open(args.pdf, "wb") as f:
            f.write(out["pdf_bytes"])
        print(f"✅ PDF guardado en: {args.pdf}")

    print("✅ Ejecutado. Resumen:")
    print(out["summary"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()