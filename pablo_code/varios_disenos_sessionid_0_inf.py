# varios_disenos_sessionid_0_inf.py

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from collections import defaultdict
import io
import warnings
import os

warnings.filterwarnings("ignore", "Glyph .* missing from font")
sns.set(style="whitegrid")


# ===============================
# LÓGICA BAYESIANA (GAMMA)
# ===============================

class ConversionBayesGamma:
    def __init__(self, priors):
        self.priors = priors.copy()
        self.historial = []
        self.acumulados = defaultdict(lambda: {'clicks': 0, 'visitas': 0})

    def actualizar_con_datos(self, datos, raw_data=None, dia=None, num_samples=100000):
        resultados = {
            'dia': dia,
            'raw_data': raw_data
        }
        muestras = {}
        grupos = list(datos.keys())

        # 1. Muestreo Gamma
        for grupo in grupos:
            alpha0, beta0 = self.priors.get(grupo, (1, 1))
            visitas, clicks = datos[grupo]

            alpha_post = alpha0 + clicks
            beta_post = beta0 + visitas

            muestras_array = np.random.gamma(
                shape=alpha_post,
                scale=1 / beta_post,
                size=num_samples
            )

            muestras[grupo] = muestras_array

            mean = np.mean(muestras_array)
            ci = np.percentile(muestras_array, [2.5, 97.5])

            self.priors[grupo] = (alpha_post, beta_post)

            self.acumulados[grupo]['clicks'] += clicks
            self.acumulados[grupo]['visitas'] += visitas

            resultados[f"acum_visitas_{grupo}"] = self.acumulados[grupo]['visitas']
            resultados[f"acum_clicks_{grupo}"] = self.acumulados[grupo]['clicks']

            resultados[grupo] = {
                'media': mean,
                'ci': ci,
                'muestras': muestras_array
            }

        # 2. Comparaciones
        for g1, g2 in combinations(grupos, 2):
            for (a, b) in [(g1, g2), (g2, g1)]:
                tasa_a = muestras[a]
                tasa_b = muestras[b]

                uplift_samples = np.where(
                    tasa_b != 0,
                    (tasa_a - tasa_b) / tasa_b,
                    0
                )
                diff_samples = tasa_a - tasa_b

                prob_mejor = np.mean(diff_samples > 0)

                ci_centered = np.percentile(uplift_samples, [2.5, 97.5])
                ci_right = np.percentile(uplift_samples, [5.0, 100.0])
                ci_left = np.percentile(uplift_samples, [0.0, 95.0])

                resultados[f"{a}_vs_{b}"] = {
                    'uplift_media': np.mean(uplift_samples),
                    'ci_centered': ci_centered,
                    'ci_right': ci_right,
                    'ci_left': ci_left,
                    'prob_mejor': prob_mejor,
                    'ganador': a if prob_mejor >= 0.95 else None,
                    'diff': diff_samples
                }

        self.historial.append(resultados)


# ===============================
# FUNCIÓN PRINCIPAL
# ===============================

def run(df: pd.DataFrame, config: dict):
    """
    Ejecuta el motor Bayesiano [0,∞] con Session ID.
    """

    num_samples = config.get("num_samples", 20000)
    generate_pdf = config.get("generate_pdf", False)
    include_ai = config.get("include_ai", False)

    priors = {'A': (1, 1), 'B': (1, 1)}
    modelo = ConversionBayesGamma(priors=priors)

    dias_unicos = sorted(df['Día'].unique())

    for dia_val in dias_unicos:
        df_dia = df[df['Día'] == dia_val].copy()

        visitas_a = df_dia['Conversiones A'].count()
        visitas_b = df_dia['Conversiones B'].count()

        conv_a = df_dia['Conversiones A'].sum()
        conv_b = df_dia['Conversiones B'].sum()

        datos = {
            'A': (visitas_a, conv_a),
            'B': (visitas_b, conv_b)
        }

        modelo.actualizar_con_datos(
            datos=datos,
            raw_data=df_dia,
            dia=f"Día {int(dia_val)}",
            num_samples=num_samples
        )

    summary_rows = []
    figures = []
    log_text = ""
    pdf_bytes = None

    for paso in modelo.historial:
        dia = paso['dia']
        grupos = [g for g in paso if isinstance(paso[g], dict) and 'media' in paso[g]]

        for grupo in grupos:
            stats = paso[grupo]
            visitas = paso.get(f"acum_visitas_{grupo}", 0)
            conversiones = paso.get(f"acum_clicks_{grupo}", 0)

            tasa_obs = conversiones / visitas if visitas > 0 else 0

            summary_rows.append({
                "dia": dia,
                "grupo": grupo,
                "media": stats['media'],
                "ci_low": stats['ci'][0],
                "ci_high": stats['ci'][1],
                "visitas": visitas,
                "conversiones": conversiones,
                "tasa_observada": tasa_obs,
                "acum_visitas": visitas,
                "acum_conversiones": conversiones,
            })

        # Gráficos
        plt.figure(figsize=(8, 4))
        for grupo in grupos:
            sns.kdeplot(paso[grupo]['muestras'], fill=True, label=f"Grupo {grupo}")
        plt.title(f"{dia} - Posterior Gamma")
        plt.legend()
        figures.append(plt.gcf())
        plt.close()

    summary_df = pd.DataFrame(summary_rows)

    if generate_pdf:
        buffer = io.BytesIO()
        with PdfPages(buffer) as pdf:
            for fig in figures:
                pdf.savefig(fig)
        buffer.seek(0)
        pdf_bytes = buffer.read()

    return {
        "summary": summary_df,
        "figures": figures,
        "pdf_bytes": pdf_bytes,
        "log_text": log_text
    }