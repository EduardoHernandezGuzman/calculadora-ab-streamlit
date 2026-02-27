from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import pandas as pd

# -------------------------
# Bayesian engines
# -------------------------
ENGINE_0_1_NO_SID = "bayes_0_1_no_sid"
ENGINE_0_1_SID = "bayes_0_1_sid"
ENGINE_0_INF_NO_SID = "bayes_0_inf_no_sid"
ENGINE_0_INF_SID = "bayes_0_inf_sid"

# -------------------------
# Frequentist engines
# -------------------------
ENGINE_FREQ_NO_SID = "freq_no_sid"
ENGINE_FREQ_SID = "freq_sid"

ENGINE_LABELS = {
    ENGINE_0_1_NO_SID: "Bayesiana [0,1] sin Session ID",
    ENGINE_0_1_SID: "Bayesiana [0,1] con Session ID",
    ENGINE_0_INF_NO_SID: "Bayesiana [0,∞] sin Session ID",
    ENGINE_0_INF_SID: "Bayesiana [0,∞] con Session ID",
    ENGINE_FREQ_NO_SID: "Frecuentista (Bootstrap) sin Session ID",
    ENGINE_FREQ_SID: "Frecuentista (Bootstrap) con Session ID",
}


@dataclass
class EngineOutput:
    summary: Optional[pd.DataFrame] = None
    figures: Optional[List[Any]] = None  # matplotlib figs, etc.
    pdf_bytes: Optional[bytes] = None
    log_text: Optional[str] = None


def get_engine_label(engine_key: Optional[str]) -> str:
    if not engine_key:
        return "—"
    return ENGINE_LABELS.get(engine_key, engine_key)


def run_engine(engine_key: str, df: pd.DataFrame, config: Dict[str, Any]) -> EngineOutput:
    # ---- Bayes [0,1] no SID
    if engine_key == ENGINE_0_1_NO_SID:
        from pablo_code import varios_disenos_0_1 as mod
        out = mod.run(df=df, config=config)
        return EngineOutput(
            summary=out.get("summary"),
            figures=out.get("figures"),
            pdf_bytes=out.get("pdf_bytes"),
            log_text=out.get("log_text"),
        )

    # ---- Bayes [0,∞] no SID
    if engine_key == ENGINE_0_INF_NO_SID:
        from pablo_code import varios_disenos_0_inf as mod
        out = mod.run(df=df, config=config)
        return EngineOutput(
            summary=out.get("summary"),
            figures=out.get("figures"),
            pdf_bytes=out.get("pdf_bytes"),
            log_text=out.get("log_text"),
        )

    # ---- Bayes [0,1] SID
    if engine_key == ENGINE_0_1_SID:
        from pablo_code import varios_disenos_sessionid_0_1 as mod
        out = mod.run(df=df, config=config)
        return EngineOutput(
            summary=out.get("summary"),
            figures=out.get("figures"),
            pdf_bytes=out.get("pdf_bytes"),
            log_text=out.get("log_text"),
        )

    # ---- Bayes [0,∞] SID
    if engine_key == ENGINE_0_INF_SID:
        from pablo_code import varios_disenos_sessionid_0_inf as mod
        out = mod.run(df=df, config=config)
        return EngineOutput(
            summary=out.get("summary"),
            figures=out.get("figures"),
            pdf_bytes=out.get("pdf_bytes"),
            log_text=out.get("log_text"),
        )

    # ---- Frequentist no SID
    if engine_key == ENGINE_FREQ_NO_SID:
        # IMPORTANTE: este nombre debe coincidir con tu archivo real
        # (por tu prueba en consola: from pablo_code import varios_diseno_frecuentista as m)
        from pablo_code import varios_diseno_frecuentista as mod
        out = mod.run(df=df, config=config)
        return EngineOutput(
            summary=out.get("summary"),
            figures=out.get("figures"),
            pdf_bytes=out.get("pdf_bytes"),
            log_text=out.get("log_text"),
        )

    # ---- Frequentist SID
    if engine_key == ENGINE_FREQ_SID:
        # Ajusta si tu archivo tiene otro nombre
        from pablo_code import varios_disenos_frecuentista_sessionid as mod
        out = mod.run(df=df, config=config)
        return EngineOutput(
            summary=out.get("summary"),
            figures=out.get("figures"),
            pdf_bytes=out.get("pdf_bytes"),
            log_text=out.get("log_text"),
        )

    raise NotImplementedError("Ese motor todavía no está conectado.")