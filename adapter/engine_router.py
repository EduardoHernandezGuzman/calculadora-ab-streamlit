from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import pandas as pd

# Claves estables para las 4 rutas bayesianas
ENGINE_0_1_NO_SID = "bayes_0_1_no_sid"
ENGINE_0_1_SID = "bayes_0_1_sid"
ENGINE_0_INF_NO_SID = "bayes_0_inf_no_sid"
ENGINE_0_INF_SID = "bayes_0_inf_sid"

ENGINE_LABELS = {
    ENGINE_0_1_NO_SID: "Bayesiana [0,1] sin Session ID",
    ENGINE_0_1_SID: "Bayesiana [0,1] con Session ID",
    ENGINE_0_INF_NO_SID: "Bayesiana [0,∞] sin Session ID",
    ENGINE_0_INF_SID: "Bayesiana [0,∞] con Session ID",
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
    """
    Stub: aquí conectaremos los 4 scripts de Pablo SIN modificar su lógica.
    Cuando los tengamos en /pablo_code, este método llamará al script correcto.
    """
    raise NotImplementedError(
        "Motor no conectado aún. Sube los 4 scripts de Pablo a /pablo_code y conectamos run_engine()."
    )