"""Carga del modelo desde su artefacto."""

from pathlib import Path
from typing import Any
import sklearn
import joblib
from src.aguaapr.constants import FEATURES

from sklearn.pipeline import Pipeline
from src.aguaapr.artifacts import leer_metadata, ErrorArtefacto

def cargar_modelo(directorio: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Carga el artefacto de `directorio` y devuelve (modelo, metadata).

    Antes de cargar el modelo comprueba que las variables de la metadata sean
    exactamente `FEATURES`, en ese orden, y que `sklearn_version` coincida con
    la versión de scikit-learn instalada. Si algo no calza lanza
    `ErrorArtefacto` con un mensaje que nombra la diferencia.
    """
    directorio = Path(directorio)
    metadata = leer_metadata(directorio)

    if metadata["features"] != list(FEATURES):
        raise ErrorArtefacto(
            "Las variables de la metadata no coinciden con FEATURES."
        )

    if metadata["sklearn_version"] != sklearn.__version__:
        raise ErrorArtefacto(
            f"Versión de scikit-learn incompatible: "
            f"{metadata['sklearn_version']} != {sklearn.__version__}"
        )

    modelo = joblib.load(directorio / "model.joblib")

    return modelo, metadata
