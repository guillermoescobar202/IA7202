"""Rutas de inferencia: información del modelo, vida y predicción."""

from fastapi import APIRouter, Depends

from src.aguaapr.artifacts import a_tabla, predecir
from src.aguaapr.constants import FEATURES, RANGOS
from src.aguaapr.contracts import RespuestaPrediccion, Servicios
from src.aguaapr.deps import crear_dependencia_clave
from src.aguaapr.validation import MedicionAgua


def crear_router_inferencia(servicios: Servicios) -> APIRouter:
    """Construye el router con tres rutas.

    - `GET /`: describe el modelo. Responde las métricas de validación de la
      metadata, la entrada (cada columna con su rango) y la
      salida.
    - `GET /health`: ruta de vida, pública. Responde
      `{"estado": "ok"}`.
    - `POST /predict`: recibe una `MedicionAgua` en el cuerpo y responde una
      `RespuestaPrediccion` con la predicción y su probabilidad. Exige la
      clave de API con `crear_dependencia_clave`.
    """
    router = APIRouter()

    @router.get("/")
    def informacion():
        return {
            "metricas_validacion": servicios.metadata["metricas_validacion"],
            "entrada": {
                campo: RANGOS[campo]
                for campo in FEATURES
            },
            "salida": {
                "prediccion": "0: No potable, 1: Potable"
            },
        }

    @router.get("/health")
    def health():
        return {"estado": "ok"}

    @router.post(
        "/predict",
        response_model=RespuestaPrediccion,
        dependencies=[Depends(crear_dependencia_clave(servicios))],
    )
    def predict(medicion: MedicionAgua):
        tabla = a_tabla([medicion])

        clases, probabilidades = predecir(
            servicios.modelo, tabla
        )

        return RespuestaPrediccion(
            prediccion=int(clases[0]),
            probabilidad=round(float(probabilidades[0]), 4),
        )

    return router
