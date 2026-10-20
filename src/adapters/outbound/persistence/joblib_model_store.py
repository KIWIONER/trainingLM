"""Adaptador secundario: Almacén de modelos serializados con Joblib."""
import os
import joblib
from typing import Any, Optional

from src.core.ports.model_store import IModelStore

class JoblibModelStore(IModelStore):
    def __init__(self, filepath: str = "modelo_entrenado.joblib"):
        self.filepath = filepath

    def guardar(self, pipeline: Any) -> None:
        joblib.dump(pipeline, self.filepath)

    def cargar(self) -> Optional[Any]:
        if os.path.exists(self.filepath):
            try:
                return joblib.load(self.filepath)
            except Exception:
                return None
        return None
