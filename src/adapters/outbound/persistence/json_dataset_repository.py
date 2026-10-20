"""Adaptador secundario: Repositorio JSON para el dataset de entrenamiento."""
import os
import json
import shutil
from typing import List, Dict, Any

from src.core.ports.dataset_repository import IDatasetRepository
from src.core.domain.constants import DATASET_BASE

class JsonDatasetRepository(IDatasetRepository):
    def __init__(self, filepath: str = "dataset_entrenamiento.json", backup_filepath: str = "dataset_entrenamiento.backup.json"):
        self.filepath = filepath
        self.backup_filepath = backup_filepath

    def cargar(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        return data
            except Exception:
                pass
        return list(DATASET_BASE)

    def guardar(self, dataset: List[Dict[str, Any]]) -> None:
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    def crear_backup(self) -> str:
        if os.path.exists(self.filepath):
            shutil.copyfile(self.filepath, self.backup_filepath)
        return self.backup_filepath
