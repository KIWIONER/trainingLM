"""Adaptador secundario: Registro en JSON Lines para auditoría y gobernanza de poda."""
import os
import json
from datetime import datetime
from typing import List, Dict, Any

from src.core.ports.audit_logger import IAuditLogger

class JsonlAuditLogger(IAuditLogger):
    def __init__(self, filepath: str = "registro_poda.jsonl", max_en_memoria: int = 20):
        self.filepath = filepath
        self.max_en_memoria = max_en_memoria
        self._memoria = self._cargar_inicial()

    def _cargar_inicial(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.filepath):
            return []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f.readlines()[-self.max_en_memoria:] if line.strip()]
        except Exception:
            return []

    def registrar_poda(self, entrada: Dict[str, Any]) -> None:
        evento = {"fecha": datetime.now().isoformat(timespec="seconds"), **entrada}
        self._memoria.append(evento)
        del self._memoria[:-self.max_en_memoria]
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(evento, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def obtener_ultimos(self, maximo: int = 20) -> List[Dict[str, Any]]:
        return list(self._memoria[-maximo:])
