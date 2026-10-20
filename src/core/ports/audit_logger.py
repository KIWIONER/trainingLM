"""Puerto abstracto para el registro de auditoría de poda y gobernanza ETL."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class IAuditLogger(ABC):
    @abstractmethod
    def registrar_poda(self, entrada: Dict[str, Any]) -> None:
        """Registra un evento de poda de dataset."""
        pass

    @abstractmethod
    def obtener_ultimos(self, maximo: int = 20) -> List[Dict[str, Any]]:
        """Recupera los últimos registros de eventos de poda."""
        pass
