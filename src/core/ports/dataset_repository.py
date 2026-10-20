"""Puerto abstracto para el repositorio de datos de entrenamiento."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class IDatasetRepository(ABC):
    @abstractmethod
    def cargar(self) -> List[Dict[str, Any]]:
        """Carga los ejemplos de entrenamiento desde el almacenamiento persistente."""
        pass

    @abstractmethod
    def guardar(self, dataset: List[Dict[str, Any]]) -> None:
        """Guarda la lista completa de ejemplos en el almacenamiento persistente."""
        pass

    @abstractmethod
    def crear_backup(self) -> str:
        """Crea una copia de seguridad y retorna la ruta o identificador del backup."""
        pass
