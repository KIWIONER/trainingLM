"""Puerto abstracto para la persistencia del modelo Scikit-Learn."""
from abc import ABC, abstractmethod
from typing import Any, Optional

class IModelStore(ABC):
    @abstractmethod
    def guardar(self, pipeline: Any) -> None:
        """Serializa y persiste el pipeline entrenado."""
        pass

    @abstractmethod
    def cargar(self) -> Optional[Any]:
        """Carga el pipeline entrenado si existe; de lo contrario retorna None."""
        pass
