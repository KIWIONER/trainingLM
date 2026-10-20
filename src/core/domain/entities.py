"""Entidades y tipos puros del Dominio de Machine Learning y NLP."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class EjemploEntrenamiento:
    texto: str
    etiqueta: str
    respuesta: Optional[str] = None
    origen: Optional[str] = None

@dataclass
class ImpactoToken:
    palabra: str
    peso: float
    impacto: str # "positivo" o "negativo"

@dataclass
class ResultadoExplicacion:
    mensaje_usuario: str
    intencion: str
    respuesta: str
    confianza: float
    palabras_reconocidas: List[Dict[str, Any]]
    palabras_nuevas: List[str]
    reglas: List[str]
    vocabulario_total: int

@dataclass
class PrediccionSimple:
    texto_original: str
    prediccion: str
    confianza: float

@dataclass
class MetricasClase:
    precision: float
    recall: float
    f1: float
    ejemplos: int
    porcentaje: float
    top_palabras: List[str]
