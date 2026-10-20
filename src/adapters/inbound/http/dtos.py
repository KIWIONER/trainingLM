"""DTOs y esquemas de validación con Pydantic para la API HTTP."""
from typing import List, Optional
from pydantic import BaseModel, Field

class DatosEntrenamientoRequest(BaseModel):
    textos: List[str] = Field(..., min_length=2, description="Lista de frases de ejemplo")
    etiquetas: List[str] = Field(..., min_length=2, description="Categoría de cada frase")

class EntrenamientoResponse(BaseModel):
    mensaje: str
    ejemplos_procesados: int
    vocabulario_aprendido: int

class PrediccionRequest(BaseModel):
    texto: str = Field(..., min_length=1, description="Texto nuevo a clasificar")

class PrediccionResponse(BaseModel):
    texto_original: str
    prediccion: str
    confianza: float

class ChatMessageRequest(BaseModel):
    mensaje: str = Field(..., min_length=1, description="Mensaje enviado por el usuario en el chat")

class PalabraImpacto(BaseModel):
    palabra: str
    peso: float
    impacto: str # "positivo" o "negativo"

class ChatResponse(BaseModel):
    mensaje_usuario: str
    intencion: str
    respuesta: str
    confianza: float
    palabras_reconocidas: List[PalabraImpacto]
    palabras_nuevas: List[str]
    reglas: List[str]
    vocabulario_total: int

class NutrirRequest(BaseModel):
    texto: str = Field(..., min_length=2, description="Nueva frase para nutrir el modelo")
    etiqueta: str = Field(..., min_length=1, description="Categoría o intención asignada")
    respuesta: Optional[str] = Field(default=None, description="Respuesta personalizada para esta intención")

class InducirEntrenamientoRequest(BaseModel):
    habilidad_id: str = Field(..., description="ID del paquete temático a inducir (ej: 'codigo', 'soporte')")
    cantidad: int = Field(default=3, ge=1, le=15, description="Número de frases a sintetizar y aprender")
