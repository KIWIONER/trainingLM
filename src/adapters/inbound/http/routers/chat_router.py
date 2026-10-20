"""Router para endpoints de chat interactivo y predicción clásica."""
from fastapi import APIRouter, Request, HTTPException, status
from src.adapters.inbound.http.dtos import (
    ChatMessageRequest, ChatResponse, PrediccionRequest, PrediccionResponse
)

router = APIRouter(prefix="/v1", tags=["Chat & Inferencia"])

@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chatear_con_ia(payload: ChatMessageRequest, request: Request):
    nlp_engine = request.app.state.nlp_engine
    try:
        resultado = nlp_engine.explicar_y_responder(payload.mensaje)
        return ChatResponse(
            mensaje_usuario=resultado.mensaje_usuario,
            intencion=resultado.intencion,
            respuesta=resultado.respuesta,
            confianza=resultado.confianza,
            palabras_reconocidas=resultado.palabras_reconocidas,
            palabras_nuevas=resultado.palabras_nuevas,
            reglas=resultado.reglas,
            vocabulario_total=resultado.vocabulario_total
        )
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/predict", response_model=PrediccionResponse, status_code=status.HTTP_200_OK)
async def predecir(payload: PrediccionRequest, request: Request):
    nlp_engine = request.app.state.nlp_engine
    try:
        resultado = nlp_engine.predict(payload.texto)
        return PrediccionResponse(
            texto_original=resultado.texto_original,
            prediccion=resultado.prediccion,
            confianza=resultado.confianza
        )
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
