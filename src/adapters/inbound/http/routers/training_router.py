"""Router para endpoints de entrenamiento, nutrición, inducción sintética y poda/balanceo."""
from fastapi import APIRouter, Request, HTTPException, status
from typing import Dict, Any

from src.adapters.inbound.http.dtos import (
    NutrirRequest, InducirEntrenamientoRequest, DatosEntrenamientoRequest, EntrenamientoResponse
)
from src.core.domain.constants import get_tokens_limpios, HABILIDADES_PACKS, OBJETIVO_POR_CLASE

router = APIRouter(prefix="/v1", tags=["Entrenamiento & Aprendizaje"])

def persistir_y_entrenar(app_state, nuevo_dataset):
    """Aplica cuota máxima, audita poda si aplica, entrena pipeline y persiste a disco."""
    dataset_limpio, registro_poda = app_state.dataset_balancer.aplicar_cuota_maxima(nuevo_dataset)
    if registro_poda:
        app_state.audit_logger.registrar_poda(registro_poda)

    app_state.nlp_engine.fit(dataset_limpio)
    app_state.dataset_repo.guardar(dataset_limpio)
    app_state.model_store.guardar(app_state.nlp_engine.pipeline)
    return dataset_limpio

@router.post("/teach", status_code=status.HTTP_200_OK)
async def nutrir_modelo(payload: NutrirRequest, request: Request):
    app_state = request.app.state
    nuevo_item = {
        "texto": payload.texto.strip(),
        "etiqueta": payload.etiqueta.strip(),
        "respuesta": payload.respuesta.strip() if payload.respuesta else f"Entendido. He incorporado el conocimiento de '{payload.texto.strip()}' a mi clase '{payload.etiqueta.strip()}'."
    }
    
    dataset = list(app_state.nlp_engine.historico_dataset)
    dataset.append(nuevo_item)
    
    stopwords_es = {"como", "para", "este", "esta", "estos", "estas", "tengo", "quiero", "necesito", "puedo", "hacer", "crear", "sobre", "entre", "desde", "hasta", "donde", "cuando", "porque", "pero", "tiene", "tienen"}
    tokens_nuevos = get_tokens_limpios(payload.texto)
    for p_id, p_data in HABILIDADES_PACKS.items():
        for t in tokens_nuevos:
            if len(t) >= 4 and t not in stopwords_es and t not in p_data["palabras_clave"]:
                p_data["palabras_clave"].append(t)

    dataset_final = persistir_y_entrenar(app_state, dataset)
    vocab_len = len(app_state.nlp_engine.pipeline.named_steps['vectorizador'].vocabulary_)
    
    return {
        "mensaje": "¡Aprendizaje exitoso! He incorporado la frase y su respuesta a mis patrones de predicción.",
        "ejemplos_totales": len(dataset_final),
        "vocabulario_aprendido": vocab_len,
        "etiqueta_asociada": payload.etiqueta
    }

@router.post("/induce-training", status_code=status.HTTP_200_OK)
async def inducir_entrenamiento(payload: InducirEntrenamientoRequest, request: Request):
    app_state = request.app.state
    try:
        pares_sinteticos, etiqueta, descubiertas = app_state.induction_service.sintetizar(
            payload.habilidad_id, 
            payload.cantidad,
            permitir_descubrimiento=True
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    nuevos_items = []
    for item in pares_sinteticos:
        if len(item) == 3:
            preg, resp, etq = item
        else:
            preg, resp = item
            etq = etiqueta
        nuevos_items.append({"texto": preg, "etiqueta": etq, "respuesta": resp})

    dataset = list(app_state.nlp_engine.historico_dataset)
    dataset.extend(nuevos_items)
    
    dataset_final = persistir_y_entrenar(app_state, dataset)
    vocab_len = len(app_state.nlp_engine.pipeline.named_steps['vectorizador'].vocabulary_)
    
    return {
        "mensaje": f"¡Inducción completada! Se sintetizaron y aprendieron {len(pares_sinteticos)} pares únicos de pregunta y respuesta.",
        "habilidad_id": payload.habilidad_id,
        "etiqueta": etiqueta,
        "palabras_descubiertas": descubiertas,
        "frases_generadas": [p[0] for p in pares_sinteticos],
        "respuestas_generadas": [p[1] for p in pares_sinteticos],
        "ejemplos_totales": len(dataset_final),
        "vocabulario_aprendido": vocab_len
    }

@router.post("/prune-dataset", status_code=status.HTTP_200_OK)
async def podar_dataset_manual(request: Request):
    app_state = request.app.state
    dataset = list(app_state.nlp_engine.historico_dataset)
    antes = len(dataset)
    
    compactado = app_state.dataset_balancer.podar_conservando_vocabulario(dataset, max_por_clase=12)
    dataset_final = persistir_y_entrenar(app_state, compactado)
    despues = len(dataset_final)
    vocab_len = len(app_state.nlp_engine.pipeline.named_steps['vectorizador'].vocabulary_)
    
    return {
        "mensaje": f"Dataset compactado con éxito: de {antes} a {despues} ejemplos.",
        "ejemplos_antes": antes,
        "ejemplos_despues": despues,
        "vocabulario_conservado": vocab_len
    }

@router.post("/balance-dataset", status_code=status.HTTP_200_OK)
async def equilibrar_clases(request: Request, objetivo: int = OBJETIVO_POR_CLASE):
    app_state = request.app.state
    objetivo = max(10, min(60, objetivo))
    backup_file = app_state.dataset_repo.crear_backup()

    dataset_actual = list(app_state.nlp_engine.historico_dataset)
    eval_antes = app_state.evaluator_service.evaluar(dataset_actual, app_state.audit_logger.obtener_ultimos())
    vocab_antes = app_state.dataset_balancer.vocabulario_de(dataset_actual)

    nuevo, informe, fuera, reg_evento = app_state.dataset_balancer.equilibrar(dataset_actual, objetivo)
    if reg_evento:
        app_state.audit_logger.registrar_poda(reg_evento)

    app_state.nlp_engine.fit(nuevo)
    app_state.dataset_repo.guardar(nuevo)
    app_state.model_store.guardar(app_state.nlp_engine.pipeline)

    vocab_despues = app_state.dataset_balancer.vocabulario_de(nuevo)
    eval_despues = app_state.evaluator_service.evaluar(nuevo, app_state.audit_logger.obtener_ultimos())

    return {
        "mensaje": f"Clases equilibradas a {objetivo} ejemplos cada una.",
        "objetivo_por_clase": objetivo,
        "informe_clases": informe,
        "descartadas_fuera_de_dominio": fuera,
        "palabras_perdidas": sorted(vocab_antes - vocab_despues)[:30],
        "total_palabras_perdidas": len(vocab_antes - vocab_despues),
        "palabras_ganadas": len(vocab_despues - vocab_antes),
        "cross_val_antes": eval_antes.get("cross_val_media"),
        "cross_val_despues": eval_despues.get("cross_val_media"),
        "backup": backup_file,
        "evaluacion": eval_despues
    }

@router.post("/train", response_model=EntrenamientoResponse, status_code=status.HTTP_200_OK)
async def entrenar_modelo(payload: DatosEntrenamientoRequest, request: Request):
    app_state = request.app.state
    dataset = [{"texto": t, "etiqueta": e, "respuesta": f"Respuesta aprendida para {e}."} for t, e in zip(payload.textos, payload.etiquetas)]
    dataset_final = persistir_y_entrenar(app_state, dataset)
    vocab_len = len(app_state.nlp_engine.pipeline.named_steps['vectorizador'].vocabulary_)
    return EntrenamientoResponse(
        mensaje="Modelo entrenado exitosamente con Scikit-Learn.",
        ejemplos_procesados=len(payload.textos),
        vocabulario_aprendido=vocab_len
    )
