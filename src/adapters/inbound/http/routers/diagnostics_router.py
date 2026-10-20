"""Router para endpoints de diagnóstico, estado, catálogo de habilidades y sugerencias dinámicas."""
import random
from fastapi import APIRouter, Request, status
from src.core.domain.constants import HABILIDADES_PACKS

router = APIRouter(prefix="/v1", tags=["Diagnóstico & Estado"])

@router.get("/evaluation", status_code=status.HTTP_200_OK)
async def evaluar_modelo(request: Request):
    app_state = request.app.state
    # Sincronizar dataset desde repositorio
    dataset = app_state.dataset_repo.cargar()
    if dataset and len(dataset) >= 2:
        if len(dataset) != len(app_state.nlp_engine.historico_dataset):
            app_state.nlp_engine.fit(dataset)
    
    registros = app_state.audit_logger.obtener_ultimos()
    return app_state.evaluator_service.evaluar(app_state.nlp_engine.historico_dataset, registros)

@router.get("/state", status_code=status.HTTP_200_OK)
async def estado_modelo(request: Request):
    app_state = request.app.state
    nlp_engine = app_state.nlp_engine
    if not nlp_engine.esta_entrenado:
        return {"esta_entrenado": False, "vocabulario_total": 0, "ejemplos_totales": 0}
        
    vectorizer = nlp_engine.pipeline.named_steps['vectorizador']
    classifier = nlp_engine.pipeline.named_steps['clasificador']
    vocab = list(vectorizer.vocabulary_.keys())
    
    return {
        "esta_entrenado": nlp_engine.esta_entrenado,
        "vocabulario_total": len(vocab),
        "ejemplos_totales": len(nlp_engine.historico_dataset),
        "clases": list(classifier.classes_),
        "vocabulario_muestra": sorted(vocab[:60]),
        "dataset_reciente": nlp_engine.historico_dataset[-8:]
    }

@router.get("/skills", status_code=status.HTTP_200_OK)
async def listar_habilidades():
    resultado = []
    for k, v in HABILIDADES_PACKS.items():
        resultado.append({
            "id": v["id"],
            "nombre": v["nombre"],
            "icono": v["icono"],
            "etiqueta": v["etiqueta"],
            "palabras_clave": v["palabras_clave"],
            "ejemplo_patron": v["plantillas_pares"][0][0]
        })
    return resultado

@router.get("/suggestions", status_code=status.HTTP_200_OK)
async def obtener_sugerencias(request: Request):
    app_state = request.app.state
    dataset = app_state.nlp_engine.historico_dataset
    sugerencias = []
    
    if dataset:
        clases_incluidas = set()
        for item in reversed(dataset):
            c = item.get("etiqueta")
            txt = item.get("texto", "").strip()
            if c not in clases_incluidas and len(txt.split()) >= 2 and len(txt) <= 70:
                sugerencias.append(txt)
                clases_incluidas.add(c)
            if len(sugerencias) >= 3:
                break
                
    habilidades_disponibles = list(HABILIDADES_PACKS.keys())
    if habilidades_disponibles:
        muestras = random.sample(habilidades_disponibles, min(3, len(habilidades_disponibles)))
        for h_id in muestras:
            try:
                pares, _, _ = app_state.induction_service.sintetizar(h_id, cantidad=1, permitir_descubrimiento=False)
                if pares and pares[0][0] not in sugerencias:
                    sugerencias.append(pares[0][0])
            except Exception:
                pass
                
    return [s.capitalize() for s in sugerencias[:6]]
