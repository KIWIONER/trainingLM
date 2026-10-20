"""Punto de entrada y Composition Root de Master NLP Language Model & Chat Studio.
Ensambla la arquitectura hexagonal inyectando dependencias entre adaptadores y el núcleo de dominio.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Adaptadores Secundarios (Persistencia en disco)
from src.adapters.outbound.persistence.json_dataset_repository import JsonDatasetRepository
from src.adapters.outbound.persistence.joblib_model_store import JoblibModelStore
from src.adapters.outbound.persistence.jsonl_audit_logger import JsonlAuditLogger

# Servicios del Dominio (ML, NLP, Diagnósticos, Inducción y Poda)
from src.core.services.nlp_engine import NLPEngine
from src.core.services.dataset_balancer import DatasetBalancerService
from src.core.services.synthetic_induction import SyntheticInductionService
from src.core.services.model_evaluator import ModelEvaluatorService

# Adaptadores Primarios (HTTP Routers)
from src.adapters.inbound.http.routers import chat_router, training_router, diagnostics_router

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

def create_app() -> FastAPI:
    """Factoría de aplicación: instancia puertos, adaptadores y ensambla la arquitectura."""
    api = FastAPI(
        title="Master NLP Language Model & Chat Studio",
        version="2.3.0",
        description="Arquitectura Hexagonal: Modelo estadístico predictivo e interpretable con inferencia token a token y aprendizaje continuo."
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 1. Instanciación de adaptadores secundarios (Infraestructura)
    dataset_repo = JsonDatasetRepository()
    model_store = JoblibModelStore()
    audit_logger = JsonlAuditLogger()

    # 2. Instanciación de servicios del dominio (Núcleo)
    dataset_balancer = DatasetBalancerService()
    induction_service = SyntheticInductionService()
    
    # Intentar cargar pipeline previo o inicializar nuevo
    pipeline_previo = model_store.cargar()
    nlp_engine = NLPEngine(pipeline=pipeline_previo)

    # Inicializar y sanear dataset en arranque
    raw_dataset = dataset_repo.cargar()
    saneado = dataset_balancer.sanear_dataset_existente(raw_dataset)
    dataset_limpio, registro_poda = dataset_balancer.aplicar_cuota_maxima(saneado)
    if registro_poda:
        audit_logger.registrar_poda(registro_poda)

    nlp_engine.fit(dataset_limpio)
    dataset_repo.guardar(dataset_limpio)
    model_store.guardar(nlp_engine.pipeline)

    evaluator_service = ModelEvaluatorService(base_pipeline=nlp_engine.pipeline)

    # 3. Inyección de dependencias en el estado de la aplicación
    api.state.dataset_repo = dataset_repo
    api.state.model_store = model_store
    api.state.audit_logger = audit_logger
    api.state.dataset_balancer = dataset_balancer
    api.state.induction_service = induction_service
    api.state.nlp_engine = nlp_engine
    api.state.evaluator_service = evaluator_service

    # 4. Registro de Routers
    api.include_router(chat_router.router)
    api.include_router(training_router.router)
    api.include_router(diagnostics_router.router)

    # 5. Servir Frontend Estático
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR, exist_ok=True)

    api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @api.get("/")
    async def raiz():
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"mensaje": "API activa. Abre /docs o crea static/index.html"}

    return api

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("traininglm:app", host="127.0.0.1", port=8001, reload=True)