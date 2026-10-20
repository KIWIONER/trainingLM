# 🧠 Contexto del Proyecto: Master NLP Language Model & Chat Studio

---

## 1. Definición del Proyecto
**Master NLP Language Model & Chat Studio** es una plataforma interactiva de procesamiento de lenguaje natural (NLP) y aprendizaje automático continuo en caliente.

A diferencia de un modelo de caja negra o un LLM externo, el sistema implementa un **modelo estadístico predictivo e interpretable**:
- Deriva y clasifica intenciones de usuario a partir de su propio dataset vivo.
- Ofrece explicabilidad token a token (peso e impacto de palabras en cada predicción).
- Permite la nutrición y expansión léxica en tiempo real (*online learning* / re-entrenamiento en caliente).
- Incorpora un módulo de diagnóstico formal con Validación Cruzada Estratificada, Detección de Sobreajuste (*Overfitting Gap*), Detección de Anomalías ETL y Auditoría Léxica.

---

## 2. Stack Tecnológico

### Arquitectura del Sistema: Hexagonal (Ports & Adapters)
El proyecto ha sido modularizado bajo una arquitectura limpia en `src/`:
- **Dominio Central (`src/core/`):**
  - `domain/`: Entidades puras (`entities.py`) y constantes léxicas/plantillas gramaticales (`constants.py`).
  - `ports/`: Interfaces abstractas (`dataset_repository.py`, `model_store.py`, `audit_logger.py`).
  - `services/`: Motores de NLP (`nlp_engine.py`), balanceo y poda (`dataset_balancer.py`), inducción sintética (`synthetic_induction.py`) y evaluación formal (`model_evaluator.py`).
- **Adaptadores (`src/adapters/`):**
  - `inbound/http/`: DTOs de Pydantic (`dtos.py`) y routers FastAPI modulares (`chat_router.py`, `training_router.py`, `diagnostics_router.py`).
  - `outbound/persistence/`: Repositorio JSON (`json_dataset_repository.py`), almacén Joblib (`joblib_model_store.py`) y registro JSONL (`jsonl_audit_logger.py`).
- **Composition Root (`traininglm.py`):** Instancia y ensambla los adaptadores con los servicios inyectando dependencias en `app.state`, garantizando compatibilidad total con Uvicorn y despliegues en la nube.

### Backend & API
- **Lenguaje:** Python 3.12+
- **Framework Web:** FastAPI (API REST asíncrona, validación con Pydantic v2).
- **Servidor ASGI:** Uvicorn con soporte de recarga en caliente (*hot-reload*).
- **Serialización:** Joblib (para volcado y carga de pipelines de Scikit-Learn) y JSON nativo.

### Machine Learning & NLP
- **Librería Central:** Scikit-Learn (v1.6+)
- **Pipeline de Modelado:**
  1. `TfidfVectorizer(ngram_range=(1, 2))`: Extracción de características por frecuencia de término y frecuencias inversas de documento sobre unigramas y bigramas.
  2. `LogisticRegression(class_weight='balanced', max_iter=1000)`: Clasificador lineal probabilístico multiclase con balanceo automático de pesos según soporte de clase.
- **Validación y Diagnóstico:** `StratifiedKFold`, `cross_val_score`, `cross_val_predict`, `classification_report`, `confusion_matrix`.
- **Cálculo Numérico:** NumPy.

### Frontend & UI
- **Tecnología:** React embebido con compilación en navegador mediante Babel Standalone.
- **Diseño & Estilos:** TailwindCSS (vía CDN) con paleta temática oscura (*dark glassmorphism*).
- **Iconografía:** Lucide Icons.
- **Ubicación:** `static/index.html` servido directamente por el backend de FastAPI en `/`.

### Almacenamiento & Persistencia
- `dataset_entrenamiento.json`: Registro vivo de ejemplos `(texto, etiqueta, respuesta)`.
- `modelo_entrenado.joblib`: Binario serializado del pipeline Scikit-Learn re-entrenado.

---

## 3. Restricciones y Reglas del Sistema

### Restricciones de Datos (ETL y Calidad Léxica)
1. **Regla de Contexto Mínimo (No Unigramas Aislados):** Ninguna frase de entrenamiento debe tener menos de 2 palabras. Las palabras atómicas aisladas distorsionan los pesos de TF-IDF y envenenan los sesgos del clasificador.
2. **Requisito de Muestras por Clase ($K \ge 2$):** Para que una clase participe en la validación cruzada estratificada (`StratifiedKFold`), debe contener al menos 2 ejemplos. Clases con 1 sola muestra (huérfanas) deben excluirse o nutrirse de inmediato.
3. **No Contradicción Léxica:** Evitar etiquetar la misma entidad semántica o expresiones equivalentes en clases opuestas (ejemplo: frases idénticas de satisfacción etiquetadas como positivo y negativo).

### Restricciones de Ejecución en Entorno Windows
- En sistemas con directivas de control de aplicaciones (AppLocker), la invocación de scripts o ejecutables `.exe` dentro de `.venv\Scripts\` puede estar restringida.
- **Regla de oro:** Ejecutar siempre a través del intérprete de Python:
  ```powershell
  .\.venv\Scripts\python.exe -m <modulo>
  ```

---

## 4. Guía de Inicialización y Ejecución

### 1. Activar el Entorno Virtual y Levantar el Servidor
Desde la raíz del proyecto (`c:\Users\balsa\Desktop\ejerciciosPY`):
```powershell
.\.venv\Scripts\python.exe -m uvicorn traininglm:app --host 127.0.0.1 --port 8001 --reload
```

### 2. Puntos de Acceso
| Interfaz | URL |
| :--- | :--- |
| **Studio Web Interactivo (Frontend React)** | [http://127.0.0.1:8001/](http://127.0.0.1:8001/) |
| **Documentación Interactiva Swagger (OpenAPI)** | [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs) |
| **Documentación ReDoc** | [http://127.0.0.1:8001/redoc](http://127.0.0.1:8001/redoc) |

---

## 5. Listado de Funcionalidades Principales

### 💬 1. Chat Explicable con Desglose Léxico (`POST /v1/chat`)
- Clasifica la intención de cualquier frase en tiempo real con puntaje de confianza probabilística.
- Desglosa las palabras reconocidas mostrando su coeficiente de impacto (+ o -) en la decisión.
- Detecta palabras nuevas (desconocidas por el vocabulario) y sugiere su aprendizaje.
- Asigna respuestas adaptadas al estilo y vocabulario de la clase predicha.

### 🔬 2. Diagnóstico Formal y Auditoría de Aprendizaje (`GET /v1/evaluation`)
- **Validación Cruzada Estratificada:** Evaluación real fold por fold (`StratifiedKFold`) con métricas Macro F1, Weighted F1 y Accuracy.
- **Medición de Sobreajuste (*Overfitting Gap*):** Compara el rendimiento en entrenamiento frente a test para alertar sobre memorización excesiva.
- **Matriz de Confusión y Ranking de Confusiones Frecuentes:** Identifica qué pares de clases colisionan y cuántas veces.
- **Auditoría de Calidad ETL:** Métricas de riqueza léxica, longitud promedio de frases, detección de duplicados y detección de términos atómicos (*outliers*).
- **Acciones de Mejora Automatizadas:** Recomendaciones operativas basadas en el estado del dataset.

### ⚡ 3. Inductor de Frases Sintéticas Híbridas (`POST /v1/induce-training`)
- **Cerebro Integral Cognitivo:** Generador sintético que combina verbos y sustantivos respetando el acuerdo gramatical por categorías gramaticales (POS).
- Permite expandir clases minoritarias (`soporte`, `datos_ia`, `agenda`) para equilibrar el dataset sin provocar desbalances.

### 📚 4. Nutrición Activa del Modelo (`POST /v1/teach`)
- Permite al usuario enseñar nuevas frases, asociarlas a una intención existente o nueva, y asignar respuestas personalizadas.
- Re-entrena el pipeline en memoria y guarda el progreso en `dataset_entrenamiento.json` y `modelo_entrenado.joblib`.

### ✂️ 5. Poda Inteligente de Dataset (`POST /v1/prune-dataset`)
- Reduce el volumen de clases masivas o hipertrofiadas (como `codigo`) limitando la cantidad de ejemplos por clase.
- **Garantía Léxica:** Conserva el 100% del vocabulario único acumulado seleccionando las frases con mayor densidad informativa.

### 🧠 6. Estado Mental y Sugerencias Dinámicas (`GET /v1/state` & `GET /v1/suggestions`)
- Monitorea el número de clases activas, total de ejemplos, tamaño del vocabulario n-grama y distribución porcentual.
- Sugiere frases contextuales para probar en el chat basadas en las últimas habilidades aprendidas.

---

## 6. Estado Validado y Métricas de Referencia (Baseline Actual)

| Métrica / Dimensión | Valor Actual | Calificación del Motor |
| :--- | :--- | :--- |
| **F1-Score Macro (Cross-Val 5-Fold)** | **89.05%** (±3.41%) | Sólido y robusto |
| **Accuracy en Entrenamiento** | **100.0%** | Ajuste óptimo |
| **Brecha de Sobreajuste (*Overfitting Gap*)** | **10.95%** | Generalización Excelente (< 15%) |
| **Volumen del Dataset Activo** | 125 ejemplos equilibrados | Ratio máx/mín: 1.5x (sin monopolio) |
| **Distribución de Clases** | `codigo`: 37, `soporte`: 33, `agenda`: 31, `datos_ia`: 24 | Ecualizado |
| **Vocabulario N-Gramas** | 890 unigramas y bigramas | Riqueza léxica: 17.3% |
| **Anomalías ETL / Outliers** | 0 términos atómicos, 0 duplicados | Sano |
| **Acciones de Mejora Pendientes** | 0 advertencias activas | Óptimo |

