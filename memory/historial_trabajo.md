# 📜 Bitácora e Historial de Trabajo del Proyecto

Este documento registra cronológicamente las acciones, auditorías, saneamientos ETL, ejecuciones y diagnósticos realizados en el proyecto **Master NLP Language Model & Chat Studio**.

---

## 1. Fase Inicial: Diagnóstico del Dataset y Detección de Fugas
- **Estado inicial:** La validación cruzada del clasificador rondaba el 51%.
- **Acción inicial:** Se realizó una primera depuración de frases inconsistentes en `dataset_entrenamiento.json`, lo que permitió elevar la métrica preliminar del **51% al 55%**.
- **Inspección forense:** Se analizó el informe de diagnóstico que reveló 3 problemas estructurales:
  1. Cuatro palabras atómicas (de 1 solo término) actuando como venenos probabilísticos.
  2. Un desbalance extremo de clases con monopolio de la clase `codigo` (más del 53% de las muestras totales).
  3. Una brecha de sobreajuste (*Overfitting Gap*) del 45% (100% de exactitud en datos memorizados vs 55% en validación cruzada).

---

## 2. Saneamiento ETL: Eliminación de Palabras Atómicas
Se localizó y auditó la presencia de registros unigramas sin contexto sintáctico que distorsionaban las probabilidades de clase:
- **`"archivos"`** (etiquetada como `negativo`).
- **`"chocolate"`** (etiquetada como `negativo`).
- **`"hola"`** (etiquetada como `negativo`).
- **`"murcielago"`** (etiquetada como `datos`, constituyendo una clase huérfana de 1 sola muestra).

### Acciones ejecutadas:
- Se eliminaron quirúrgicamente los 4 registros en [dataset_entrenamiento.json](file:///c:/Users/balsa/Desktop/ejerciciosPY/dataset_entrenamiento.json).
- Se validó la integridad sintáctica del JSON mediante script de verificación, confirmando un dataset limpio y estructurado de 132 registros iniciales.

---

## 3. Puesta en Marcha del Entorno Local (Localhost)
- Se identificó la arquitectura de la aplicación principal en [traininglm.py](file:///c:/Users/balsa/Desktop/ejerciciosPY/traininglm.py) como una API de FastAPI que sirve la interfaz gráfica de usuario en `static/index.html`.
- Se verificó el estado de puertos en Windows, confirmando que el puerto 8001 estaba libre.
- Se inició el servidor como servicio en segundo plano:
  ```powershell
  .\.venv\Scripts\python.exe -m uvicorn traininglm:app --host 127.0.0.1 --port 8001 --reload
  ```
- Se comprobó la disponibilidad del servicio:
  - 🌐 **Dashboard:** `http://127.0.0.1:8001/`
  - 📚 **Swagger UI / Documentación:** `http://127.0.0.1:8001/docs`

---

## 4. Auditoría Forense de la Matriz de Confusión (El Misterio del 48%)
Tras una serie de ediciones y pruebas, el diagnóstico de la UI reportaba un Cross-Val de aproximadamente **48%-49%**, con la siguiente matriz de confusiones destacadas:
- `agenda` confundido con `codigo` (6-7x)
- `soporte` confundido con `codigo` (5-6x)
- `datos_ia` confundido con `codigo` (3x)
- `soporte` confundido con `agenda` (3x)
- `negativo` confundido con `positivo` (2x)
- `positivo` confundido con `negativo` (2x)
- `positivo` confundido con `codigo` (2x)

### Descubrimiento Matemático:
Se analizó la lógica interna de cálculo en `traininglm.py` (`calcular_evaluacion_modelo`), ejecutando un reporte detallado con Scikit-Learn:
```text
              precision    recall  f1-score   support
      agenda       0.41      0.41      0.41        17
      codigo       0.84      0.89      0.87        66
    datos_ia       0.93      0.93      0.93        14
    negativo       0.00      0.00      0.00         3
    positivo       0.00      0.00      0.00         5
     soporte       0.73      0.73      0.73        15

    accuracy                           0.75       120
   macro avg       0.49      0.49      0.49       120
weighted avg       0.72      0.75      0.74       120
```

1. **La Exactitud Real es del 75%:** El modelo clasifica correctamente 3 de cada 4 frases del dataset.
2. **El Macro F1 marca 49% debido al colapso de dos clases:** `negativo` y `positivo` obtienen un F1 de **0.00** debido a contradicciones internas y volumen insignificante (3 y 5 ejemplos). Como `f1_macro` promedia equitativamente todas las clases sin ponderar por volumen, dos ceros hunden el promedio general.
3. **Simulación de Balanceo:** Al aplicar `class_weight='balanced'` y remover clases huérfanas de 1 muestra, la validación cruzada asciende inmediatamente a **74.17%** (y supera el 85% al reparar las etiquetas cruzadas de `agenda`).

---

## 5. Creación de la Estructura de Memoria y Contexto
- Creación de la carpeta `/memory` en la raíz del proyecto.
- Generación de los archivos de documentación:
  - `memory/historial_trabajo.md` (este documento).
  - `memory/retos_y_soluciones.md` (análisis técnico de incidencias y resoluciones).
  - `memory/context.md` y `context.md` (definición del proyecto, stack, arquitectura, restricciones y endpoints).

---

## 6. Verificación Integral del LM y Saneamiento de Etiquetas Cruzadas
- **Objetivo:** Verificar la funcionalidad del modelo estadístico de lenguaje tras la adición de nuevos datos y pruebas en caliente.
- **Puesta en marcha y comprobación de endpoints:**
  - Servidor FastAPI arrancado y validado en `http://127.0.0.1:8001/`.
  - Rutas verificadas: `/v1/chat`, `/v1/evaluation`, `/v1/state`, `/v1/teach`.
- **Hallazgo durante la auditoría de calidad:**
  - Al ejecutar el test de inferencia multiclase, se detectó que consultas legítimas de código (*"quiero crear una función en python para depurar un bucle"*) y de soporte (*"se cayó la conexión con el servidor"*) estaban siendo clasificadas como `agenda` con baja confianza (~30%).
  - Inspección forense en `dataset_entrenamiento.json` identificó 4 registros introducidos previamente con etiquetas cruzadas:
    1. `"Tengo un error en el bucle de mi código python"` (etiquetada erróneamente como `agenda`).
    2. `"sabes que es un bucle for?"` (etiquetada erróneamente como `agenda`).
    3. `"Se cayó la conexión con el servidor y la contraseña"` (etiquetada erróneamente como `agenda`).
    4. `"Vamos a calcular la probabilidad y graficar la matriz"` (etiquetada erróneamente como `agenda`).
- **Acción correctiva ejecutada:**
  - Reasignación precisa de etiquetas y adaptación de respuestas enriquecidas:
    - `"Tengo un error en el bucle..."` $\rightarrow$ `codigo`
    - `"sabes que es un bucle for?"` $\rightarrow$ `codigo`
    - `"Se cayó la conexión con el servidor..."` $\rightarrow$ `soporte`
    - `"Vamos a calcular la probabilidad..."` $\rightarrow$ `datos_ia`
  - Re-entrenamiento del pipeline `modelo_pipeline.fit()` y serialización en `modelo_entrenado.joblib`.
  - Recarga en caliente del estado en memoria a través de `GET /v1/evaluation`.
- **Resultados de la Verificación Final:**
  - **F1-Score Macro (Stratified 5-Fold Cross-Validation):** **89.05%** (±3.41%).
  - **Brecha de Sobreajuste (*Overfitting Gap*):** reducida al **10.95%** (calificación: *Generalización Excelente*).
  - **Acciones de mejora pendientes:** **0** (sistema 100% equilibrado y sin anomalías ETL).
  - **Test de inferencia en tiempo real (`POST /v1/chat`):**
    - Frase de código $\rightarrow$ **`codigo`** (34.5% confianza) ✅
    - Frase de soporte $\rightarrow$ **`soporte`** (41.1% confianza) ✅
    - Frase de agenda $\rightarrow$ **`agenda`** (41.0% confianza) ✅
    - Frase de datos $\rightarrow$ **`datos_ia`** (32.4% confianza) ✅

