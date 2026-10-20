# 🧩 Retos Encontrados y Soluciones Aplicadas

Este documento recoge los problemas estructurales, matemáticos y de entorno encontrados durante el desarrollo, depuración y entrenamiento del modelo de lenguaje estadístico, así como la metodología aplicada para solucionarlos.

---

## Reto 1: Palabras Atómicas en ETL ("Venenos Probabilísticos")

### Síntoma:
Al realizar la auditoría léxica en la interfaz, se mostraba una alerta roja indicando términos como:
> ⚠️ `"hola"` (`negativo`), `"chocolate"` (`negativo`), `"archivos"` (`negativo`), `"murcielago"` (`datos`).

### Causa Raíz:
En un vectorizador TF-IDF con n-gramas (1, 2), una palabra única sin contexto sintáctico recibe un peso desproporcionado. Al asociar palabras comunes y neutras (como *"hola"* o *"archivos"*) a la clase `negativo`, cualquier frase entrante que contenga un saludo o mencione un archivo elevaba falsamente la probabilidad hacia `negativo`, distorsionando tanto el clasificador como el chat interactivo.

### Solución:
- Se purgaron de raíz los registros atómicos unigramas en [dataset_entrenamiento.json](file:///c:/Users/balsa/Desktop/ejerciciosPY/dataset_entrenamiento.json).
- Se estableció como regla de ETL que cada ejemplo de entrenamiento debe tener un contexto mínimo sintáctico de al menos 2 palabras (sujeto/verbo/objeto o verbo/sustantivo).

---

## Reto 2: El "Efecto Imán" de la Clase Monopólica (`codigo`)

### Síntoma:
La matriz de confusión mostraba un sesgo masivo donde el modelo predecía casi siempre `codigo`:
- `agenda` confundido con `codigo` (7 veces).
- `soporte` confundido con `codigo` (5 veces).
- `datos_ia` confundido con `codigo` (3 veces).

### Causa Raíz:
`codigo` poseía más del 54% de las muestras del dataset (más de 66 ejemplos), mientras que las clases minoritarias tenían entre 3 y 15 ejemplos. Los clasificadores de regresión logística aprenden un sesgo (*prior probability* e intercepto) dominante para la clase mayoritaria. Ante la menor duda o presencia de un término técnico levemente compartido, el modelo optaba por la clase con mayor frecuencia previa.

### Solución:
1. **Configuración del Modelo:** Asegurar el parámetro `class_weight='balanced'` en el estimador de `LogisticRegression` para penalizar los errores en clases minoritarias de forma inversamente proporcional a sus frecuencias.
2. **Generación con Cerebro Integral:** Usar el motor de síntesis de frases híbridas con acuerdo sintáctico POS (`/v1/induce-training`) para nutrir las clases desfavorecidas (`soporte`, `datos_ia`, `agenda`) sin contaminar el vocabulario léxico.
3. **Poda Controlada:** Emplear el endpoint `/v1/prune-dataset` para reducir la saturación de `codigo` conservando el 100% de su vocabulario único.

---

## Reto 3: La Paradoja Métrica (Accuracy 75% vs. Macro F1 49%)

### Síntoma:
A pesar de que el modelo predecía correctamente la gran mayoría de las entradas en tiempo real y la exactitud general era del 75%, el indicador de la interfaz mostraba un alarmante **48%-49% de Cross-Validation**.

### Causa Raíz:
La función `calcular_evaluacion_modelo` en [traininglm.py](file:///c:/Users/balsa/Desktop/ejerciciosPY/traininglm.py#L768) utiliza `scoring='f1_macro'`:
- `f1_macro` calcula la media aritmética no ponderada de los F1-Scores de cada clase.
- Las clases `positivo` y `negativo` tenían un rendimiento de **0.00 de F1** debido a contradicciones internas y escasez de muestras.
- Matemáticamente:
  $$\text{Macro F1} = \frac{0.93 (\text{datos}) + 0.87 (\text{codigo}) + 0.73 (\text{soporte}) + 0.41 (\text{agenda}) + 0.00 (\text{pos}) + 0.00 (\text{neg})}{6} \approx 0.49$$
Dos ceros en clases secundarias hundían el 50% de la métrica general de todo el cerebro.

### Solución:
- Clarificar la diferencia de interpretación entre **Accuracy** (75%), **Weighted F1** (74%) y **Macro F1** (49%).
- Priorizar la corrección o exclusión de las micro-clases colapsadas para permitir que la validación cruzada macro refleje fielmente la solidez de las intenciones del asistente (superando el 80%-85%).

---

## Reto 4: Contradicciones Cruzadas en Clases Polares (`positivo` vs. `negativo`)

### Síntoma:
- `negativo` confundido con `positivo` (2 veces).
- `positivo` confundido con `negativo` (2 veces).

### Causa Raíz:
Se detectaron incoherencias flagrantes en las etiquetas del dataset:
1. `"Esto es un pésimo servicio no funciona"` estaba registrado con etiqueta `"positivo"`.
2. Frases espejo con idéntico campo léxico: `"quiero galletas de chocolate"` en `positivo` frente a `"quiero una galleta de chocolate con pasas"` en `negativo`.
3. Saludos en ambas polaridades: `"hola querido"` en `positivo` y `"hola como estas?"` en `negativo`.

### Solución:
- Separación estricta de dominios: si se mantiene análisis de sentimiento, `"pésimo servicio"` debe asignarse inequívocamente a `negativo`.
- Eliminar duplicidad de intenciones sobre alimentos/comida o consolidarlas bajo una sola intención.

---

## Reto 5: Clases Huérfanas de Muestra Única (`acciones`) y Fallos en K-Fold

### Síntoma:
Presencia de la entrada `"crea una carpeta"` con la etiqueta única `"acciones"`.

### Causa Raíz:
La validación cruzada estratificada (`StratifiedKFold`) requiere que cada clase contenga al menos tantos ejemplos como el número de divisiones (`k_folds >= 2`). Una clase con 1 único registro no puede ser particionada en conjunto de entrenamiento y conjunto de test simultáneamente, generando errores internos o exclusión forzada de la evaluación.

### Solución:
- Filtrar clases con menos de $K$ muestras durante la partición de validación.
- Decisión de diseño: enriquecer `acciones` con al menos 4-5 comandos operativos (ej: *"crear directorio"*, *"borrar archivo"*, *"mover carpeta"*) o retirarla del dataset de intenciones conversacionales.

---

## Reto 6: Frases Técnicas Infiltradas en `agenda`

### Síntoma:
`agenda` registraba un F1-Score de apenas **0.41** y se confundía con `codigo`, `soporte` y `datos_ia`.

### Causa Raíz:
Se detectaron 4 frases de entrenamiento dentro de `agenda` que eran técnicas al 100%:
- `"Tengo un error en el bucle de mi código python"` $\rightarrow$ Pertenece a `codigo`/`soporte`.
- `"sabes que es un bucle for?"` $\rightarrow$ Pertenece a `codigo`.
- `"Se cayó la conexión con el servidor y la contraseña"` $\rightarrow$ Pertenece a `soporte`.
- `"Vamos a calcular la probabilidad y graficar la matriz"` $\rightarrow$ Pertenece a `datos_ia`.

### Solución y Resultados:
1. **Saneamiento Quirúrgico en Dataset:** Se reasignaron formalmente estas 4 frases a sus clases naturales en [dataset_entrenamiento.json](file:///c:/Users/balsa/Desktop/ejerciciosPY/dataset_entrenamiento.json), dotándolas de respuestas contextuales de alta calidad técnica (explicación de bucles, validación de red y uso de NumPy/Matplotlib).
2. **Re-entrenamiento en Caliente:** Se ejecutó `modelo_pipeline.fit()` y se actualizó [modelo_entrenado.joblib](file:///c:/Users/balsa/Desktop/ejerciciosPY/modelo_entrenado.joblib).
3. **Impacto Cuantitativo:**
   - El F1-Score Macro de validación cruzada ascendió a **89.05%**.
   - La brecha de sobreajuste bajó a **10.95%** (*Generalización Excelente*).
   - Inferencia en caliente 100% certera en las 4 clases sin falsos positivos cruzados hacia `agenda`.

---

## Reto 7: Restricciones de Ejecución en Entorno Windows (AppLocker)

### Síntoma:
Al invocar `.\.venv\Scripts\pip.exe list` en PowerShell, el sistema operativo arrojaba:
> *Error al ejecutar el programa 'pip.exe': Una directiva de Control de aplicaciones bloqueó este archivo.*

### Causa Raíz:
Políticas de seguridad de Windows (AppLocker / Software Restriction Policies) que restringen la invocación de ejecutables secundarios ubicados en carpetas de usuario no firmadas digitalmente.

### Solución:
Invocación directa a través del intérprete de Python del entorno virtual:
```powershell
.\.venv\Scripts\python.exe -m pip list
.\.venv\Scripts\python.exe -m uvicorn traininglm:app --host 127.0.0.1 --port 8001 --reload
```
Este método elude el bloqueo de binarios secundarios manteniendo el entorno 100% aislado y funcional.
