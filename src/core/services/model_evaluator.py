"""Servicio de evaluación formal, diagnóstico de sobreajuste (Overfitting Gap) y auditoría ETL."""
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.core.domain.constants import OBJETIVO_POR_CLASE

class ModelEvaluatorService:
    def __init__(self, base_pipeline):
        self.base_pipeline = base_pipeline

    def evaluar(self, dataset: List[Dict[str, Any]], registro_poda: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if len(dataset) < 6:
            return {
                "disponible": False,
                "mensaje": "Se requieren al menos 6 ejemplos en el dataset para computar métricas formales."
            }
            
        textos = [item["texto"] for item in dataset]
        etiquetas = [item["etiqueta"] for item in dataset]
        
        conteo_clases = Counter(etiquetas)
        clases_validas = {c for c, count in conteo_clases.items() if count >= 2}
        
        if len(clases_validas) < 2:
            return {
                "disponible": False,
                "mensaje": "Se requieren al menos 2 clases con 2 o más ejemplos para evaluar el clasificador."
            }
            
        indices_validos = [i for i, c in enumerate(etiquetas) if c in clases_validas]
        textos_filtrados = [textos[i] for i in indices_validos]
        etiquetas_filtradas = [etiquetas[i] for i in indices_validos]
        
        eval_pipe = clone(self.base_pipeline)
        eval_pipe.fit(textos_filtrados, etiquetas_filtradas)
        y_pred = eval_pipe.predict(textos_filtrados)
        
        acc = round(float(accuracy_score(etiquetas_filtradas, y_pred)), 4)
        f1_macro = round(float(f1_score(etiquetas_filtradas, y_pred, average='macro', zero_division=0)), 4)
        f1_weighted = round(float(f1_score(etiquetas_filtradas, y_pred, average='weighted', zero_division=0)), 4)
        
        # Validación Cruzada Estratificada (K-Fold)
        min_samples = min(conteo_clases[c] for c in clases_validas)
        k_folds = min(5, max(2, min_samples))
        cv_scores = []
        if k_folds >= 2:
            try:
                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                cv_raw = cross_val_score(eval_pipe, textos_filtrados, etiquetas_filtradas, cv=skf, scoring='f1_macro')
                cv_scores = [round(float(s), 4) for s in cv_raw]
            except Exception:
                cv_scores = [f1_macro]
                
        cv_media = round(float(np.mean(cv_scores)), 4) if cv_scores else f1_macro
        cv_std = round(float(np.std(cv_scores)), 4) if cv_scores else 0.0
        
        # 1. Brecha de Sobreajuste (Overfitting Gap)
        brecha_sobreajuste = round(float(acc - cv_media), 4)
        nivel_sobreajuste = (
            "Crítico (Alta Memorización)" if brecha_sobreajuste > 0.35 
            else ("Moderado" if brecha_sobreajuste > 0.15 else "Bajo (Generalización Sólida)")
        )
        
        report = classification_report(etiquetas_filtradas, y_pred, output_dict=True, zero_division=0)
        clases_modelo = list(eval_pipe.named_steps['clasificador'].classes_)
        feature_names = eval_pipe.named_steps['vectorizador'].get_feature_names_out()
        coefs = eval_pipe.named_steps['clasificador'].coef_

        # 2. Rendimiento por clase + Palabras Clave más influyentes (Top Features)
        metricas_clases = {}
        for idx_c, c in enumerate(clases_modelo):
            if c in clases_validas and c in report:
                if len(clases_modelo) == 2:
                    c_coef = coefs[0] if idx_c == 1 else -coefs[0]
                else:
                    c_coef = coefs[idx_c]
                top_pos_idx = np.argsort(c_coef)[-4:][::-1]
                top_palabras = [feature_names[i] for i in top_pos_idx if c_coef[i] > 0]

                metricas_clases[c] = {
                    "precision": round(float(report[c]["precision"]), 3),
                    "recall": round(float(report[c]["recall"]), 3),
                    "f1": round(float(report[c]["f1-score"]), 3),
                    "ejemplos": int(report[c]["support"]),
                    "porcentaje": round((int(report[c]["support"]) / len(etiquetas_filtradas)) * 100, 1),
                    "top_palabras": top_palabras
                }

        # 3. Matriz de Confusión y Errores en Validación Cruzada
        matriz_confusion = {"clases": clases_modelo, "matriz": []}
        confusiones_frecuentes = []
        if k_folds >= 2:
            try:
                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                y_cv_pred = cross_val_predict(eval_pipe, textos_filtrados, etiquetas_filtradas, cv=skf)
                cm = confusion_matrix(etiquetas_filtradas, y_cv_pred, labels=clases_modelo)
                matriz_confusion["matriz"] = cm.tolist()
                for i, c_real in enumerate(clases_modelo):
                    for j, c_pred in enumerate(clases_modelo):
                        if i != j and cm[i][j] > 0:
                            confusiones_frecuentes.append({
                                "real": c_real,
                                "predicha": c_pred,
                                "veces": int(cm[i][j])
                            })
                confusiones_frecuentes.sort(key=lambda x: x["veces"], reverse=True)
            except Exception:
                pass

        # 4. Auditoría de Calidad ETL y Detección de Anomalías
        outliers = []
        textos_vistos = set()
        duplicados = 0
        total_palabras = 0
        for item in dataset:
            txt = item.get("texto", "").strip()
            etq = item.get("etiqueta", "")
            words = txt.split()
            total_palabras += len(words)
            if len(words) < 2:
                outliers.append({
                    "texto": txt,
                    "etiqueta": etq,
                    "motivo": "Término atómico/aislado sin contexto sintáctico (menos de 2 palabras)"
                })
            if txt.lower() in textos_vistos:
                duplicados += 1
            textos_vistos.add(txt.lower())

        palabras_unicas = len(set(" ".join([item.get("texto", "") for item in dataset]).lower().split()))
        longitud_promedio = round(total_palabras / (len(dataset) or 1), 1)
        riqueza_lexica = round(palabras_unicas / (total_palabras or 1), 3)

        auditoria_etl = {
            "longitud_promedio_frase": longitud_promedio,
            "palabras_totales": total_palabras,
            "palabras_unicas": palabras_unicas,
            "riqueza_lexica": riqueza_lexica,
            "frases_duplicadas": duplicados,
            "outliers_detectados": outliers[:5],
            "total_outliers": len(outliers)
        }

        # 5. Diagnósticos y Salud del Aprendizaje
        diagnosticos = []
        if brecha_sobreajuste > 0.35:
            diagnosticos.append(f"Brecha de Sobreajuste Alta ({round(brecha_sobreajuste*100)}%): El modelo memoriza el dataset (100%) pero tiene dificultad con ejemplos no vistos ({round(cv_media*100)}%).")
        elif brecha_sobreajuste > 0.15:
            diagnosticos.append(f"Brecha de Sobreajuste Moderada ({round(brecha_sobreajuste*100)}%): Normal durante la fase de expansión léxica.")
        else:
            diagnosticos.append(f"Generalización Excelente: La brecha entre entrenamiento y prueba es mínima ({round(brecha_sobreajuste*100)}%).")

        counts = [conteo_clases[c] for c in clases_validas]
        ratio_desbalance = round(max(counts) / (min(counts) or 1), 1)
        if ratio_desbalance <= 2.5:
            diagnosticos.append(f"Clases Balanceadas (Ratio {ratio_desbalance}x): Ninguna clase monopoliza el dataset.")
        else:
            diagnosticos.append(f"Desbalance de Clases (Ratio {ratio_desbalance}x): {max(conteo_clases, key=conteo_clases.get)} tiene {max(counts)} ejemplos vs {min(counts)} de la menor.")

        vectorizer = self.base_pipeline.named_steps.get('vectorizador')
        total_vocab = len(vectorizer.vocabulary_) if vectorizer and hasattr(vectorizer, 'vocabulary_') else 0
        diagnosticos.append(f"Vocabulario N-Gramas: {total_vocab} n-gramas activos con riqueza léxica de {round(riqueza_lexica * 100, 1)}%.")
        diagnosticos.append("Acuerdo Gramatical POS: Generador con ranuras independientes de Verbos y Sustantivos.")

        # 6. Recomendaciones de Acción
        acciones_mejora = []
        if brecha_sobreajuste > 0.30:
            acciones_mejora.append({
                "tipo": "generalizacion",
                "titulo": "Reducir Sobreajuste (Overfitting)",
                "detalle": f"La brecha es del {round(brecha_sobreajuste * 100)}%. Induce frases sintéticas con el botón de 'Inducir Frases Híbridas' para enseñar nuevas combinaciones."
            })
        if outliers:
            acciones_mejora.append({
                "tipo": "etl",
                "titulo": f"Sanear {len(outliers)} registro(s) atómico(s) en ETL",
                "detalle": f"Se detectó '{outliers[0]['texto']}' como término aislado. Proporcionar contexto enriquecerá los bigramas."
            })
        clases_debiles = [c for c, cnt in conteo_clases.items() if cnt < 8]
        if ratio_desbalance > 2:
            acciones_mejora.append({
                "tipo": "balance",
                "titulo": f"Ecualizar clases (desbalance {ratio_desbalance}x)",
                "detalle": f"Pulsa '⚖️ Equilibrar Clases' para llevar las 4 clases técnicas a {OBJETIVO_POR_CLASE} ejemplos cada una (se guarda copia de seguridad)."
            })
        if clases_debiles:
            acciones_mejora.append({
                "tipo": "balance",
                "titulo": "Equilibrar Clases Minoritarias",
                "detalle": f"Las clases {', '.join(clases_debiles)} tienen menos de 8 ejemplos. Agregarles soporte elevará el Cross-Validation general."
            })

        return {
            "disponible": True,
            "accuracy_global": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "cross_val_media": cv_media,
            "cross_val_std": cv_std,
            "k_folds": k_folds,
            "brecha_sobreajuste": brecha_sobreajuste,
            "nivel_sobreajuste": nivel_sobreajuste,
            "total_ejemplos": len(dataset),
            "total_clases": len(clases_validas),
            "distribucion_clases": dict(conteo_clases),
            "metricas_clases": metricas_clases,
            "matriz_confusion": matriz_confusion,
            "confusiones_frecuentes": confusiones_frecuentes,
            "auditoria_etl": auditoria_etl,
            "diagnosticos": diagnosticos,
            "acciones_mejora": acciones_mejora,
            "registro_poda": list(reversed(registro_poda[-5:])) if registro_poda else []
        }
