"""Motor NLP: Pipeline TF-IDF + Regresión Logística, inferencia explicable (XAI) y derivación de respuestas."""
import re
import random
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.core.domain.constants import get_tokens_limpios
from src.core.domain.entities import ResultadoExplicacion, PrediccionSimple

class NLPEngine:
    def __init__(self, pipeline: Optional[Pipeline] = None):
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = Pipeline([
                ('vectorizador', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.85, min_df=1)),
                ('clasificador', LogisticRegression(class_weight='balanced', C=0.6, max_iter=1000, solver='lbfgs'))
            ])
        self.esta_entrenado = False
        self.historico_dataset: List[Dict[str, Any]] = []

    def fit(self, dataset: List[Dict[str, Any]]) -> None:
        """Entrena el clasificador con los ejemplos provistos."""
        if len(dataset) < 2:
            return
        textos = [item["texto"] for item in dataset]
        etiquetas = [item["etiqueta"] for item in dataset]
        self.pipeline.fit(textos, etiquetas)
        self.esta_entrenado = True
        self.historico_dataset = list(dataset)

    def predict(self, texto: str) -> PrediccionSimple:
        if not self.esta_entrenado:
            raise RuntimeError("El modelo aún no está entrenado.")
        prediccion = self.pipeline.predict([texto])[0]
        probabilidades = self.pipeline.predict_proba([texto])[0]
        return PrediccionSimple(
            texto_original=texto,
            prediccion=str(prediccion),
            confianza=round(float(max(probabilidades)), 4)
        )

    def predecir_respuesta_desde_dataset(
        self, 
        texto_usuario: str, 
        intencion: str, 
        palabras_reconocidas: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Predice la respuesta adecuada buscando en el dataset los ejemplos aprendidos más cercanos."""
        if not self.historico_dataset:
            return "He procesado tu frase en memoria.", "sin_datos"
            
        ejemplos_clase = [item for item in self.historico_dataset if item.get("etiqueta") == intencion and item.get("respuesta")]
        if not ejemplos_clase:
            ejemplos_clase = [item for item in self.historico_dataset if item.get("respuesta")]
            
        tokens_usuario = set(re.findall(r'\b[a-záéíóúñA-ZÁÉÍÓÚÑ0-9_]+\b', texto_usuario.lower()))
        
        mejor_score = -1
        mejor_ejemplo_texto = ""
        mejor_respuesta = ""
        
        for item in ejemplos_clase:
            txt = item.get("texto", "")
            tokens_item = set(re.findall(r'\b[a-záéíóúñA-ZÁÉÍÓÚÑ0-9_]+\b', txt.lower()))
            coincidencias = tokens_usuario.intersection(tokens_item)
            score = len(coincidencias)
            if score > mejor_score:
                mejor_score = score
                mejor_ejemplo_texto = txt
                mejor_respuesta = item.get("respuesta", "")
                
        if mejor_respuesta and mejor_score > 0:
            origen = f"Predicción derivada del ejemplo aprendido: \"{mejor_ejemplo_texto}\" (coincidencia: {mejor_score} términos)"
            return mejor_respuesta, origen
            
        if ejemplos_clase:
            ejemplo_reciente = random.choice(ejemplos_clase[-10:])
            origen = f"Predicción generalizada a partir del corpus de la clase '{intencion}'"
            return ejemplo_reciente.get("respuesta"), origen
            
        return f"He clasificado tu consulta en '{intencion}' basándome en mi vocabulario.", "general"

    def explicar_y_responder(self, texto: str) -> ResultadoExplicacion:
        if not self.esta_entrenado:
            raise RuntimeError("El modelo aún no está entrenado.")
            
        vectorizer = self.pipeline.named_steps['vectorizador']
        classifier = self.pipeline.named_steps['clasificador']
        
        # 1. Inferencia de Intención y Confianza
        intencion = str(self.pipeline.predict([texto])[0])
        probabilidades = self.pipeline.predict_proba([texto])[0]
        confianza = round(float(max(probabilidades)), 4)
        clases = list(classifier.classes_)
        idx_clase_predicha = clases.index(intencion)
        
        # 2. Análisis de Términos
        analyzer = vectorizer.build_analyzer()
        tokens_frase = analyzer(texto)
        vocab = vectorizer.vocabulary_
        
        palabras_reconocidas = []
        palabras_nuevas = []
        reglas = []
        
        is_binary = (len(clases) == 2 and classifier.coef_.shape[0] == 1)
        
        for palabra in set(tokens_frase):
            if palabra in vocab:
                col_idx = vocab[palabra]
                if is_binary:
                    peso = classifier.coef_[0][col_idx] if idx_clase_predicha == 1 else -classifier.coef_[0][col_idx]
                else:
                    peso = classifier.coef_[idx_clase_predicha][col_idx]
                
                palabras_reconocidas.append({
                    "palabra": palabra,
                    "peso": round(float(peso), 4),
                    "impacto": "positivo" if peso > 0 else "negativo"
                })
            else:
                if " " not in palabra:
                    palabras_nuevas.append(palabra)
                    
        palabras_reconocidas.sort(key=lambda x: abs(x["peso"]), reverse=True)
        
        # 3. Predicción Viva de la Respuesta desde el Dataset
        respuesta_viva, regla_origen = self.predecir_respuesta_desde_dataset(texto, intencion, palabras_reconocidas)
        reglas.append(f"Regla generativa: {regla_origen}")
        
        if palabras_reconocidas:
            top_favor = [p for p in palabras_reconocidas if p["peso"] > 0][:3]
            if top_favor:
                tokens_str = ", ".join([f"\"{p['palabra']}\" (+{p['peso']})" for p in top_favor])
                reglas.append(f"Regla de activación clave: Los términos {tokens_str} inclinaron fuertemente la balanza hacia '{intencion}'.")
                
            reglas.append(f"Regla probabilística: Vectorización TF-IDF con función Softmax asignó un {round(confianza * 100, 1)}% de probabilidad a '{intencion}'.")
        else:
            reglas.append("Regla por defecto (Vocabulario desconocido): Ninguna palabra de la frase coincide con el vocabulario actual.")
            reglas.append(f"La respuesta fue inferida por sesgo probabilístico general de la clase '{intencion}'.")
            
        if palabras_nuevas:
            reglas.append(f"Términos no aprendidos detectados: {', '.join([f'\"{p}\"' for p in palabras_nuevas[:4]])}. ¡Nutre a la IA con ellos!")
            
        return ResultadoExplicacion(
            mensaje_usuario=texto,
            intencion=intencion,
            respuesta=respuesta_viva,
            confianza=confianza,
            palabras_reconocidas=palabras_reconocidas,
            palabras_nuevas=list(set(palabras_nuevas)),
            reglas=reglas,
            vocabulario_total=len(vocab)
        )
