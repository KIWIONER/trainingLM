from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import joblib
import os
import json
import random
import re
import shutil
from datetime import datetime

import numpy as np
from collections import Counter

# Librerías de Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

# ==========================================
# 1. MODELOS DE DATOS (Pydantic)
# ==========================================
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


# ==========================================
# 2. PACKS DE HABILIDADES CON PARES (PREGUNTA <-> RESPUESTA VIVA)
# ==========================================
# ==========================================
# 2. PACK DE HABILIDAD UNIFICADA (CEREBRO INTEGRAL COGNITIVO)
# ==========================================
# ==========================================
# 2. PACK DE HABILIDAD UNIFICADA (CEREBRO INTEGRAL COGNITIVO CON ACUERDO POS)
# ==========================================

VERBOS_CANONICOS = {
    "crear", "depurar", "compilar", "ejecutar", "reiniciar", "configurar", 
    "actualizar", "calcular", "graficar", "entrenar", "agendar", "organizar", 
    "planificar", "optimizar", "solucionar", "proyectar", "diagnosticar", 
    "estructurar", "conectar", "bloquear", "desplegar", "normalizar", "inspeccionar", "sincronizar"
}

def es_termino_verbo(palabra: str) -> bool:
    p = palabra.lower().strip()
    if p in VERBOS_CANONICOS:
        return True
    return p.endswith(("ar", "er", "ir")) and len(p) >= 5 and p not in {"servidor", "cluster"}


_VERBOS_BASE = [
    "crear", "depurar", "compilar", "ejecutar", "reiniciar", "configurar", 
    "actualizar", "calcular", "graficar", "entrenar", "agendar", "organizar", 
    "planificar", "optimizar", "solucionar", "proyectar", "desplegar", "sincronizar"
]

_SUSTANTIVOS_BASE = [
    # Programación & Código
    "python", "funcion", "bucle", "variable", "algoritmo", "codigo", 
    "error", "debug", "libreria", "sintaxis", "retorno", "script",
    # Soporte & Redes
    "problema", "contraseña", "conexion", "pantalla", "fallo", 
    "sistema", "servidor", "bloqueo", "firewall", "router", "proxy",
    # Datos & Machine Learning
    "probabilidad", "estadistica", "matriz", "datos", "grafico", 
    "prediccion", "dataset", "media", "analisis", "vector", "correlacion",
    # Agenda & Productividad
    "reunion", "calendario", "tarea", "recordatorio", "horario", 
    "pendiente", "prioridad", "evento", "cita", "semana", "deadline"
]

HABILIDADES_PACKS = {
    "cerebro_integral": {
        "id": "cerebro_integral",
        "nombre": "Cerebro Integral (Código, Soporte, Datos & Agenda)",
        "icono": "🧠",
        "etiqueta": "asistente",
        "verbos": list(_VERBOS_BASE),
        "sustantivos": list(_SUSTANTIVOS_BASE),
        "palabras_clave": list(_VERBOS_BASE + _SUSTANTIVOS_BASE),
        "banco_descubrimiento": [
            # Código
            "docker", "backend", "frontend", "middleware", "endpoint", "commit", 
            "repositorio", "deploy", "asincrono", "promesa", "json", "token", 
            "query", "pipeline", "clase", "objeto", "herencia", "servidor", 
            "request", "response", "modulo", "framework", "terminal", "socket", 
            "depurador", "async", "await", "decorador",
            # Soporte
            "router", "firewall", "proxy", "antivirus", "ethernet", "latencia", 
            "ping", "dns", "gateway", "ip", "subred", "interfaz", "driver", 
            "bios", "controlador", "memoria", "disco", "backup", "permisos", 
            "credenciales", "protocolo", "vpn", "ssh", "ssl", "ancho_banda",
            # Datos
            "clustering", "regresion", "outliers", "percentil", "varianza", 
            "desviacion", "tensor", "precision", "recall", "overfitting", 
            "loss", "gradiente", "distribucion", "normalizacion", "dataframe", 
            "features", "dimension", "histograma", "coeficiente", "series_temporales",
            # Agenda
            "deadline", "sprint", "backlog", "kanban", "entregable", "hito", 
            "retrospectiva", "objetivo", "notificacion", "itinerario", "bloque", 
            "descanso", "enfoque", "prioritario", "duracion", "alarma", 
            "cronograma", "planificacion", "asunto", "compromiso"
        ],
        "plantillas_pares": [
            # --- 1. Plantillas Gramaticales Híbridas (Código + Soporte + Datos + Agenda) ---
            (
                "como puedo {v} el {s1} para analizar la {s2} antes de la {s3}",
                "Para {v} el {s1} y analizar la {s2} antes de tu {s3}, valida los tipos de datos y corre pruebas unitarias.",
                "codigo"
            ),
            (
                "tengo un problema de {s1} al {v} la {s2} en el {s3}",
                "Incidencia de {s1} registrada al {v} la {s2} en tu {s3}. Te recomiendo reiniciar el proceso y revisar logs.",
                "soporte"
            ),
            (
                "necesito {v} la {s1} de datos antes de la {s2} de entrega",
                "He procedido a {v} la {s1} de datos. Los resultados quedan listos antes de la {s2}.",
                "datos_ia"
            ),
            (
                "agendar una {s1} urgente para {v} el {s2} con el equipo",
                "Reunión de {s1} agendada en tu calendario para {v} el {s2} con prioridad alta.",
                "agenda"
            ),
            (
                "como {v} la {s1} del {s2} para evitar un {s3} en el sistema",
                "Para {v} la {s1} del {s2} y evitar un {s3}, limpia la memoria en búfer y ajusta los parámetros de ejecución.",
                "soporte"
            ),
            (
                "crear un {s1} para {v} el {s2} de forma automatizada",
                "Script configurado: he creado el {s1} para {v} tu {s2} de forma periódica en segundo plano.",
                "codigo"
            ),
            (
                "como {v} los {s1} en este {s2} de machine learning",
                "Para {v} los {s1} en tu {s2}, calcula las medidas de dispersión y proyecta la matriz de correlación.",
                "datos_ia"
            ),
            (
                "organizar un {s1} semanal para {v} los pendientes de {s2}",
                "Itinerario de {s1} organizado en tu agenda para {v} las metas pendientes de {s2}.",
                "agenda"
            ),
            (
                "se cayo el {s1} y no es posible {v} la {s2} de red",
                "Ante la caída del {s1} que impide {v} la {s2}, verifica el gateway y comprueba los puertos del firewall.",
                "soporte"
            ),
            (
                "ayuda para {v} y optimizar la {s1} sin errores en {s2}",
                "Para {v} y optimizar la {s1} en {s2}, añade puntos de interrupción con el depurador y valida la memoria.",
                "codigo"
            ),
            (
                "planificar una {s1} tecnica para {v} el {s2} de datos",
                "Sesión de {s1} agendada en tu calendario para coordinar el trabajo y {v} el {s2} correctamente.",
                "agenda"
            ),
            (
                "entrenar un modelo para {v} los patrones en este {s1}",
                "Para {v} en tu {s1}, normaliza las columnas numéricas y divide en subconjuntos de entrenamiento y test.",
                "datos_ia"
            )
        ]
    }
}


# ==========================================
# 3. INICIALIZACIÓN DE LA API Y PIPELINE
# ==========================================
app = FastAPI(
    title="Master NLP Language Model & Chat Studio",
    version="2.2.0",
    description="Modelo de lenguaje estadístico predictivo donde las respuestas se derivan y predicen directamente de su dataset."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARCHIVO_MODELO = "modelo_entrenado.joblib"
ARCHIVO_DATASET = "dataset_entrenamiento.json"
ARCHIVO_BACKUP = "dataset_entrenamiento.backup.json"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Pipeline con balanceo de pesos, escalamiento logarítmico sublineal y regularización L2
modelo_pipeline = Pipeline([
    ('vectorizador', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.85, min_df=1)),
    ('clasificador', LogisticRegression(class_weight='balanced', C=0.6, max_iter=1000, solver='lbfgs'))
])

esta_entrenado = False
historico_dataset = []

DATASET_BASE = [
    {
        "texto": "como crear una funcion con bucle for en python", 
        "etiqueta": "codigo", 
        "respuesta": "Para crear una funcion con bucle en python, usa 'def nombre():' y coloca dentro 'for item in lista:'."
    },
    {
        "texto": "se cayo la conexion con el servidor y tengo error de contraseña", 
        "etiqueta": "soporte", 
        "respuesta": "Si se cayo la conexion con el servidor, reinicia el adaptador de red y restablece la contraseña de usuario."
    },
    {
        "texto": "vamos a calcular la probabilidad y graficar la matriz de datos", 
        "etiqueta": "datos_ia", 
        "respuesta": "Para calcular la probabilidad y graficar la matriz de datos, normalizamos el dataset y proyectamos la distribución."
    },
    {
        "texto": "agendar una cita o reunion en mi calendario para mañana", 
        "etiqueta": "agenda", 
        "respuesta": "Cita registrada en tu calendario para el día de mañana con recordatorio prioritario configurado."
    }
]


def get_tokens_limpios(texto: str) -> set:
    return set(re.findall(r'\b[a-záéíóúñA-ZÁÉÍÓÚÑ0-9_]+\b', texto.lower()))


def podar_dataset_conservando_vocabulario(dataset: list, max_por_clase: int = 15) -> list:
    """
    Poda inteligente con conservación del 100% del vocabulario:
    Garantiza que todas las palabras únicas sigan presentes en al menos un ejemplo,
    eliminando únicamente frases duplicadas o redundantes para evitar el crecimiento excesivo del JSON.
    """
    if len(dataset) <= 30:
        return dataset

    # 1. Identificar todo el vocabulario existente
    vocabulario_total = set()
    for item in dataset:
        vocabulario_total.update(get_tokens_limpios(item.get("texto", "")))

    palabras_cubiertas = set()
    seleccionados = []
    textos_vistos = set()

    # Agrupar por clase
    por_clase = {}
    for item in dataset:
        c = item.get("etiqueta", "general")
        por_clase.setdefault(c, []).append(item)

    # 2. Selección codiciosa priorizando frases que aportan palabras nuevas o mantienen balance
    for c, items in por_clase.items():
        contador_clase = 0
        for item in reversed(items): # Priorizar frases más recientes
            txt = item.get("texto", "").strip().lower()
            if txt in textos_vistos:
                continue
            toks = get_tokens_limpios(txt)
            palabras_nuevas = toks - palabras_cubiertas

            # Si aporta palabras no cubiertas aún O no alcanzamos la cuota base de la clase
            if palabras_nuevas or contador_clase < max_por_clase:
                seleccionados.append(item)
                textos_vistos.add(txt)
                palabras_cubiertas.update(toks)
                contador_clase += 1

    return seleccionados


# ==========================================
# REGULACIÓN DE CLASES (Ecualizador 1:1:1:1 + Cuota Máxima)
# ==========================================
CLASES_TECNICAS = ["codigo", "agenda", "datos_ia", "soporte"]
OBJETIVO_POR_CLASE = 25
CUOTA_MAXIMA_CLASE = 0.30


def seleccionar_representativos(items: list, n: int) -> list:
    """Elige n ejemplos de una clase priorizando los recientes que aportan vocabulario nuevo."""
    if len(items) <= n:
        return list(items)
    elegidos = []
    elegidos_set = set()
    cubiertas = set()
    vistos = set()
    recientes = list(range(len(items) - 1, -1, -1))
    # Pasada 1: frases que aportan palabras todavía no cubiertas
    for i in recientes:
        if len(elegidos) >= n:
            break
        txt = items[i].get("texto", "").strip().lower()
        if txt in vistos:
            continue
        toks = get_tokens_limpios(txt)
        if toks - cubiertas:
            elegidos.append(i)
            elegidos_set.add(i)
            cubiertas.update(toks)
            vistos.add(txt)
    # Pasada 2: completar la cuota con las más recientes restantes
    for i in recientes:
        if len(elegidos) >= n:
            break
        txt = items[i].get("texto", "").strip().lower()
        if i not in elegidos_set and txt not in vistos:
            elegidos.append(i)
            elegidos_set.add(i)
            vistos.add(txt)
    return [items[i] for i in sorted(elegidos)]


def generar_frases_para_clase(clase: str, cantidad: int, textos_existentes: set) -> list:
    """Genera frases sintéticas (con acuerdo POS) de una clase concreta, sin duplicados."""
    pack = HABILIDADES_PACKS["cerebro_integral"]
    plantillas = [p for p in pack["plantillas_pares"] if len(p) == 3 and p[2] == clase]
    if not plantillas or cantidad <= 0:
        return []
    verbos = pack.get("verbos") or list(_VERBOS_BASE)
    sustantivos = pack.get("sustantivos") or list(_SUSTANTIVOS_BASE)
    nuevas = []
    intentos = 0
    while len(nuevas) < cantidad and intentos < cantidad * 30:
        intentos += 1
        preg_t, resp_t, _ = random.choice(plantillas)
        s = random.sample(sustantivos, 3)
        kw = {"v": random.choice(verbos), "s1": s[0], "s2": s[1], "s3": s[2],
              "p1": s[0], "p2": s[1], "p3": s[2]}
        preg = preg_t.format(**kw)
        if preg.lower() in textos_existentes:
            continue
        textos_existentes.add(preg.lower())
        nuevas.append({"texto": preg, "etiqueta": clase, "respuesta": resp_t.format(**kw), "origen": "sintetico"})
    return nuevas


def equilibrar_dataset(dataset: list, objetivo: int = OBJETIVO_POR_CLASE):
    """Lleva cada clase técnica exactamente a `objetivo` ejemplos (poda o síntesis)."""
    por_clase = {}
    for item in dataset:
        por_clase.setdefault(item.get("etiqueta"), []).append(item)
    textos_existentes = {item.get("texto", "").strip().lower() for item in dataset}

    resultado, informe = [], {}
    for clase in CLASES_TECNICAS:
        items = por_clase.get(clase, [])
        antes = len(items)
        if antes > objetivo:
            seleccion, sinteticas = seleccionar_representativos(items, objetivo), []
        else:
            seleccion = list(items)
            sinteticas = generar_frases_para_clase(clase, objetivo - antes, textos_existentes)
        resultado.extend(seleccion + sinteticas)
        informe[clase] = {
            "antes": antes,
            "despues": len(seleccion) + len(sinteticas),
            "podadas": antes - len(seleccion),
            "sinteticas": len(sinteticas)
        }
    fuera_de_dominio = sum(len(v) for k, v in por_clase.items() if k not in CLASES_TECNICAS)

    # Rescate de vocabulario: las palabras que desaparecerían al podar se reinyectan
    resultado, rescatadas, perdidas = rescatar_vocabulario(dataset, resultado)
    registrar_poda({
        "origen": "ecualizador",
        "clases": informe,
        "frases_podadas": sum(v["podadas"] for v in informe.values()),
        "palabras_rescatadas": rescatadas,
        "palabras_perdidas": perdidas
    })
    return resultado, informe, fuera_de_dominio


def aplicar_cuota_maxima(dataset: list, cuota: float = CUOTA_MAXIMA_CLASE):
    """Regla de ingesta: ninguna clase puede superar `cuota` del dataset total.
    Devuelve (dataset_resultante, registro_de_poda | None)."""
    conteo = Counter(item.get("etiqueta") for item in dataset)
    if len(conteo) < 4:  # con menos de 4 clases una cuota del 30% es inalcanzable
        return dataset, None
    resultado = list(dataset)
    clases_podadas = {}
    for clase, n in conteo.most_common():
        resto = len(resultado) - n
        limite = max(int(resto * cuota / (1 - cuota)), 10)
        if n > limite:
            items = [x for x in resultado if x.get("etiqueta") == clase]
            conservar = {id(x) for x in seleccionar_representativos(items, limite)}
            resultado = [x for x in resultado if x.get("etiqueta") != clase or id(x) in conservar]
            clases_podadas[clase] = {"antes": n, "despues": limite, "podadas": n - limite, "sinteticas": 0}
    if not clases_podadas:
        return dataset, None
    resultado, rescatadas, perdidas = rescatar_vocabulario(dataset, resultado)
    return resultado, {
        "origen": "cuota_30",
        "clases": clases_podadas,
        "frases_podadas": sum(v["podadas"] for v in clases_podadas.values()),
        "palabras_rescatadas": rescatadas,
        "palabras_perdidas": perdidas
    }


def vocabulario_de(dataset: list) -> set:
    vocab = set()
    for item in dataset:
        vocab.update(get_tokens_limpios(item.get("texto", "")))
    return vocab


# ------------------------------------------
# Rescate de vocabulario y registro de gobernanza de la poda
# ------------------------------------------
ARCHIVO_REGISTRO_PODA = "registro_poda.jsonl"
PALABRAS_VACIAS = {
    "me", "te", "se", "de", "la", "el", "los", "las", "un", "una", "y", "o", "en",
    "para", "por", "con", "que", "como", "mi", "tu", "es", "al", "del", "lo", "no", "si"
}


def _plantilla_a_regex(plantilla: str):
    partes = re.split(r"\{(?:v|s1|s2|s3|p1|p2|p3)\}", plantilla)
    return re.compile("^" + r"\S+".join(re.escape(p) for p in partes) + "$", re.IGNORECASE)


_REGEX_SINTETICAS = [_plantilla_a_regex(p[0]) for p in HABILIDADES_PACKS["cerebro_integral"]["plantillas_pares"]]


def es_relleno_sintetico(item: dict) -> bool:
    """Frase generada por plantilla y sustituible (las 'portadoras' de rescate no lo son)."""
    if item.get("origen") == "portadora":
        return False
    if item.get("origen") == "sintetico":
        return True
    txt = item.get("texto", "").strip()
    return any(r.match(txt) for r in _REGEX_SINTETICAS)


def rescatar_vocabulario(original: list, nuevo: list):
    """Recupera las palabras que una poda haría desaparecer del dataset SIN cambiar el tamaño
    de cada clase: sustituye frases sintéticas de relleno por
      1) frases originales que contienen palabras que no encajan en plantillas, y
      2) frases 'portadoras' (plantilla + acuerdo POS) con las palabras encajables.
    Solo si una clase no tiene relleno suficiente se añaden frases extra.
    Devuelve (dataset, palabras_rescatadas, palabras_aun_perdidas)."""
    perdidas = vocabulario_de(original) - vocabulario_de(nuevo) - PALABRAS_VACIAS
    if not perdidas:
        return nuevo, [], []

    # Clase de origen de cada palabra perdida (donde más aparecía)
    origen = {}
    for w in perdidas:
        votos = Counter(x.get("etiqueta") for x in original if w in get_tokens_limpios(x.get("texto", "")))
        origen[w] = votos.most_common(1)[0][0]

    pack = HABILIDADES_PACKS["cerebro_integral"]
    lexico = set(pack.get("banco_descubrimiento", [])) | set(pack.get("sustantivos", [])) | set(pack.get("verbos", []))
    resultado = list(nuevo)
    textos = {x.get("texto", "").strip().lower() for x in resultado}

    for clase in sorted(set(origen.values())):
        palabras = {w for w, c in origen.items() if c == clase}
        encajables = {w for w in palabras if w in lexico or es_termino_verbo(w)}
        pendientes_orig = palabras - encajables
        nuevas = []

        # 1) Frases originales para las palabras que no encajan en una plantilla
        candidatos = [x for x in original if x.get("etiqueta") == clase
                      and x.get("texto", "").strip().lower() not in textos]
        while pendientes_orig and candidatos:
            mejor = max(candidatos, key=lambda x: len(get_tokens_limpios(x["texto"]) & pendientes_orig))
            toks = get_tokens_limpios(mejor["texto"])
            if not toks & pendientes_orig:
                break
            nuevas.append(mejor)
            candidatos.remove(mejor)
            textos.add(mejor["texto"].strip().lower())
            pendientes_orig -= toks
            encajables -= toks

        # 2) Frases portadoras con las palabras encajables (1 verbo + 3 sustantivos por frase)
        plantillas = [p for p in pack["plantillas_pares"] if len(p) == 3 and p[2] == clase]
        verbos_p = sorted(w for w in encajables if es_termino_verbo(w))
        sust_p = sorted(w for w in encajables if not es_termino_verbo(w))
        relleno_s = [s for s in pack.get("sustantivos", []) if s not in sust_p] or list(_SUSTANTIVOS_BASE)
        relleno_v = pack.get("verbos") or list(_VERBOS_BASE)
        while plantillas and (verbos_p or sust_p):
            preg_t, resp_t, _ = random.choice(plantillas)
            v = verbos_p.pop() if verbos_p else random.choice(relleno_v)
            s = [sust_p.pop() if sust_p else random.choice(relleno_s) for _ in range(3)]
            kw = {"v": v, "s1": s[0], "s2": s[1], "s3": s[2], "p1": s[0], "p2": s[1], "p3": s[2]}
            nuevas.append({"texto": preg_t.format(**kw), "etiqueta": clase,
                           "respuesta": resp_t.format(**kw), "origen": "portadora"})

        # 3) Colocar sustituyendo relleno sintético (el que menos vocabulario único aporta primero)
        frecuencia = Counter()
        for x in resultado:
            frecuencia.update(get_tokens_limpios(x.get("texto", "")))
        huecos = [i for i, x in enumerate(resultado) if x.get("etiqueta") == clase and es_relleno_sintetico(x)]
        huecos.sort(key=lambda i: -sum(1 for t in get_tokens_limpios(resultado[i]["texto"]) if frecuencia[t] == 1))
        for item in nuevas:
            if huecos:
                resultado[huecos.pop()] = item
            else:
                resultado.append(item)

    aun_perdidas = sorted(vocabulario_de(original) - vocabulario_de(resultado) - PALABRAS_VACIAS)
    rescatadas = sorted(perdidas - set(aun_perdidas))
    return resultado, rescatadas, aun_perdidas


def _cargar_registro_poda(maximo: int = 20) -> list:
    if not os.path.exists(ARCHIVO_REGISTRO_PODA):
        return []
    try:
        with open(ARCHIVO_REGISTRO_PODA, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f.readlines()[-maximo:] if l.strip()]
    except Exception:
        return []


REGISTRO_PODA = _cargar_registro_poda()


def registrar_poda(entrada: dict):
    """Gobernanza de datos: deja constancia de cada eliminación (qué, cuánto y qué se perdió)."""
    entrada = {"fecha": datetime.now().isoformat(timespec="seconds"), **entrada}
    REGISTRO_PODA.append(entrada)
    del REGISTRO_PODA[:-20]
    try:
        with open(ARCHIVO_REGISTRO_PODA, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception:
        pass


def entrenar_con_dataset(dataset):
    global esta_entrenado, modelo_pipeline, historico_dataset
    if len(dataset) < 2:
        return

    # Cuota máxima por clase (con rescate de vocabulario y registro auditado)
    dataset, registro = aplicar_cuota_maxima(dataset)
    if registro:
        registrar_poda(registro)

    textos = [item["texto"] for item in dataset]
    etiquetas = [item["etiqueta"] for item in dataset]
    modelo_pipeline.fit(textos, etiquetas)
    esta_entrenado = True
    joblib.dump(modelo_pipeline, ARCHIVO_MODELO)
    
    with open(ARCHIVO_DATASET, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    historico_dataset = dataset


# Limpieza y actualización de respuestas genéricas obsoletas
def sanear_dataset_existente(dataset):
    frases_genericas = [
        "Módulo de soporte técnico: te sugiero verificar tu conexión de red",
        "Módulo de agenda y productividad: tarea agregada a tu itinerario",
        "Módulo de datos: aplicaremos matrices de dispersión",
        "Módulo de programación activado: he analizado tu consulta",
        "¡Entendido! Ya sé cómo responder"
    ]
    modificados = 0
    for item in dataset:
        resp = item.get("respuesta", "")
        # Si tiene una respuesta genérica repetitiva, derivarle una respuesta viva basada en su texto
        if any(gen in resp for gen in frases_genericas):
            txt = item.get("texto", "")
            tokens = re.findall(r'\b[a-záéíóúñA-ZÁÉÍÓÚÑ0-9_]+\b', txt)
            t_str = ", ".join(tokens[:3]) if tokens else "tu consulta"
            etq = item.get("etiqueta", "general")
            
            if etq == "agenda":
                item["respuesta"] = f"He registrado la solicitud de agenda para '{t_str}' en tu calendario con alerta activa."
            elif etq == "soporte":
                item["respuesta"] = f"Diagnóstico de soporte para '{t_str}': te sugiero reiniciar el servicio y verificar credenciales de red."
            elif etq == "codigo":
                item["respuesta"] = f"Para programar '{t_str}' en tu script, valida la sintaxis, tipos y lógica del bloque de ejecución."
            elif etq == "datos_ia":
                item["respuesta"] = f"Para analizar '{t_str}', calculamos la matriz muestral y evaluamos la probabilidad en el dataset."
            else:
                item["respuesta"] = f"Procesado: he adaptado la respuesta para '{t_str}' según el vocabulario de esta clase."
            modificados += 1
    return dataset


# Cargar dataset existente
if os.path.exists(ARCHIVO_DATASET):
    try:
        with open(ARCHIVO_DATASET, "r", encoding="utf-8") as f:
            historico_dataset = json.load(f)
        historico_dataset = sanear_dataset_existente(historico_dataset)
        entrenar_con_dataset(historico_dataset)
    except Exception:
        entrenar_con_dataset(DATASET_BASE)
else:
    entrenar_con_dataset(DATASET_BASE)


# ==========================================
# 4. MOTOR PREDICTIVO DE RESPUESTAS VIVAS
# ==========================================
def sintetizar_frases_de_habilidad(habilidad_id: str = "cerebro_integral", cantidad: int = 3, permitir_descubrimiento: bool = True):
    pack = HABILIDADES_PACKS.get(habilidad_id) or list(HABILIDADES_PACKS.values())[0]
        
    pares = []
    palabras = pack["palabras_clave"]
    verbos = pack.get("verbos", list(_VERBOS_BASE))
    sustantivos = pack.get("sustantivos", list(_SUSTANTIVOS_BASE))
    banco = pack.get("banco_descubrimiento", [])
    plantillas_pares = pack["plantillas_pares"]
    
    palabras_descubiertas = []
    
    # 1. Capacidad del generador automático de descubrir e introducir palabras nuevas al dataset
    if permitir_descubrimiento and banco:
        candidatas = [w for w in banco if w not in palabras]
        if candidatas:
            num_a_descubrir = min(len(candidatas), random.choice([1, 2]))
            nuevas = random.sample(candidatas, num_a_descubrir)
            for n in nuevas:
                palabras.append(n)
                palabras_descubiertas.append(n)
                if es_termino_verbo(n):
                    if n not in verbos:
                        verbos.append(n)
                else:
                    if n not in sustantivos:
                        sustantivos.append(n)
                
    for i in range(cantidad):
        plantilla_item = random.choice(plantillas_pares)
        if len(plantilla_item) == 3:
            plantilla_preg, plantilla_resp, etq_item = plantilla_item
        else:
            plantilla_preg, plantilla_resp = plantilla_item
            etq_item = pack.get("etiqueta", "codigo")
            
        verbo_sel = random.choice(verbos) if verbos else "analizar"
        sust_sel = random.sample(sustantivos, min(4, len(sustantivos))) if sustantivos else ["sistema", "proceso", "datos", "script"]
        
        # Inyección gramaticalmente correcta de palabras descubiertas
        if palabras_descubiertas and i == 0:
            protagonista = palabras_descubiertas[0]
            if es_termino_verbo(protagonista):
                verbo_sel = protagonista
            else:
                sust_sel[0] = protagonista
                
        format_kwargs = {
            "v": verbo_sel,
            "s1": sust_sel[0],
            "s2": sust_sel[1] if len(sust_sel) > 1 else sust_sel[0],
            "s3": sust_sel[2] if len(sust_sel) > 2 else sust_sel[0],
            "p1": sust_sel[0],
            "p2": sust_sel[1] if len(sust_sel) > 1 else sust_sel[0],
            "p3": sust_sel[2] if len(sust_sel) > 2 else sust_sel[0]
        }
        
        frase = plantilla_preg.format(**format_kwargs)
        resp = plantilla_resp.format(**format_kwargs)
        pares.append((frase, resp, etq_item))
        
    return pares, pack.get("etiqueta", "codigo"), palabras_descubiertas





def predecir_respuesta_desde_dataset(texto_usuario: str, intencion: str, palabras_reconocidas: List[PalabraImpacto]) -> Tuple[str, str]:
    """
    Predice la respuesta adecuada buscando en el dataset los ejemplos aprendidos más cercanos
    en función de las palabras clave reconocidas del usuario.
    """
    global historico_dataset
    if not historico_dataset:
        return "He procesado tu frase en memoria.", "sin_datos"
        
    # Filtrar ejemplos de la misma intención predicha
    ejemplos_clase = [item for item in historico_dataset if item.get("etiqueta") == intencion and item.get("respuesta")]
    if not ejemplos_clase:
        ejemplos_clase = [item for item in historico_dataset if item.get("respuesta")]
        
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
        
    # Si no hubo coincidencia exacta de palabras, toma un ejemplo representativo reciente de esa clase
    if ejemplos_clase:
        ejemplo_reciente = random.choice(ejemplos_clase[-10:])
        origen = f"Predicción generalizada a partir del corpus de la clase '{intencion}'"
        return ejemplo_reciente.get("respuesta"), origen
        
    return f"He clasificado tu consulta en '{intencion}' basándome en mi vocabulario.", "general"


def explicar_y_responder(texto: str) -> Dict[str, Any]:
    global esta_entrenado, modelo_pipeline
    
    if not esta_entrenado:
        raise HTTPException(status_code=400, detail="El modelo aún no está entrenado.")
        
    vectorizer = modelo_pipeline.named_steps['vectorizador']
    classifier = modelo_pipeline.named_steps['clasificador']
    
    # 1. Inferencia de Intención y Confianza
    intencion = str(modelo_pipeline.predict([texto])[0])
    probabilidades = modelo_pipeline.predict_proba([texto])[0]
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
    respuesta_viva, regla_origen = predecir_respuesta_desde_dataset(texto, intencion, palabras_reconocidas)
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
        
    return {
        "mensaje_usuario": texto,
        "intencion": intencion,
        "respuesta": respuesta_viva,
        "confianza": confianza,
        "palabras_reconocidas": palabras_reconocidas,
        "palabras_nuevas": list(set(palabras_nuevas)),
        "reglas": reglas,
        "vocabulario_total": len(vocab)
    }


# ==========================================
# 5. ENDPOINTS DE LA API
# ==========================================

# 1. Chat interactivo
@app.post("/v1/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chatear_con_ia(payload: ChatMessageRequest):
    return explicar_y_responder(payload.mensaje)


# 2. Catálogo de Habilidades Temáticas
@app.get("/v1/skills", status_code=status.HTTP_200_OK)
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


# 3. Inducción Temática Automática (Generador de Pares Pregunta <-> Respuesta Viva)
@app.post("/v1/induce-training", status_code=status.HTTP_200_OK)
async def inducir_entrenamiento(payload: InducirEntrenamientoRequest):
    global historico_dataset
    try:
        pares_sinteticos, etiqueta, descubiertas = sintetizar_frases_de_habilidad(
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

    historico_dataset.extend(nuevos_items)
    entrenar_con_dataset(historico_dataset)
    
    vocab_len = len(modelo_pipeline.named_steps['vectorizador'].vocabulary_)
    
    return {
        "mensaje": f"¡Inducción completada! Se sintetizaron y aprendieron {len(pares_sinteticos)} pares únicos de pregunta y respuesta.",
        "habilidad_id": payload.habilidad_id,
        "etiqueta": etiqueta,
        "palabras_descubiertas": descubiertas,
        "frases_generadas": [p[0] for p in pares_sinteticos],
        "respuestas_generadas": [p[1] for p in pares_sinteticos],
        "ejemplos_totales": len(historico_dataset),
        "vocabulario_aprendido": vocab_len
    }


# 4. Nutrir manualmente con frase y respuesta personalizada
@app.post("/v1/teach", status_code=status.HTTP_200_OK)
async def nutrir_modelo(payload: NutrirRequest):
    global historico_dataset
    nuevo_item = {
        "texto": payload.texto.strip(),
        "etiqueta": payload.etiqueta.strip(),
        "respuesta": payload.respuesta.strip() if payload.respuesta else f"Entendido. He incorporado el conocimiento de '{payload.texto.strip()}' a mi clase '{payload.etiqueta.strip()}'."
    }
        
    historico_dataset.append(nuevo_item)
    
    # Sincronizar también las palabras sustantivas de la frase con el pack unificado
    stopwords_es = {"como", "para", "este", "esta", "estos", "estas", "tengo", "quiero", "necesito", "puedo", "hacer", "crear", "sobre", "entre", "desde", "hasta", "donde", "cuando", "porque", "pero", "tiene", "tienen"}
    tokens_nuevos = get_tokens_limpios(payload.texto)
    for p_id, p_data in HABILIDADES_PACKS.items():
        for t in tokens_nuevos:
            if len(t) >= 4 and t not in stopwords_es and t not in p_data["palabras_clave"]:
                p_data["palabras_clave"].append(t)

    entrenar_con_dataset(historico_dataset)
    
    vocab_len = len(modelo_pipeline.named_steps['vectorizador'].vocabulary_)
    return {
        "mensaje": f"¡Aprendizaje exitoso! He incorporado la frase y su respuesta a mis patrones de predicción.",
        "ejemplos_totales": len(historico_dataset),
        "vocabulario_aprendido": vocab_len,
        "etiqueta_asociada": payload.etiqueta
    }


# 5. Estado del modelo
@app.get("/v1/state", status_code=status.HTTP_200_OK)
async def estado_modelo():
    global esta_entrenado, modelo_pipeline, historico_dataset
    if not esta_entrenado:
        return {"esta_entrenado": False, "vocabulario_total": 0, "ejemplos_totales": 0}
        
    vectorizer = modelo_pipeline.named_steps['vectorizador']
    classifier = modelo_pipeline.named_steps['clasificador']
    vocab = list(vectorizer.vocabulary_.keys())
    
    return {
        "esta_entrenado": esta_entrenado,
        "vocabulario_total": len(vocab),
        "ejemplos_totales": len(historico_dataset),
        "clases": list(classifier.classes_),
        "vocabulario_muestra": sorted(vocab[:60]),
        "dataset_reciente": historico_dataset[-8:]
    }


# 6. Endpoint para Podar y Compactar Dataset manualmente
@app.post("/v1/prune-dataset", status_code=status.HTTP_200_OK)
async def podar_dataset_manual():
    """Compacta el dataset eliminando redundancias mientras preserva el 100% del vocabulario."""
    global historico_dataset, modelo_pipeline
    antes = len(historico_dataset)
    historico_dataset = podar_dataset_conservando_vocabulario(historico_dataset, max_por_clase=12)
    entrenar_con_dataset(historico_dataset)
    despues = len(historico_dataset)
    vocab_len = len(modelo_pipeline.named_steps['vectorizador'].vocabulary_)
    return {
        "mensaje": f"Dataset compactado con éxito: de {antes} a {despues} ejemplos.",
        "ejemplos_antes": antes,
        "ejemplos_despues": despues,
        "vocabulario_conservado": vocab_len
    }


# 6b. Ecualizador de clases 1:1:1:1 (con copia de seguridad automática)
@app.post("/v1/balance-dataset", status_code=status.HTTP_200_OK)
async def equilibrar_clases(objetivo: int = OBJETIVO_POR_CLASE):
    global historico_dataset
    objetivo = max(10, min(60, objetivo))
    if os.path.exists(ARCHIVO_DATASET):
        shutil.copyfile(ARCHIVO_DATASET, ARCHIVO_BACKUP)

    eval_antes = calcular_evaluacion_modelo(historico_dataset)
    vocab_antes = vocabulario_de(historico_dataset)

    nuevo, informe, fuera = equilibrar_dataset(historico_dataset, objetivo)
    entrenar_con_dataset(nuevo)

    vocab_despues = vocabulario_de(historico_dataset)
    eval_despues = calcular_evaluacion_modelo(historico_dataset)

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
        "backup": ARCHIVO_BACKUP,
        "evaluacion": eval_despues
    }


# ==========================================
# 7. MOTOR DE EVALUACIÓN Y DIAGNÓSTICO DEL LM
# ==========================================
def calcular_evaluacion_modelo(dataset):
    if len(dataset) < 6:
        return {
            "disponible": False,
            "mensaje": "Se requieren al menos 6 ejemplos en el dataset para computar métricas formales."
        }
        
    textos = [item["texto"] for item in dataset]
    etiquetas = [item["etiqueta"] for item in dataset]
    
    conteo_clases = Counter(etiquetas)
    # Consideramos clases con al menos 2 ejemplos para clasificación
    clases_validas = {c for c, count in conteo_clases.items() if count >= 2}
    
    if len(clases_validas) < 2:
        return {
            "disponible": False,
            "mensaje": "Se requieren al menos 2 clases con 2 o más ejemplos para evaluar el clasificador."
        }
        
    indices_validos = [i for i, c in enumerate(etiquetas) if c in clases_validas]
    textos_filtrados = [textos[i] for i in indices_validos]
    etiquetas_filtradas = [etiquetas[i] for i in indices_validos]
    
    # Misma configuración que el modelo en producción (evita medir un modelo distinto)
    eval_pipe = clone(modelo_pipeline)
    
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

    # 4. Auditoría de Calidad ETL y Detección de Anomalías (Inspirado en Infografía ETL)
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

    vectorizer = modelo_pipeline.named_steps['vectorizador'] if 'vectorizador' in modelo_pipeline.named_steps else None
    total_vocab = len(vectorizer.vocabulary_) if vectorizer and hasattr(vectorizer, 'vocabulary_') else 0
    diagnosticos.append(f"Vocabulario N-Gramas: {total_vocab} n-gramas activos con riqueza léxica de {round(riqueza_lexica * 100, 1)}%.")
    diagnosticos.append("Acuerdo Gramatical POS: Generador con ranuras independientes de Verbos y Sustantivos.")

    # 6. Recomendaciones de Acción (Paso 8 del Ciclo ML)
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
        "registro_poda": list(reversed(REGISTRO_PODA[-5:]))
    }


@app.get("/v1/evaluation", status_code=status.HTTP_200_OK)
async def evaluar_modelo():
    """Diagnóstico formal del aprendizaje del LM: Cross-validation, F1-Score, balance de clases y salud léxica."""
    global historico_dataset, modelo_pipeline
    if os.path.exists(ARCHIVO_DATASET):
        try:
            with open(ARCHIVO_DATASET, "r", encoding="utf-8") as f:
                disco_dataset = json.load(f)
            if disco_dataset and len(disco_dataset) >= 2:
                historico_dataset = disco_dataset
                # Re-entrenar modelo principal en caliente para que responda con los nuevos datos
                entrenar_con_dataset(historico_dataset)
        except Exception:
            pass
    return calcular_evaluacion_modelo(historico_dataset)


# 8. Sugerencias Dinámicas basadas en Vocabulario y Habilidades Aprendidas
@app.get("/v1/suggestions", status_code=status.HTTP_200_OK)
async def obtener_sugerencias():
    global historico_dataset, HABILIDADES_PACKS
    sugerencias = []
    
    if historico_dataset:
        clases_incluidas = set()
        for item in reversed(historico_dataset):
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
                pares, _, _ = sintetizar_frases_de_habilidad(h_id, cantidad=1, permitir_descubrimiento=False)
                if pares and pares[0][0] not in sugerencias:
                    sugerencias.append(pares[0][0])
            except Exception:
                pass
                
    return [s.capitalize() for s in sugerencias[:6]]


# 7. Endpoints clásicos de compatibilidad
@app.post("/v1/train", response_model=EntrenamientoResponse, status_code=status.HTTP_200_OK)
async def entrenar_modelo(payload: DatosEntrenamientoRequest):
    dataset = [{"texto": t, "etiqueta": e, "respuesta": f"Respuesta aprendida para {e}."} for t, e in zip(payload.textos, payload.etiquetas)]
    entrenar_con_dataset(dataset)
    vocab_len = len(modelo_pipeline.named_steps['vectorizador'].vocabulary_)
    return EntrenamientoResponse(
        mensaje="Modelo entrenado exitosamente con Scikit-Learn.",
        ejemplos_procesados=len(payload.textos),
        vocabulario_aprendido=vocab_len
    )

@app.post("/v1/predict", response_model=PrediccionResponse, status_code=status.HTTP_200_OK)
async def predecir(payload: PrediccionRequest):
    prediccion = modelo_pipeline.predict([payload.texto])[0]
    probabilidades = modelo_pipeline.predict_proba([payload.texto])[0]
    return PrediccionResponse(
        texto_original=payload.texto,
        prediccion=str(prediccion),
        confianza=round(float(max(probabilidades)), 4)
    )

# 8. Servir Frontend React
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def raiz():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"mensaje": "API activa. Abre /docs o crea static/index.html"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("traininglm:app", host="127.0.0.1", port=8001, reload=True)