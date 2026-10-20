"""Constantes de dominio, packs léxicos y plantillas gramaticales (POS)."""
import re

VERBOS_CANONICOS = {
    "crear", "depurar", "compilar", "ejecutar", "reiniciar", "configurar", 
    "actualizar", "calcular", "graficar", "entrenar", "agendar", "organizar", 
    "planificar", "optimizar", "solucionar", "proyectar", "diagnosticar", 
    "estructurar", "conectar", "bloquear", "desplegar", "normalizar", "inspeccionar", "sincronizar"
}

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

CLASES_TECNICAS = ["codigo", "agenda", "datos_ia", "soporte", "conversacion"]
OBJETIVO_POR_CLASE = 25
CUOTA_MAXIMA_CLASE = 0.30

PALABRAS_VACIAS = {
    "me", "te", "se", "de", "la", "el", "los", "las", "un", "una", "y", "o", "en",
    "para", "por", "con", "que", "como", "mi", "tu", "es", "al", "del", "lo", "no", "si"
}

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
    },
    {
        "texto": "hola como estas", 
        "etiqueta": "conversacion", 
        "respuesta": "¡Hola! Estoy muy bien, gracias por preguntar. ¿En qué te puedo ayudar hoy?"
    },
    {
        "texto": "buenos dias que tal estas hoy", 
        "etiqueta": "conversacion", 
        "respuesta": "¡Buenos días! Todo excelente por aquí, listo para charlar y ayudarte."
    },
    {
        "texto": "muchas gracias por tu ayuda", 
        "etiqueta": "conversacion", 
        "respuesta": "¡De nada! Es un placer ayudarte. Avísame si necesitas algo más."
    },
    {
        "texto": "adios hasta pronto que te vaya bien", 
        "etiqueta": "conversacion", 
        "respuesta": "¡Hasta pronto! Que tengas un gran día."
    }
]

def es_termino_verbo(palabra: str) -> bool:
    """Detecta si una palabra es un verbo en infinitivo según reglas morfológicas o léxico."""
    p = palabra.lower().strip()
    if p in VERBOS_CANONICOS:
        return True
    return p.endswith(("ar", "er", "ir")) and len(p) >= 5 and p not in {"servidor", "cluster"}

def get_tokens_limpios(texto: str) -> set:
    """Extrae conjunto de palabras y caracteres alfanuméricos limpios en minúsculas."""
    return set(re.findall(r'\b[a-záéíóúñA-ZÁÉÍÓÚÑ0-9_]+\b', texto.lower()))
