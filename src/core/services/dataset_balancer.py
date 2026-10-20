"""Servicio de balanceo, poda inteligente y rescate de vocabulario del dataset."""
import re
import random
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Set

from src.core.domain.constants import (
    HABILIDADES_PACKS, CLASES_TECNICAS, OBJETIVO_POR_CLASE, CUOTA_MAXIMA_CLASE,
    PALABRAS_VACIAS, _VERBOS_BASE, _SUSTANTIVOS_BASE, get_tokens_limpios, es_termino_verbo
)

def _plantilla_a_regex(plantilla: str):
    partes = re.split(r"\{(?:v|s1|s2|s3|p1|p2|p3)\}", plantilla)
    return re.compile("^" + r"\S+".join(re.escape(p) for p in partes) + "$", re.IGNORECASE)

_REGEX_SINTETICAS = [_plantilla_a_regex(p[0]) for p in HABILIDADES_PACKS["cerebro_integral"]["plantillas_pares"]]

class DatasetBalancerService:
    def __init__(self):
        pass

    @staticmethod
    def vocabulario_de(dataset: List[Dict[str, Any]]) -> Set[str]:
        vocab = set()
        for item in dataset:
            vocab.update(get_tokens_limpios(item.get("texto", "")))
        return vocab

    @staticmethod
    def es_relleno_sintetico(item: Dict[str, Any]) -> bool:
        if item.get("origen") == "portadora":
            return False
        if item.get("origen") == "sintetico":
            return True
        txt = item.get("texto", "").strip()
        return any(r.match(txt) for r in _REGEX_SINTETICAS)

    @staticmethod
    def seleccionar_representativos(items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
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

    @classmethod
    def podar_conservando_vocabulario(cls, dataset: List[Dict[str, Any]], max_por_clase: int = 15) -> List[Dict[str, Any]]:
        if len(dataset) <= 30:
            return dataset

        vocabulario_total = set()
        for item in dataset:
            vocabulario_total.update(get_tokens_limpios(item.get("texto", "")))

        palabras_cubiertas = set()
        seleccionados = []
        textos_vistos = set()

        por_clase = {}
        for item in dataset:
            c = item.get("etiqueta", "general")
            por_clase.setdefault(c, []).append(item)

        for c, items in por_clase.items():
            contador_clase = 0
            for item in reversed(items):
                txt = item.get("texto", "").strip().lower()
                if txt in textos_vistos:
                    continue
                toks = get_tokens_limpios(txt)
                palabras_nuevas = toks - palabras_cubiertas

                if palabras_nuevas or contador_clase < max_por_clase:
                    seleccionados.append(item)
                    textos_vistos.add(txt)
                    palabras_cubiertas.update(toks)
                    contador_clase += 1

        return seleccionados

    @classmethod
    def generar_frases_para_clase(cls, clase: str, cantidad: int, textos_existentes: Set[str]) -> List[Dict[str, Any]]:
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

    @classmethod
    def rescatar_vocabulario(cls, original: List[Dict[str, Any]], nuevo: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        perdidas = cls.vocabulario_de(original) - cls.vocabulario_de(nuevo) - PALABRAS_VACIAS
        if not perdidas:
            return nuevo, [], []

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

            frecuencia = Counter()
            for x in resultado:
                frecuencia.update(get_tokens_limpios(x.get("texto", "")))
            huecos = [i for i, x in enumerate(resultado) if x.get("etiqueta") == clase and cls.es_relleno_sintetico(x)]
            huecos.sort(key=lambda i: -sum(1 for t in get_tokens_limpios(resultado[i]["texto"]) if frecuencia[t] == 1))
            for item in nuevas:
                if huecos:
                    resultado[huecos.pop()] = item
                else:
                    resultado.append(item)

        aun_perdidas = sorted(cls.vocabulario_de(original) - cls.vocabulario_de(resultado) - PALABRAS_VACIAS)
        rescatadas = sorted(perdidas - set(aun_perdidas))
        return resultado, rescatadas, aun_perdidas

    @classmethod
    def equilibrar(cls, dataset: List[Dict[str, Any]], objetivo: int = OBJETIVO_POR_CLASE) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int, Dict[str, Any]]:
        por_clase = {}
        for item in dataset:
            por_clase.setdefault(item.get("etiqueta"), []).append(item)
        textos_existentes = {item.get("texto", "").strip().lower() for item in dataset}

        resultado, informe = [], {}
        for clase in CLASES_TECNICAS:
            items = por_clase.get(clase, [])
            antes = len(items)
            if antes > objetivo:
                seleccion, sinteticas = cls.seleccionar_representativos(items, objetivo), []
            else:
                seleccion = list(items)
                sinteticas = cls.generar_frases_para_clase(clase, objetivo - antes, textos_existentes)
            resultado.extend(seleccion + sinteticas)
            informe[clase] = {
                "antes": antes,
                "despues": len(seleccion) + len(sinteticas),
                "podadas": antes - len(seleccion),
                "sinteticas": len(sinteticas)
            }
        fuera_de_dominio = sum(len(v) for k, v in por_clase.items() if k not in CLASES_TECNICAS)

        resultado, rescatadas, perdidas = cls.rescatar_vocabulario(dataset, resultado)
        registro_evento = {
            "origen": "ecualizador",
            "clases": informe,
            "frases_podadas": sum(v["podadas"] for v in informe.values()),
            "palabras_rescatadas": rescatadas,
            "palabras_perdidas": perdidas
        }
        return resultado, informe, fuera_de_dominio, registro_evento

    @classmethod
    def aplicar_cuota_maxima(cls, dataset: List[Dict[str, Any]], cuota: float = CUOTA_MAXIMA_CLASE) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        conteo = Counter(item.get("etiqueta") for item in dataset)
        if len(conteo) < 4:
            return dataset, None
        resultado = list(dataset)
        clases_podadas = {}
        for clase, n in conteo.most_common():
            resto = len(resultado) - n
            limite = max(int(resto * cuota / (1 - cuota)), 10)
            if n > limite:
                items = [x for x in resultado if x.get("etiqueta") == clase]
                conservar = {id(x) for x in cls.seleccionar_representativos(items, limite)}
                resultado = [x for x in resultado if x.get("etiqueta") != clase or id(x) in conservar]
                clases_podadas[clase] = {"antes": n, "despues": limite, "podadas": n - limite, "sinteticas": 0}
        if not clases_podadas:
            return dataset, None
        resultado, rescatadas, perdidas = cls.rescatar_vocabulario(dataset, resultado)
        return resultado, {
            "origen": "cuota_30",
            "clases": clases_podadas,
            "frases_podadas": sum(v["podadas"] for v in clases_podadas.values()),
            "palabras_rescatadas": rescatadas,
            "palabras_perdidas": perdidas
        }

    @staticmethod
    def sanear_dataset_existente(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        frases_genericas = [
            "Módulo de soporte técnico: te sugiero verificar tu conexión de red",
            "Módulo de agenda y productividad: tarea agregada a tu itinerario",
            "Módulo de datos: aplicaremos matrices de dispersión",
            "Módulo de programación activado: he analizado tu consulta",
            "¡Entendido! Ya sé cómo responder"
        ]
        for item in dataset:
            resp = item.get("respuesta", "")
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
        return dataset
