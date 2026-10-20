"""Servicio de inducción sintética de frases y descubrimiento léxico con acuerdo POS."""
import random
from typing import List, Tuple, Dict, Any
from src.core.domain.constants import (
    HABILIDADES_PACKS, _VERBOS_BASE, _SUSTANTIVOS_BASE, es_termino_verbo
)

class SyntheticInductionService:
    def __init__(self, packs: Dict[str, Any] = HABILIDADES_PACKS):
        self.packs = packs

    def sintetizar(
        self, 
        habilidad_id: str = "cerebro_integral", 
        cantidad: int = 3, 
        permitir_descubrimiento: bool = True
    ) -> Tuple[List[Tuple[str, str, str]], str, List[str]]:
        pack = self.packs.get(habilidad_id) or list(self.packs.values())[0]
            
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
