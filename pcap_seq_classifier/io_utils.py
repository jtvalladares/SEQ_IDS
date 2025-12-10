"""I/O utilities para encontrar y cargar PCAPs.

Funciones expuestas:
- find_pcaps(root_dir) -> List[Path]
- load_pcap(path) -> List[dict]
- measure_time(func) -> decorator

Nota: `load_pcap` debe devolver una lista de diccionarios con, al menos, las claves:
`timestamp, src, dst, sport, dport, proto, payload`.
"""
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple, TypeVar, ParamSpec
from scapy.all import rdpcap, IP, TCP, UDP, Raw

import time


def find_pcaps(root_dir: str) -> List[Path]:
    """Recorre root_dir recursivamente y retorna una lista de Paths a archivos pcap/pcapng, ordenada alfabéticamente."""
    
    # Convierte la cadena de ruta a un objeto Path para facilitar la manipulación
    root_path = Path(root_dir)
    
    # Lista de extensiones a buscar (comunes para archivos de captura de red)
    extensions = ('*.pcap', '*.pcapng', '*.cap')
    
    pcap_files = []
    
    # Usamos glob para la búsqueda recursiva
    # rglob(patrón) busca el patrón en el directorio actual y en todos los subdirectorios
    for ext in extensions:
        # Añade los Paths encontrados para cada extensión
        pcap_files.extend(root_path.rglob(ext))
        
    # Ordena la lista final de Paths alfabéticamente
    pcap_files.sort()
    
    return pcap_files


def load_pcap(path: Path) -> List[Dict[str, Any]]:
    """Carga un pcap y retorna lista de paquetes como diccionarios.

    Cada paquete debe contener: timestamp, src, dst, sport, dport, proto, payload (bytes).
    Solo procesa paquetes con capas IP, TCP o UDP.
    """
    
    # Lista para almacenar los paquetes en formato diccionario
    packet_list: List[Dict[str, Any]] = []

    try:
        # Cargar los paquetes desde el archivo
        packets = rdpcap(str(path))
    except FileNotFoundError:
        print(f"❌ Error: Archivo no encontrado en la ruta: {path}")
        return packet_list
    except Exception as e:
        print(f"❌ Error al leer el archivo {path}: {e}")
        return packet_list

    # Recorrer cada paquete capturado
    for packet in packets:
        # 1. Asegurarse de que el paquete tiene capa IP (necesario para src/dst IP)
        if not packet.haslayer(IP):
            continue
        
        # Inicializar valores comunes
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        proto = packet[IP].proto  # 6 para TCP, 17 para UDP, etc.
        
        # Inicializar valores de puerto y payload (dependerán de la capa de transporte)
        src_port = 0
        dst_port = 0
        payload_bytes = b""
        
        # 2. Extraer información de la capa de transporte (TCP o UDP)
        if packet.haslayer(TCP):
            transport_layer = packet[TCP]
            src_port = transport_layer.sport
            dst_port = transport_layer.dport
            proto_name = "TCP"
        elif packet.haslayer(UDP):
            transport_layer = packet[UDP]
            src_port = transport_layer.sport
            dst_port = transport_layer.dport
            proto_name = "UDP"
        else:
            # Si no es TCP ni UDP (ej: ICMP, ARP), establecer puertos a 0 y proto genérico
            proto_name = str(proto)
            # Solo se procesa la capa IP para el resto, si existe
            if packet.haslayer(Raw):
                payload_bytes = packet[Raw].load

        # 3. Extraer el payload (Datos RAW)
        if (packet.haslayer(TCP) or packet.haslayer(UDP)) and packet.haslayer(Raw):
            # Si existe capa de transporte y capa Raw, extraemos la carga útil
            payload_bytes = packet[Raw].load

        # 4. Crear el diccionario de paquete
        packet_dict = {
            "timestamp": packet.time,  # Timestamp de Scapy (formato epoch)
            "src": src_ip,
            "dst": dst_ip,
            "sport": src_port,
            "dport": dst_port,
            "proto": proto_name,
            "payload": payload_bytes
        }
        
        packet_list.append(packet_dict)

    return packet_list

# Definición de tipos genéricos para soportar cualquier firma de función
P = ParamSpec('P')
R = TypeVar('R')

def measure_time(func: Callable[P, R]) -> Callable[P, Tuple[R, float]]:
    """Decorator para medir tiempo de ejecución de funciones.

    Retorna una tupla que contiene: (resultado_de_la_función, tiempo_en_segundos).
    """
    
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Tuple[R, float]:
        # Registrar el tiempo de inicio
        start_time = time.perf_counter()
        
        # Ejecutar la función original
        result = func(*args, **kwargs)
        
        # Registrar el tiempo de finalización
        end_time = time.perf_counter()
        
        # Calcular el tiempo transcurrido
        elapsed_time = end_time - start_time
        
        # Retornar el resultado de la función y el tiempo transcurrido
        return result, elapsed_time
        
    return wrapper