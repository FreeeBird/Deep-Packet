import numpy as np
from scapy.compat import raw
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.dns import DNS
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse


# 去以太帧头
def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload
    return packet


# 改ip
def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'
    return packet


# 填充udp
def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()
        # build a padding layer
        pad = Padding()
        pad.load = '\x00' * 12
        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after
        return packet
    return packet


def packet_to_sparse_array(packet, max_length=1500):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    arr = sparse.csr_matrix(arr)
    return arr


def transform_packet(packet):
    if should_omit_packet(packet):
        return None

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)

    arr = packet_to_sparse_array(packet)

    return arr


def should_omit_packet(packet):
    # SYN, ACK or FIN flags set to 1 and no payload
    # 010011 -> 19 -> 0x13
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    # DNS segment
    if DNS in packet:
        return True

    return False
