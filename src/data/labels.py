"""Attack taxonomy for the corrected CICIDS2017 dataset (Engelen et al., WTMC 2021).

Raw `Label` values include a `- Attempted` suffix for connections that attempted
but did not complete an attack; these are folded into their base attack class
(they are still attacker-originated traffic, not benign).
"""

CICIDS2017_TAXONOMY = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS",
    "DoS Hulk - Attempted": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS GoldenEye - Attempted": "DoS",
    "DoS slowloris": "DoS",
    "DoS slowloris - Attempted": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS Slowhttptest - Attempted": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Bot": "Bot",
    "Bot - Attempted": "Bot",
    "FTP-Patator": "Brute Force",
    "FTP-Patator - Attempted": "Brute Force",
    "SSH-Patator": "Brute Force",
    "SSH-Patator - Attempted": "Brute Force",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - Brute Force - Attempted": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - XSS - Attempted": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Web Attack - Sql Injection - Attempted": "Web Attack",
    "Infiltration": "Infiltration",
    "Infiltration - Attempted": "Infiltration",
    "Heartbleed": "Heartbleed",
}

CICIDS2017_CLASSES = [
    "Benign",
    "DoS",
    "DDoS",
    "PortScan",
    "Bot",
    "Brute Force",
    "Web Attack",
    "Infiltration",
    "Heartbleed",
]

# Non-feature / leakage-prone columns dropped before modeling: identifiers (Flow ID,
# IPs, ports/timestamp are flow-metadata, not traffic-shape signal) and the raw label.
CICIDS2017_DROP_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Timestamp",
    "Label",
]

# Domain-knowledge feature groups used for SHAP feature/attack alignment checks
# (Phase 4 gate: do top SHAP features match known attack signatures?).
ATTACK_SIGNATURE_FEATURES = {
    "DoS": ["Flow Duration", "Flow Packets/s", "Flow Bytes/s", "Flow IAT Mean", "Fwd Packets/s"],
    "DDoS": ["Flow Packets/s", "Flow Bytes/s", "Fwd Packets/s", "Flow Duration", "Packet Length Mean"],
    "PortScan": ["Flow Duration", "Total Fwd Packet", "Fwd Packets/s", "Down/Up Ratio", "FIN Flag Count"],
    "Bot": ["Flow Duration", "Flow IAT Mean", "Idle Mean", "Fwd IAT Mean", "Bwd Packet Length Mean"],
    "Brute Force": ["Flow Duration", "Total Fwd Packet", "Fwd IAT Mean", "Flow IAT Mean", "Fwd Packet Length Mean"],
    "Web Attack": ["Flow Duration", "Total Length of Fwd Packet", "Fwd Packet Length Mean", "Flow Bytes/s"],
    "Infiltration": ["Flow Duration", "Bwd Packet Length Mean", "Flow IAT Std", "Idle Mean"],
    "Heartbleed": ["Bwd Packet Length Max", "Flow Duration", "Total Length of Bwd Packet"],
}
