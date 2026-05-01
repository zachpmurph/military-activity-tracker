CIVILIAN_PREFIXES = (
    # Major US
    "AAL","UAL","DAL","SWA","FFT","JBU","ASA","SKW",

    # Cargo
    "FDX","UPS",

    # Europe
    "DLH","BAW","AFR","KLM","RYR","EZY","VLG",

    # Middle East / Asia
    "QTR","UAE","ETD","SIA","ANA","JAL",

    # Canada
    "ACA","WJA","JZA",

    # Latin America
    "AMX","CMP","BWA","RAM",
    "WZZ","EZY","RYR","VLG","IBE","SAS",
    "RPA","SKW","ENY","ASH",
    "CFE","BAW","DLH","KLM",

    "ASA","EJU","WZZ","RPA","CFE",
    "SWR","ITY","SVA","BAW","DLH",
    "AFR","KLM","EZY","RYR"
)

MILITARY_PREFIXES = (
    "REACH", "SPAR", "VENUS", "KNIFE",
    "ASCOT", "TARTAN", "MAGMA", "EVAC",
)


def classify_aircraft(a):
    callsign = a["callsign"]

    if callsign.startswith(CIVILIAN_PREFIXES):
        return "CIVILIAN"

    if callsign.startswith(MILITARY_PREFIXES):
        return "MILITARY"

    if callsign.startswith("RCH"):
        return "US_CARGO"

    if callsign.startswith(("QID", "K35R", "DRAG")):
        return "TANKER"

    if callsign.startswith("RRR"):
        return "UK_CARGO"

    return "UNKNOWN"
