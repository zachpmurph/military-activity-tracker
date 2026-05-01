CIVILIAN_PREFIXES = (
    # Major US carriers
    "AAL", "UAL", "DAL", "SWA", "FFT", "JBU", "ASA", "SKW",
    # Cargo
    "FDX", "UPS",
    # Europe
    "DLH", "BAW", "AFR", "KLM", "RYR", "EZY", "VLG", "SWR", "ITY", "IBE", "SAS", "EJU",
    # Middle East / Asia
    "QTR", "UAE", "ETD", "SIA", "ANA", "JAL", "SVA",
    # Canada
    "ACA", "WJA", "JZA",
    # Latin America / Other
    "AMX", "CMP", "BWA", "RAM", "WZZ", "IBE", "RPA", "ENY", "ASH", "CFE",
)

# Checked after CIVILIAN_PREFIXES — no military prefix overlaps with civilian prefixes.
# Order matters: civilian check must remain first.
MILITARY_PREFIXES = (
    "REACH", "SPAR", "VENUS", "KNIFE",
    "ASCOT", "TARTAN", "MAGMA", "EVAC",
)


def classify_aircraft(a):
    callsign = a["callsign"].upper()

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
