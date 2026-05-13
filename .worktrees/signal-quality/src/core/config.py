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
    "AMX", "CMP", "BWA", "RAM", "WZZ", "RPA", "ENY", "ASH", "CFE",
)

# Checked after CIVILIAN_PREFIXES — no military prefix overlaps with civilian prefixes.
# Order matters: civilian check must remain first.
MILITARY_PREFIXES = (
    "REACH", "SPAR", "VENUS", "KNIFE",
    "ASCOT", "TARTAN", "MAGMA", "EVAC",
)


_US_MIL_LOW  = 0xAE0000
_US_MIL_HIGH = 0xAEFFFF
_UK_MIL_LOW  = 0x43C000
_UK_MIL_HIGH = 0x43CFFF


def _is_military_hex(icao24: str) -> bool:
    try:
        val = int(icao24, 16)
    except (ValueError, TypeError):
        return False
    return (_US_MIL_LOW <= val <= _US_MIL_HIGH or
            _UK_MIL_LOW <= val <= _UK_MIL_HIGH)


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

    if _is_military_hex(a.get("icao24", "")):
        return "MILITARY"
    return "UNKNOWN"
