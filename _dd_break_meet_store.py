"""
_dd_break_meet_store.py -- does the SHIPPING AddressStore inherit the float wall?

AddressStore.make_triple builds (a,p,q)=(band, band+1, band+2+value) and reads
zeta off the float-based ComplexPlane3D. value = round(zeta) - (3*band+3).
If band is large enough that 3*band+value+3 > 2^53, the float zeta rounds and
the recovered value is WRONG. Find the band where the store starts lying.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aethos_address_store import make_triple, AddressStore

OUT=[]
def log(s=""): OUT.append(str(s))

log("="*70)
log("AddressStore triple value-recovery vs band magnitude")
log("="*70)
log("  zeta = 3*band + 3 + value  (float).  Wall at zeta >= 2^53.")
for bexp in [20, 30, 40, 48, 50, 51, 52, 53, 54, 60]:
    band = 1 << bexp
    value = 12345
    try:
        node = make_triple(axis_id=0, band=band, value=value)
        rec = node.value
        ok = (rec == value)
        zeta = node.zeta
        log(f"  band=2^{bexp:<2} zeta~={zeta:.0f} recovered_value={rec} "
            f"{'OK' if ok else f'*** WRONG (got {rec}, want {value}) ***'}")
    except Exception as ex:
        log(f"  band=2^{bexp:<2} EXCEPTION {type(ex).__name__}: {ex}")

log("")
log("="*70)
log("Default store config: base=2^20, band_size=2^20 -- how many axes before wall?")
log("="*70)
base = 1<<20; band_size = 1<<20
# largest axis_id whose band top stays < 2^53
# band(axis) = base + axis*band_size ; need 3*band + value + 3 < 2^53
maxband = (2**53) // 3
max_axis = (maxband - base)//band_size
log(f"  base=2^20 band_size=2^20: float-safe up to axis_id ~ {max_axis:,} (~2^{max_axis.bit_length()})")
log(f"  -> the DEFAULT store is float-safe for the first ~{max_axis/1e6:.1f}M axes;")
log(f"     it only lies once you allocate billions of axes (band tops cross 2^53).")

# Demonstrate an ACTUAL store lie by forcing a huge band
log("")
log("  Forced lie: store one key on a band above the wall")
s = AddressStore(band_size=1<<20, base=1<<53)   # base already at the wall
try:
    s.put("k", {0: 7})
    got = s.get("k")
    log(f"    base=2^53, put value 7 -> get -> {got}  "
        f"{'OK' if got.get(0)==7 else '*** CORRUPTED (float wall) ***'}")
except Exception as ex:
    log(f"    base=2^53, put value 7 -> {type(ex).__name__}: {ex}")
    log(f"    *** WALL: above 2^53 the float anchors (band, band+1) ROUND TO EQUAL,")
    log(f"        so normalize_chain raises 'anchors must be distinct' -- the store")
    log(f"        cannot even encode, let alone decode. Hard fail (not silent).")
# also show the SILENT-corruption regime just below the duplicate-collapse:
# pick a band where 3*band+value+3 crosses 2^53 but anchors still distinct
log("")
log("  Silent-corruption regime (band just under the wall, value large):")
band = (1<<53)//3 + 5     # 3*band ~ 2^53
for value in [0, 100, 10_000, 1_000_000]:
    try:
        node = make_triple(axis_id=0, band=band, value=value)
        rec = node.value
        log(f"    band~2^53/3 value={value:>9} -> recovered={rec}  "
            f"{'OK' if rec==value else f'*** SILENT WRONG (got {rec}) ***'}")
    except Exception as ex:
        log(f"    band~2^53/3 value={value:>9} -> {type(ex).__name__}: {ex}")

txt="\n".join(OUT)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"_dd_break_meet_store_out.txt"),
          "w",encoding="utf-8") as f: f.write(txt)
print(txt)
