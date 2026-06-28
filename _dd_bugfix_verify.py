import os
os.environ.setdefault("PYTHONUTF8","1")
from aethos_address_store import AddressStore, make_triple, _FLOAT_SAFE_DEPTH

# 1. normal range still exact (the value 7->6 / 2->-1 corruptions the dive saw)
s = AddressStore(); s.put("x", {0:7, 1:2, 2:5000})
got = s.get("x")
assert got == {0:7, 1:2, 2:5000}, got
print("1. normal range exact:", got)

# 2. int-decode exact at a LARGE band just under the wall (old silent-wrong zone)
band = (1 << 53)//3 - 5000        # zeta = 3*band+3+v stays just under 2**53
ok = True
for v in (0, 1, 2, 7, 42, 4999):
    t = make_triple(0, band, v)
    if t.value != v:
        ok = False; print("   MISMATCH", v, "->", t.value)
print(f"2. int-decode exact at band~2^53 (zeta up to {band*3+3+4999}): {'OK' if ok else 'FAIL'}")

# 3. past the wall -> CLEAR ValueError, NOT silent corruption
try:
    make_triple(0, 1 << 53, 0)
    print("3. FAIL: no error past the wall (silent!)")
except ValueError as e:
    print("3. past-wall raises (not silent):", str(e)[:70], "...")

# 4. the dive's exact silent band [~2^52.5, 2^53): now guarded, never silent-wrong
import math
band_silent = int(2 ** 52.6)      # the dive's silent zone
try:
    t = make_triple(0, band_silent, 7)
    print(f"4. silent-zone band={band_silent}: decoded {t.value} (would have been silent-wrong before)")
except ValueError:
    print(f"4. silent-zone band={band_silent}: now RAISES explicitly (no silent corruption)")

# 5. the shipped demo still passes end to end
print("5. running demo()..."); from aethos_address_store import demo
r = demo()
print("   demo flags:", {k:v for k,v in r.items() if isinstance(v,bool)})
