"""Demo: once = intersection only; split meaning only when contexts truly differ."""

from aethos_frequency import FrequencyProfile
from aethos_natural import NaturalReader

r = NaturalReader(rebuild_every=2)
r.read(
    "xylophone appeared once in music text",
    "phone phone phone technical hardware",
    "phone phone chip software",
    "apple phone chip technical",
    "apple fruit pie orchard baking",
)

fp = FrequencyProfile.from_reader(r)
print("=" * 60)
print("INTERSECTION POLICY — no wasted primes")
print("=" * 60)

for w in ("xylophone", "phone", "apple"):
    print(f"\n  {fp.explain(w)}")
    if w in r.registry.intersections:
        print(f"    -> intersection record, L4-L9 still active")
    if (r.registry.promoted.get(("L3_WORD", w))):
        print(f"    -> dedicated L3 prime allocated (contexts differ)")

print("\n--- apple disambiguation still works via L7-L9 ---")
print(r.explain_natural("apple", ["phone", "chip"]))
print(r.explain_natural("apple", ["fruit", "pie"]))
