"""
k-way meet algebra — derived from AETHOS spec §4–§5 (prime-by-prime branching).

How 2-way naturally creates 3-way (and 3-way → 4-way, …)
----------------------------------------------------------
**§4 solo (2-way swap):**  bank(a) @ n=p  ≡  bank(p) @ n=a

**§5 pair (2-way branch):**  chain (a,p), transgressor n crosses anchors:
  Case 1: n < a          velocity regime A
  Case 2: a ≤ n < p      velocity change at a
  Case 3: n ≥ p          velocity change at p

**3-way from 2-way (compose):**  two pair banks sharing prefix a:
  bank(a,p) @ n=q  =  bank(a,q) @ n=p          (pair cross)
  bank(a,p,q) @ n=p  confirms triple            (velocity extension)

**3-way lock (sunflower):**  for P=(p₁,p₂,p₃), every 2-way subset missing one
prime witnesses at n = missing prime → same Ψ  (k=3 only).

**3-way slide:**  bank(p₁,p₂) @ p₃  =  bank(p₂,p₃) @ p₁  (oriented merge).

**k≥3 velocity extension (4-way, 5-way, …):**  when the (k−1)-chain transgresses
to the new prime, n meets the prior anchor on the deep chain — same §5 rule:

  bank(P[:-1]) @ P[-1]  ≡  bank(P) @ P[-2]

Two (k−1) triple vectors sharing prefix P[:-2] generally do **not** meet each
other at equal Z; both meet the k-way node at bank(P) @ P[-2].
"""

from __future__ import annotations

from dataclasses import dataclass

from aethos_complex_plane import equalize_witness, swap_meet
from aethos_lattice import BranchKind, LatticeId
from aethos_recursive import LatticeBank32K, normalize_primes, segment_index

DEFAULT_BRANCH = BranchKind.VA1
DEFAULT_WING = 1
DEFAULT_LID = LatticeId.L01


def _sorted_primes(*primes: int) -> tuple[int, ...]:
    return normalize_primes(primes)


@dataclass(frozen=True)
class SwapMeetWitness:
    """§4 solo swap: bank(a)@n=b ≡ bank(b)@n=a."""

    a: int
    b: int
    coord: tuple[float, float, float]
    unified: bool

    def explain(self) -> dict:
        return {
            "way": 2,
            "primes": [self.a, self.b],
            "coord": list(self.coord),
            "unified": self.unified,
            "rule": "bank(a)@n=b ≡ bank(b)@n=a",
        }


@dataclass(frozen=True)
class SlideMeetWitness:
    """Oriented slide: bank(P[:-1])@p_k = bank(P[1:])@p_1."""

    primes: tuple[int, ...]
    left_chain: tuple[int, ...]
    right_chain: tuple[int, ...]
    n_left: int
    n_right: int
    coord_left: tuple[int, int, int]
    coord_right: tuple[int, int, int]
    unified: bool

    def explain(self) -> dict:
        return {
            "way": len(self.primes),
            "primes": list(self.primes),
            "left_chain": list(self.left_chain),
            "right_chain": list(self.right_chain),
            "n_left": self.n_left,
            "n_right": self.n_right,
            "coord": list(self.coord_left) if self.unified else None,
            "unified": self.unified,
            "rule": "bank(P[:-1])@p_k = bank(P[1:])@p_1",
        }


@dataclass(frozen=True)
class VelocityMeetWitness:
    """
    §5 velocity extension — shallow transgresses to new prime, deep meets at prior anchor.

    bank(P[:-1]) @ P[-1]  ≡  bank(P) @ P[-2]
    """

    primes: tuple[int, ...]
    shallow_chain: tuple[int, ...]
    n_shallow: int
    n_deep: int
    shallow_segment: int
    deep_segment: int
    coord: tuple[int, int, int]
    unified: bool

    def explain(self) -> dict:
        return {
            "way": len(self.primes),
            "primes": list(self.primes),
            "shallow_chain": list(self.shallow_chain),
            "n_shallow": self.n_shallow,
            "n_deep": self.n_deep,
            "shallow_segment": self.shallow_segment,
            "deep_segment": self.deep_segment,
            "coord": list(self.coord),
            "unified": self.unified,
            "rule": "bank(P[:-1])@P[-1] ≡ bank(P)@P[-2]",
        }


@dataclass(frozen=True)
class BranchComposeWitness:
    """
    Two (k−1) banks sharing prefix P[:-2] — how 3-way vectors spawn k-way.

    left  = P[:-1]   transgresses @ P[-1]
    right = P[:-2]+(P[-1],)  transgresses @ P[-2]
    k-node = bank(P) @ P[-2]  (velocity anchor — both shallow rails meet here)
    """

    primes: tuple[int, ...]
    left_chain: tuple[int, ...]
    right_chain: tuple[int, ...]
    n_left: int
    n_right: int
    n_k_node: int
    coord_left: tuple[int, int, int]
    coord_right: tuple[int, int, int]
    coord_k_node: tuple[int, int, int]
    pair_vectors_unified: bool
    left_meets_k_node: bool
    right_meets_k_node: bool
    k_node_unified: bool

    def explain(self) -> dict:
        return {
            "way": len(self.primes),
            "primes": list(self.primes),
            "left_chain": list(self.left_chain),
            "right_chain": list(self.right_chain),
            "n_left": self.n_left,
            "n_right": self.n_right,
            "n_k_node": self.n_k_node,
            "coord_k_node": list(self.coord_k_node),
            "pair_vectors_unified": self.pair_vectors_unified,
            "left_meets_k_node": self.left_meets_k_node,
            "right_meets_k_node": self.right_meets_k_node,
            "k_node_unified": self.k_node_unified,
            "rule": "two (k-1) vectors meet k-node at prior anchor, not each other",
        }


@dataclass(frozen=True)
class SubSunflowerWitness:
    """One (k-1)-subset lock: witness n = dropped prime."""

    full_chain: tuple[int, ...]
    dropped_prime: int
    subset: tuple[int, ...]
    witness_n: float
    coord: tuple[float, float, float]

    def explain(self) -> dict:
        return {
            "full_chain": list(self.full_chain),
            "subset": list(self.subset),
            "dropped_prime": self.dropped_prime,
            "witness_n": self.witness_n,
            "coord": list(self.coord),
        }


@dataclass(frozen=True)
class DeepSegment:
    """Coord on full chain at anchor n (velocity boundary crossing)."""

    primes: tuple[int, ...]
    n: int
    segment: int
    coord: tuple[int, int, int]

    def explain(self) -> dict:
        return {
            "n": self.n,
            "segment": self.segment,
            "coord": list(self.coord),
        }


@dataclass(frozen=True)
class ComposeStep:
    """One promotion shallow → deep via velocity extension."""

    shallow: tuple[int, ...]
    deep: tuple[int, ...]
    added_prime: int
    n_shallow: int
    n_deep: int
    coord: tuple[int, int, int] | None
    unified: bool

    def explain(self) -> dict:
        return {
            "shallow": list(self.shallow),
            "deep": list(self.deep),
            "added_prime": self.added_prime,
            "n_shallow": self.n_shallow,
            "n_deep": self.n_deep,
            "coord": list(self.coord) if self.coord else None,
            "unified": self.unified,
            "rule": "bank(shallow)@added ≡ bank(deep)@prior_anchor",
        }


@dataclass(frozen=True)
class KMeetReport:
    """Glass-box k-way meet report."""

    primes: tuple[int, ...]
    k: int
    swap: SwapMeetWitness | None
    slide: SlideMeetWitness | None
    velocity: VelocityMeetWitness | None
    branch_compose: BranchComposeWitness | None
    sub_sunflowers: tuple[SubSunflowerWitness, ...]
    full_sunflower_unified: bool
    deep_segments: tuple[DeepSegment, ...]
    compose_steps: tuple[ComposeStep, ...]

    def explain(self) -> dict:
        return {
            "primes": list(self.primes),
            "k": self.k,
            "swap": self.swap.explain() if self.swap else None,
            "slide": self.slide.explain() if self.slide else None,
            "velocity": self.velocity.explain() if self.velocity else None,
            "branch_compose": self.branch_compose.explain() if self.branch_compose else None,
            "full_sunflower_unified": self.full_sunflower_unified,
            "sub_sunflowers": [s.explain() for s in self.sub_sunflowers],
            "deep_segments": [d.explain() for d in self.deep_segments],
            "compose_steps": [c.explain() for c in self.compose_steps],
        }


def swap_meet_primes(
    a: int,
    b: int,
    *,
    branch: BranchKind = DEFAULT_BRANCH,
    wing: int = DEFAULT_WING,
) -> SwapMeetWitness:
    left, right = swap_meet(a, b, branch, wing)
    coord = left.coord
    return SwapMeetWitness(a=a, b=b, coord=coord, unified=left.coord == right.coord)


def slide_meet(
    *primes: int,
    lid: LatticeId = DEFAULT_LID,
) -> SlideMeetWitness | None:
    if not primes:
        return None
    ps = _sorted_primes(*primes)
    if len(ps) < 3:
        return None
    left, right = ps[:-1], ps[1:]
    pk, p1 = ps[-1], ps[0]
    bank_l = LatticeBank32K(left)
    bank_r = LatticeBank32K(right)
    c_left = bank_l[lid].at(pk)
    c_right = bank_r[lid].at(p1)
    return SlideMeetWitness(
        primes=ps,
        left_chain=left,
        right_chain=right,
        n_left=pk,
        n_right=p1,
        coord_left=c_left,
        coord_right=c_right,
        unified=c_left == c_right,
    )


def velocity_meet(
    *primes: int,
    lid: LatticeId = DEFAULT_LID,
) -> VelocityMeetWitness | None:
    """
    §5 velocity extension — bank(P[:-1])@P[-1] ≡ bank(P)@P[-2].

    Requires |P| ≥ 3. This is how (k−1)-way transgression creates the k-way node.
    """
    if not primes:
        return None
    ps = _sorted_primes(*primes)
    if len(ps) < 3:
        return None
    shallow = ps[:-1]
    n_shallow = ps[-1]
    n_deep = ps[-2]
    bank_s = LatticeBank32K(shallow)
    bank_d = LatticeBank32K(ps)
    c_s = bank_s[lid].at(n_shallow)
    c_d = bank_d[lid].at(n_deep)
    return VelocityMeetWitness(
        primes=ps,
        shallow_chain=shallow,
        n_shallow=n_shallow,
        n_deep=n_deep,
        shallow_segment=segment_index(shallow, n_shallow),
        deep_segment=segment_index(ps, n_deep),
        coord=c_s if c_s == c_d else c_d,
        unified=c_s == c_d,
    )


def branch_compose(
    *primes: int,
    lid: LatticeId = DEFAULT_LID,
) -> BranchComposeWitness | None:
    """
    Two (k−1) banks sharing prefix — how 3-way vectors meet to open k-way.

    Requires |P| ≥ 4.
    """
    if not primes:
        return None
    ps = _sorted_primes(*primes)
    if len(ps) < 4:
        return None
    left = ps[:-1]
    right = ps[:-2] + (ps[-1],)
    n_left = ps[-1]
    n_right = ps[-2]
    n_k = ps[-2]
    bank_l = LatticeBank32K(left)
    bank_r = LatticeBank32K(right)
    bank_k = LatticeBank32K(ps)
    c_l = bank_l[lid].at(n_left)
    c_r = bank_r[lid].at(n_right)
    c_k = bank_k[lid].at(n_k)
    return BranchComposeWitness(
        primes=ps,
        left_chain=left,
        right_chain=right,
        n_left=n_left,
        n_right=n_right,
        n_k_node=n_k,
        coord_left=c_l,
        coord_right=c_r,
        coord_k_node=c_k,
        pair_vectors_unified=c_l == c_r,
        left_meets_k_node=c_l == c_k,
        right_meets_k_node=c_r == c_k,
        k_node_unified=c_l == c_k,
    )


def pair_branch_compose(
  a: int,
  b: int,
  c: int,
  *,
  lid: LatticeId = DEFAULT_LID,
) -> tuple[tuple[int, int, int], int] | None:
    """
    §5 pair cross → triple confirm: bank(a,b)@c = bank(a,c)@b = bank(a,b,c)@b.

    Returns (coord, n_deep) when unified.
    """
    bank_ab = LatticeBank32K((a, b))
    bank_ac = LatticeBank32K((a, c))
    bank_abc = LatticeBank32K((a, b, c))
    c_ab = bank_ab[lid].at(c)
    c_ac = bank_ac[lid].at(b)
    if c_ab != c_ac:
        return None
    c_deep = bank_abc[lid].at(b)
    if c_ab != c_deep:
        return None
    return c_ab, b


def sub_sunflowers(
    *primes: int,
    branch: BranchKind = DEFAULT_BRANCH,
    wing: int = DEFAULT_WING,
) -> tuple[SubSunflowerWitness, ...]:
    if len(primes) < 2:
        return ()
    ps = _sorted_primes(*primes)
    out: list[SubSunflowerWitness] = []
    for drop in ps:
        sub = tuple(p for p in ps if p != drop)
        n_w, psi = equalize_witness(ps, sub, branch, wing)
        out.append(
            SubSunflowerWitness(
                full_chain=ps,
                dropped_prime=drop,
                subset=sub,
                witness_n=n_w,
                coord=psi.coord,
            )
        )
    return tuple(out)


def full_sunflower_unified(subs: tuple[SubSunflowerWitness, ...]) -> bool:
    if not subs:
        return False
    return len({s.coord for s in subs}) == 1


def deep_segments(
    *primes: int,
    lid: LatticeId = DEFAULT_LID,
) -> tuple[DeepSegment, ...]:
    if not primes:
        return ()
    ps = _sorted_primes(*primes)
    bank = LatticeBank32K(ps)
    lat = bank[lid]
    return tuple(
        DeepSegment(primes=ps, n=n, segment=segment_index(ps, n), coord=lat.at(n))
        for n in ps
    )


def compose_k(
    *primes: int,
    lid: LatticeId = DEFAULT_LID,
    branch: BranchKind = DEFAULT_BRANCH,
    wing: int = DEFAULT_WING,
) -> KMeetReport:
    ps = _sorted_primes(*primes)
    k = len(ps)

    swap = swap_meet_primes(ps[0], ps[1], branch=branch, wing=wing) if k == 2 else None
    slide = slide_meet(*ps, lid=lid) if k >= 3 else None
    velocity = velocity_meet(*ps, lid=lid) if k >= 3 else None
    branch_w = branch_compose(*ps, lid=lid) if k >= 4 else None
    subs = sub_sunflowers(*ps, branch=branch, wing=wing)
    unified = full_sunflower_unified(subs)
    segments = deep_segments(*ps, lid=lid)

    steps: list[ComposeStep] = []
    if k >= 2:
        for i in range(2, k + 1):
            prefix = ps[:i]
            if len(prefix) == 2:
                a, b = prefix
                bank_s = LatticeBank32K((a,))
                bank_d = LatticeBank32K(prefix)
                c_s = bank_s[lid].at(b)
                c_d = bank_d[lid].at(a)
                steps.append(
                    ComposeStep(
                        shallow=(a,),
                        deep=prefix,
                        added_prime=b,
                        n_shallow=b,
                        n_deep=a,
                        coord=c_s if c_s == c_d else None,
                        unified=c_s == c_d,
                    )
                )
            else:
                vel = velocity_meet(*prefix, lid=lid)
                steps.append(
                    ComposeStep(
                        shallow=prefix[:-1],
                        deep=prefix,
                        added_prime=prefix[-1],
                        n_shallow=prefix[-1],
                        n_deep=prefix[-2],
                        coord=vel.coord if vel and vel.unified else None,
                        unified=bool(vel and vel.unified),
                    )
                )

    return KMeetReport(
        primes=ps,
        k=k,
        swap=swap,
        slide=slide,
        velocity=velocity,
        branch_compose=branch_w,
        sub_sunflowers=subs,
        full_sunflower_unified=unified,
        deep_segments=segments,
        compose_steps=tuple(steps),
    )
