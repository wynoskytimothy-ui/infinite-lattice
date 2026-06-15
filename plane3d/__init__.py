"""
plane3d — standalone 3D complex plane (Ψ ∈ ℂ × ℝ).

No RAG, no registry, no BEIR, no hub signatures.
Geometry only: canon, wings, meets, spring operators, κ keys.

Usage:
    import plane3d as P

    psi = P.imaginary_start(7)           # z = 7 + 7i
    psi = P.wing_transform(P.BranchKind.VA1, (3, 5, 7), 5, wing=1)
    left, right = P.swap_meet(3, 5)
    key = P.kappa(psi.z, psi.zeta)
    report = P.verify_all_gates()
"""

from plane3d.gates import GateReport, verify_all_gates
from plane3d.velocity import (
    ImaginaryAxes,
    VelocityPlane,
    decompose_around_critical,
    four_branch_rotation_planes,
    rotate_around_critical,
    rotate_around_critical_line,
    rotate_psi_around_critical,
    segment_velocity,
    u_hat_complex,
    velocity_plane,
    verify_branch_rotation_gates,
)
from plane3d.roots import (
    SheetIndex,
    apply_sheets,
    depth_square,
    layer0_depth_from_spring,
    layer0_spring_from_depth,
    spring_square,
    sqrt_depth,
    sqrt_spring,
    verify_sqrt_gates,
    wing_sheet_mask,
)
from plane3d.key import (
    AttractorKey,
    DEFAULT_QUANTIZE,
    attractor_neighbors,
    kappa,
    kappa_psi,
    keys_from_psi,
)
from plane3d.lattice import (
    BranchKind,
    Coord,
    LatticeId,
    VECTORS,
    apply_vector,
    lattice_id_parts,
)
from plane3d.psi import (
    ComplexPlane3D,
    LatticeAddress,
    all_branch_phases,
    canon_complex,
    depth_at,
    equalize_witness,
    imaginary_start,
    missing_member,
    segment_at,
    swap_meet,
    triple_equalization,
    trigger_history,
    wing_transform,
    wing_transform_lid,
)
from plane3d.sequences import SequenceKind, canon_on_chain, make_chain, normalize_chain
from plane3d.spring import (
    SpringPoint,
    conj_act,
    i_act,
    is_on_critical_line,
    verify_critical_line_rotation,
    verify_i_act_axioms,
    verify_va_vb_swap_diagonal,
)

__all__ = [
    "AttractorKey",
    "BranchKind",
    "ComplexPlane3D",
    "Coord",
    "DEFAULT_QUANTIZE",
    "GateReport",
    "ImaginaryAxes",
    "LatticeAddress",
    "VelocityPlane",
    "LatticeId",
    "SequenceKind",
    "SheetIndex",
    "SpringPoint",
    "VECTORS",
    "all_branch_phases",
    "apply_sheets",
    "apply_vector",
    "attractor_neighbors",
    "canon_complex",
    "canon_on_chain",
    "conj_act",
    "depth_at",
    "decompose_around_critical",
    "depth_square",
    "four_branch_rotation_planes",
    "equalize_witness",
    "i_act",
    "imaginary_start",
    "is_on_critical_line",
    "kappa",
    "kappa_psi",
    "keys_from_psi",
    "layer0_depth_from_spring",
    "layer0_spring_from_depth",
    "lattice_id_parts",
    "make_chain",
    "missing_member",
    "normalize_chain",
    "rotate_around_critical",
    "rotate_around_critical_line",
    "rotate_psi_around_critical",
    "segment_at",
    "segment_velocity",
    "spring_square",
    "sqrt_depth",
    "sqrt_spring",
    "swap_meet",
    "u_hat_complex",
    "triple_equalization",
    "trigger_history",
    "verify_all_gates",
    "verify_branch_rotation_gates",
    "verify_sqrt_gates",
    "velocity_plane",
    "wing_sheet_mask",
    "verify_critical_line_rotation",
    "verify_i_act_axioms",
    "verify_va_vb_swap_diagonal",
    "wing_transform",
    "wing_transform_lid",
]

__version__ = "1.0.0"
