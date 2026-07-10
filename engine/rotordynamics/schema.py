"""
Pydantic contracts for a rotordynamic analysis run.

These models are the interface both the LLM tool (`run_rotordynamic_analysis`)
and the web UI's manual form are built from: they validate parameters before
anything touches the physics, and they describe the structured result callers
get back. Field defaults mirror the reference shaft-disk-bearing system in
`analysis.RotordynamicAnalysis.__init__`, so a partial spec still runs.

Coverage notes (matches what the engine can actually do):
* Bearing *type* and ball-bearing stiffness/damping are per bearing.
* The journal oil-film geometry (diameter, length, clearance, viscosity) is a
  single set shared by both bearings — that is how the solver is written.
* Component axial positions are optional; when given, the FE mesh nodes are
  remapped piecewise-linearly onto them (see tools._apply_params).

Backward compatibility: a legacy top-level "bearing" object (one spec for
both bearings) is still accepted and expanded automatically.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ShaftParams(BaseModel):
    diameter_m: float = Field(13e-3, gt=0, description="Shaft diameter [m]")
    length_m: float = Field(747e-3, gt=0, description="Shaft length [m]")
    youngs_modulus_pa: float = Field(207e9, gt=0, description="Young's modulus E [Pa]")
    density_kg_m3: float = Field(7850, gt=0, description="Material density [kg/m^3]")


class DiskParams(BaseModel):
    diameter_m: float = Field(90e-3, gt=0, description="Disk diameter [m]")
    length_m: float = Field(47e-3, gt=0, description="Disk length [m]")
    mass_kg: float = Field(2.3, gt=0, description="Disk mass [kg]")
    unbalance_kg_m: float = Field(3.7e-5, ge=0, description="Unbalance m*e [kg.m]")


class BallBearingParams(BaseModel):
    """Rolling-element bearing: fixed stiffness/damping (speed-independent)."""
    kxx_n_m: float = Field(1e8, gt=0, description="Stiffness X [N/m]")
    kyy_n_m: float = Field(1e8, gt=0, description="Stiffness Y [N/m]")
    kxy_n_m: float = Field(0, description="Cross-coupling stiffness XY [N/m]")
    kyx_n_m: float = Field(0, description="Cross-coupling stiffness YX [N/m]")
    cxx_ns_m: float = Field(1e3, ge=0, description="Damping X [N.s/m]")
    cyy_ns_m: float = Field(1e3, ge=0, description="Damping Y [N.s/m]")
    cxy_ns_m: float = Field(0, description="Cross-coupling damping XY [N.s/m]")
    cyx_ns_m: float = Field(0, description="Cross-coupling damping YX [N.s/m]")


class BearingSpec(BaseModel):
    """One bearing: its kind, plus ball properties when kind == 'ball'.
    Journal bearings take their film geometry from RunParams.journal_film."""
    kind: str = Field("journal", description="'journal' (hydrodynamic) or 'ball'")
    ball: BallBearingParams = Field(default_factory=BallBearingParams,
                                    description="Used only when kind == 'ball'")


class JournalFilmParams(BaseModel):
    """Oil-film geometry shared by both journal bearings (engine limitation)."""
    diameter_m: float = Field(30e-3, gt=0, description="Bearing bore diameter [m]")
    length_m: float = Field(20e-3, gt=0, description="Bearing axial length [m]")
    radial_clearance_m: float = Field(90e-6, gt=0, description="Radial clearance [m]")
    viscosity_pa_s: float = Field(0.051, gt=0, description="Oil absolute viscosity [Pa.s]")


class PositionParams(BaseModel):
    """Optional axial stations [m from the shaft's left end]. When omitted, the
    reference mesh layout is scaled with the shaft length. Must be strictly
    increasing and inside the shaft."""
    bearing1_m: float = Field(26e-3, gt=0)
    disk_m: float = Field(171e-3, gt=0)
    bearing2_m: float = Field(721e-3, gt=0)


class SpeedRange(BaseModel):
    start_rad_s: float = Field(10, gt=0)
    stop_rad_s: float = Field(2000, gt=0)
    step_rad_s: float = Field(10, gt=0)


class RunParams(BaseModel):
    """Everything needed to configure one analysis run."""
    shaft: ShaftParams = Field(default_factory=ShaftParams)
    disk: DiskParams = Field(default_factory=DiskParams)
    bearing1: BearingSpec = Field(default_factory=BearingSpec)
    bearing2: BearingSpec = Field(default_factory=BearingSpec)
    journal_film: JournalFilmParams = Field(default_factory=JournalFilmParams)
    positions: PositionParams | None = Field(
        None, description="Optional axial stations; default = reference layout scaled with shaft length")
    speed: SpeedRange = Field(default_factory=SpeedRange)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_bearing(cls, data):
        """Accept the old single 'bearing' spec: kind goes to both bearings,
        film geometry goes to journal_film."""
        if isinstance(data, dict) and "bearing" in data:
            legacy = data.pop("bearing") or {}
            kind = legacy.get("kind", "journal")
            data.setdefault("bearing1", {"kind": kind})
            data.setdefault("bearing2", {"kind": kind})
            film = {k: v for k, v in legacy.items()
                    if k in ("diameter_m", "length_m", "radial_clearance_m", "viscosity_pa_s")}
            if film:
                data.setdefault("journal_film", film)
        return data

    @model_validator(mode="after")
    def _check_consistency(self):
        for spec, name in ((self.bearing1, "bearing1"), (self.bearing2, "bearing2")):
            if spec.kind.lower().strip() not in ("journal", "ball"):
                raise ValueError(f"{name}.kind must be 'journal' or 'ball', got {spec.kind!r}")
        if self.positions is not None:
            p, L = self.positions, self.shaft.length_m
            if not (0 < p.bearing1_m < p.disk_m < p.bearing2_m < L):
                raise ValueError(
                    "positions must satisfy 0 < bearing1 < disk < bearing2 < shaft length")
        if self.speed.stop_rad_s <= self.speed.start_rad_s:
            raise ValueError("speed.stop_rad_s must be greater than speed.start_rad_s")
        return self


class RunResult(BaseModel):
    """Structured output the agent (and the web UI) reasons over and cites."""
    critical_speeds_rad_s: list[float] = Field(default_factory=list)
    bearing_reactions_n: list[float] = Field(default_factory=list)
    speed_points: int = 0
    report_slug: str = Field(
        "",
        description="Slug of the ingested wiki page for this run, e.g. 'runs/2026-07-10-run-001'",
    )
    summary: str = ""
