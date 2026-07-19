---
type: Practice Note
title: API 610 - Lateral Analysis Requirements (5.2.4 / Appendix I)
description: The binding two-step API 610 pipeline for pump rotor critical speeds - Step 1, is a lateral analysis even required (SS5.2.4.1.1 classically-stiff screen); Step 2, if so, the Appendix I pass/fail criteria - summarized in original words, with an honest note on what this project's engine currently computes.
tags: [rotordynamics, critical-speed, separation-margin, api-610, lateral-analysis, standards]
timestamp: 2026-07-19T00:00:00Z
---

## Overview

API Standard 610 ("Centrifugal Pumps for Petroleum, Heavy Duty Chemical, and Gas
Industry Services") is this project's normative reference for critical-speed /
lateral-analysis compliance - the class of machine this copilot's FEA engine models
(a shaft-disk-bearing rotor with hydrodynamic or rolling-element supports) is exactly
what SS5.2.4 "Dynamics" and Appendix I "Lateral Analysis" govern. Section and figure
numbers below refer to the 8th Edition (1995) of API 610; always confirm against the
current edition before using a figure as a binding requirement on real equipment.
Standard text is summarized in original words, not reproduced.

## The pipeline, in two steps

1. **Screen** (SS5.2.4.1.1): is a lateral analysis even required for this rotor? Most
   rotors are exempt - only a minority actually need Step 2.
2. **Analyze** (Appendix I), only if Step 1 says it's required: calculate natural
   frequencies, check separation margin vs. damping, and (if that check fails) compute
   the damped response to unbalance against an allowable-displacement limit.

A "does this comply with API 610" question is really "does it clear Step 1, and if
not, does it clear Step 2" - not a single percentage-of-operating-speed rule.

## Step 1 - Do you even need a lateral analysis? (SS5.2.4.1.1)

Unless the purchaser specifies otherwise, this is decided by a short decision tree
(API 610 Figure 5-1), not run unconditionally:

1. If the pump is **identical** to an existing, already-qualified pump (same size,
   hydraulic design, stage count, speed, clearances, shaft-seal type, bearing type,
   coupling weight/overhang, and pumped liquid) - or **similar** by agreement between
   purchaser and manufacturer on those same factors - no analysis is recommended.
2. Otherwise, if the rotor is **classically stiff**, no analysis is recommended either.
   Classically stiff means the first *dry* critical speed sits above the pump's
   **maximum allowable continuous speed (MCS)** by a margin of:
   * **20%**, for rotors designed for wet running only, or
   * **30%**, for rotors also designed to run dry.

   As an explicit inequality, so there's no ambiguity about direction:
   `dry_critical_speed >= 1.20 x MCS` (wet-running-only), or
   `dry_critical_speed >= 1.30 x MCS` (can also run dry).
   The threshold is a floor **raised above** MCS - it is never `MCS` minus a
   margin, and it is never a two-sided band around MCS. A dry critical speed
   sitting anywhere at or below MCS itself obviously fails this test too.
3. Only if neither exemption applies is a lateral analysis **required** -> go to Step 2.

**What "dry" and "wet" mean here, precisely (SS1.4.7, SS1.4.8) - and why it matters
for this copilot:** the distinction is sharper than "with vs. without process liquid":

* *Dry* critical speed: the rotor's natural frequency assuming it is supported only
  at its bearings, and that **those bearings have infinite stiffness** - a rigid-support
  idealization, not a real bearing's actual behavior.
* *Wet* critical speed: the natural frequency including the pumped liquid's additional
  stiffness/damping at internal running clearances, **and** allowing for the bearings'
  real (finite) flexibility and damping.

So "dry" is not simply "wet minus the liquid" - it also swaps the bearing's real,
finite stiffness for an idealized infinite one. Since infinite bearing stiffness is
the stiffest possible support, the true dry critical speed is the *highest* value a
given shaft/bearing arrangement can have; any calculation using a bearing's real
(finite) stiffness - wet or otherwise - sits at or below it. See the engine note
below for what this means in practice.

## Step 2 - Carrying out and assessing the analysis (Appendix I)

**Natural frequencies (SS I.1.2).** All rotor natural frequencies are calculated across
the frequency band 0 to **2.2x** MCS, sweeping a running-speed range of **25-125%** of
rated speed. The model must account for internal running-clearance stiffness/damping
(at both new and 2x-new clearance, with the pumped liquid, plus new clearance with
water at test temperature), labyrinth seal stiffness/damping if applicable, bearing
stiffness/damping at average clearance and oil temperature, bearing support
mass/stiffness, and coupling-hub/spacer inertia. Any natural frequency with a damping
ratio below **0.4** must be flagged in the report.

**Separation margin vs. damping (SS I.1.3).** For both new and 2x-new clearances, every
bending natural frequency is checked against the running-speed range on a damping-factor
vs. frequency-ratio chart (Figure I-2, f_ni/f_run on the x-axis, damping factor xi on the
y-axis). The chart's "acceptable" region is widest away from synchronous coincidence and
narrows sharply as the frequency ratio approaches 1.0, where it demands the most damping
(figure not reproduced here - it's a proprietary illustration; the shape is described,
not the digitized curve). A mode landing in the "unacceptable" band there triggers the
next step instead of an automatic fail.

Note 2 to this section gives the damping-metric relationships used throughout Appendix I:
delta = 2*pi*xi / sqrt(1 - xi^2), and for xi up to 0.4, xi ~= delta/(2*pi) ~= 1/(2*AF).
**Critically damped** corresponds to xi >= 0.2, delta >= 1.2, AF <= 2.5 - the same
AF <= 2.5 "no margin required" threshold used in [[critical-speeds-and-separation-margin]].

**Damped unbalance response (SS I.1.4).** If a mode fails the Figure I-2 check, its
response is instead computed directly: at new and 2x-new clearances, with the pumped
liquid, exciting the mode with **4x** the rotor's maximum allowable unbalance (per
SS5.2.4.2.1), one mode per run.

**Allowable displacement (SS I.1.5).** However the mode was assessed, the resulting
peak-to-peak shaft displacement at the point of maximum response must not exceed **35%**
of the diametral running clearance at that point.

## What this copilot's engine actually computes today

Be precise about this with users - it's the difference between an honest partial
answer and an invented one:

* The engine's Campbell-diagram sweep always uses **real, finite** bearing
  stiffness/damping - journal bearings: hydrodynamic oil-film coefficients from the
  Reynolds-equation solution at each speed; ball bearings: the fixed user-supplied
  stiffness/damping. It never uses the infinite/rigid-bearing idealization SS1.4.7
  defines as "dry", and it never adds the pumped-liquid running-clearance effects
  SS1.4.8 adds for "wet". A run's reported critical speed is therefore neither the
  literal dry nor the literal wet number - it is a third case: real bearings, no
  process-liquid seal effects.
* That third case is still useful for Step 1, **in one direction only**. Because it
  uses finite (not infinite) bearing stiffness, a run's critical speed is
  systematically **lower than or equal to** the true dry value for the same rotor
  (finite support stiffness can only match or reduce the natural frequency relative
  to the infinite-stiffness idealization, never raise it above it). That makes a
  run's critical speed a conservative **lower bound** on the true dry critical speed:
  - If the run's critical speed already clears `1.20x` / `1.30x` MCS, the true dry
    critical speed (rigid bearings) would clear it too - safe to report the rotor as
    passing the classically-stiff screen.
  - If the run's critical speed does **not** clear that threshold, that is
    inconclusive, not a fail - the true (higher) dry value might still clear it. Say
    the screen is inconclusive from this run and that the true dry (rigid-bearing)
    value would be needed to settle it - do not report "lateral analysis required" on
    this basis alone.
  - Ball-bearing runs sit closer to the true dry value than journal-bearing runs,
    since rolling-element bearings are typically far stiffer (closer to the
    infinite-stiffness idealization) than a hydrodynamic oil film - but neither is
    exact, and the "inconclusive on fail" rule above applies to both.
* `RunParams`/`RunResult` still have no **maximum allowable continuous speed (MCS)**
  field - the speed sweep is a start/stop/step range, not an operating-point spec - so
  Step 1 always needs the user to supply MCS separately; never invent it or substitute
  the user's stated "operating"/"rated" speed for it.
* Step 2's natural frequencies and AF, by contrast, **are** exactly what the engine
  already reports (the Campbell sweep and half-power AF), so a Step 2 separation-margin
  question can be answered from a real run once Step 1 has determined it's needed.

## Sources

* API Standard 610, 8th Edition (August 1995), Section 1.4 "Definition of Terms"
  (SS1.4.6-SS1.4.8, critical speed / dry / wet), Section 5.2.4 "Dynamics", and Appendix I
  "Lateral Analysis" - criteria summarized in original words from the source held at
  `knowledge/raw/API 610.pdf`; standard text is not reproduced verbatim and figures 5-1,
  I-1 and I-2 are described, not copied.
* Mark A. Corbo and Stanley B. Malanoski, "Pump Rotordynamics Made Simple,"
  *Proceedings of the 15th International Pump Users Symposium* - the Lomakin effect
  and the physical mechanism behind the wet/dry critical-speed discrepancy in pumps,
  summarized in original words.

## Related pages

* [Critical Speeds and Separation Margin](critical-speeds-and-separation-margin.md)
* [Rotordynamics Glossary](rotordynamics-glossary.md)
