---
type: Theory Note
title: Journal Bearing Theory
description: Hydrodynamic journal bearings - how the oil film carries load, eccentricity and attitude angle, linearized stiffness and damping coefficients, oil whirl and whip; original notes citing Childs and Vance.
tags: [rotordynamics, bearings, hydrodynamic, stiffness, damping]
timestamp: 2026-07-10T00:00:00Z
---

## Overview

A hydrodynamic (plain journal) bearing supports a rotating shaft on a thin film of oil rather than on rolling elements. As the journal spins, it drags viscous oil into the converging gap between shaft and bushing; the resulting pressure field carries the load with no metal-to-metal contact at running speed. Because the film's reaction depends on where the journal sits *and* how fast it moves, a journal bearing behaves like a speed-dependent spring-damper pair in each radial direction - which is exactly why bearing modelling dominates rotordynamic predictions (refs: Childs, "Turbomachinery Rotordynamics"; Vance, "Rotordynamics of Turbomachinery").

## Static equilibrium: eccentricity and attitude angle

Under a static load W the journal centre does not sit directly below the bearing centre. It settles at:

| Quantity | Symbol | Meaning |
|---|---|---|
| Eccentricity ratio | epsilon = e / delta | journal centre offset e as a fraction of the radial clearance delta; 0 = concentric, 1 = touching |
| Attitude angle | phi | angle between the load direction and the line of centres |

The equilibrium (epsilon, phi) at each speed is found by balancing the film's radial and tangential force components against the applied load - in this project's engine that balance is solved by Newton-Raphson at every speed point. Light load or high speed pushes epsilon toward 0; heavy load or low speed pushes it toward 1. The locus of the journal centre across the speed range traces the familiar arc inside the clearance circle (see any run report's "Bearing Locus" plot).

## Linearized coefficients

For small vibrations about equilibrium, the film force is expanded to first order, giving four stiffness and four damping coefficients:

    F_y = -K_yy y - K_yz z - C_yy dy/dt - C_yz dz/dt
    F_z = -K_zy y - K_zz z - C_zy dy/dt - C_zz dz/dt

Two properties matter for rotordynamics:

* All eight coefficients vary with speed, because equilibrium (epsilon, phi) varies with speed.
* The cross-coupled stiffnesses K_yz and K_zy are not equal. Their asymmetry feeds energy from rotation into whirl and is the classic destabilizing mechanism in plain journal bearings (refs: Childs; Vance).

The engine evaluates these coefficients from closed-form influence functions of epsilon (short-bearing theory, see [Reynolds Equation](reynolds-equation.md)) scaled by W/delta for stiffness and W/(omega delta) for damping.

## Oil whirl and oil whip

At light load / high speed, the mean oil-film circulates at just under half the shaft speed. The film can then drive a forward sub-synchronous orbit near 0.4-0.48x running speed ("oil whirl"). If running speed rises until that whirl frequency locks onto a shaft natural frequency, the instability saturates into "oil whip", which does not track speed anymore and can be destructive. Practical fixes raise the film's effective load or break its symmetry: higher unit load, shorter bearings, lobed or pressure-dam bores, tilting pads (which nearly eliminate cross-coupling) (refs: Vance; Childs).

## Ball bearings by contrast

Rolling-element bearings carry load through elastic contact, so their stiffness is roughly constant with speed and much higher than an oil film's, with very little damping and negligible cross-coupling. The engine models them as fixed K and C values. Systems on ball bearings show critical speeds that barely move with speed, while journal-bearing systems show speed-dependent criticals - visible directly in the Campbell diagram of any run report.

## Sources

* D. Childs, *Turbomachinery Rotordynamics: Phenomena, Modeling, and Analysis*, Wiley, 1993 - bearing coefficient derivations and stability.
* J. Vance, *Rotordynamics of Turbomachinery*, Wiley, 1988 - oil whirl/whip phenomenology.
* These notes are original summaries written for this project; no standard or book text is reproduced.

## Related pages

* [Reynolds Equation](reynolds-equation.md)
* [Critical Speeds and Separation Margin](critical-speeds-and-separation-margin.md)
* [Rotordynamics Glossary](rotordynamics-glossary.md)
