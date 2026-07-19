---
type: Reference
title: Rotordynamics Glossary
description: Working definitions - critical speed, whirl, gyroscopic effect, unbalance, eccentricity ratio, attitude angle, amplification factor, Campbell diagram, separation margin, mode shape, FRF.
tags: [rotordynamics, glossary, reference]
timestamp: 2026-07-10T00:00:00Z
---

## Overview

Short working definitions used across this wiki and in the engine's reports. Original wording.

## Terms

| Term | Definition |
|---|---|
| Critical speed | Running speed at which synchronous (1x) excitation coincides with a system natural frequency; response peaks there. |
| Natural frequency | Frequency at which the rotor-bearing system vibrates freely; speed-dependent when bearing coefficients or gyroscopics matter. |
| Campbell diagram | Plot of natural frequencies vs running speed; critical speeds are the intersections with the 1X synchronous line. |
| Whirl | Orbiting of the deflected shaft centreline; forward whirl follows rotation, backward whirl opposes it. |
| Oil whirl / oil whip | Sub-synchronous instability of plain journal bearings near 0.4-0.48x speed; whip is its locked-on, saturated form at a natural frequency. |
| Unbalance | Offset between mass centre and geometric centre; quantified as m.e [kg.m]; drives 1x synchronous excitation with force m.e.omega^2. |
| Eccentricity ratio (epsilon) | Journal centre offset divided by radial clearance; 0 = concentric, 1 = wall contact. |
| Attitude angle (phi) | Angle between static load direction and the shaft-bushing line of centres at equilibrium. |
| Stiffness coefficients (K_ij) | Linearized film force per unit displacement; cross-coupled terms (i != j) transfer energy between directions. |
| Damping coefficients (C_ij) | Linearized film force per unit velocity; the squeeze-film effect. |
| Gyroscopic effect | Speed-proportional coupling from spinning inertia (disk Ip); stiffens forward whirl, softens backward whirl, splits the Campbell branches. |
| Mode shape | Deflection pattern of the rotor at a natural frequency (e.g. first bending, conical). |
| FRF | Frequency response function: vibration amplitude at a node vs excitation frequency; peaks mark resonances. |
| Amplification factor (AF) | Sharpness of a resonance peak, AF = N_c/(N_2 - N_1) by the half-power method; high AF = lightly damped. |
| Separation margin | Required distance between any critical speed and the operating speed range; grows with AF. Binding numeric criteria: API 610 SS5.2.4/Appendix I. |
| Radial clearance (delta) | Gap between journal and bushing radii; the most sensitive journal-bearing design variable. |
| Sommerfeld-type solutions | Family of closed-form Reynolds-equation solutions (long/short bearing, cavitation conditions). |
| Run report | This project's citable record of one FEA execution, ingested at wiki/runs/<date>-run-NNN. |

## Sources

* Definitions written for this project; concepts standard in Childs (1993), Vance (1988), API 610.

## Related pages

* [Journal Bearing Theory](journal-bearing-theory.md)
* [Reynolds Equation](reynolds-equation.md)
* [Critical Speeds and Separation Margin](critical-speeds-and-separation-margin.md)
* [API 610 - Lateral Analysis Requirements (5.2.4 / Appendix I)](api-610-lateral-analysis.md)
