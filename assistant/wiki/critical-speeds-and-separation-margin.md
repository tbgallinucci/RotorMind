---
type: Practice Note
title: Critical Speeds and Separation Margin
description: Critical speeds, Campbell diagrams, and amplification factor summarized in original words - the general theory behind the engine's output; the binding API 610 pass/fail pipeline lives on api-610-lateral-analysis.
tags: [rotordynamics, critical-speed, campbell, api-610, separation-margin]
timestamp: 2026-07-19T00:00:00Z
---

## Overview

A critical speed is a rotating speed at which the frequency of a synchronous excitation - almost always residual unbalance at exactly 1x running speed - coincides with a natural frequency of the rotor-bearing system. Passing through or dwelling near a critical produces amplified vibration limited only by the system damping. These notes summarize the general theory *in my own words*. For the actual pass/fail pipeline - whether a lateral analysis is even required, and the criteria it's checked against - see [[api-610-lateral-analysis]] (API Standard 610, SS5.2.4 and Appendix I), the binding standard for the shaft-disk-bearing rotor class this project models.

## Campbell diagram

The Campbell diagram plots the system natural frequencies against running speed. For journal-bearing machines the natural frequencies themselves change with speed (bearing coefficients are speed-dependent) and split into forward and backward whirl branches under gyroscopic effects. Overlaying the synchronous line (frequency = running speed, the "1X line") identifies the critical speeds as intersections. This is exactly how the engine finds them: it solves the eigenvalue problem at every speed step, tracks the natural-frequency lines, and interpolates each crossing of the 1X line (see the "Frequency Response & Campbell Diagram" plot of any run report).

## Separation margin - the concept

Operating continuously *at* a critical is unacceptable; a machine's operating speed range must be separated from its critical speeds by some margin. How much margin is needed depends on how strongly the machine responds at that critical, quantified through the amplification factor (AF) measured or predicted at resonance: sharply-tuned, lightly-damped criticals (high AF) demand a wide margin; heavily damped criticals (low AF, AF <= 2.5) are considered critically damped enough that no separate margin check is required. That AF <= 2.5 cutoff is not just a rule of thumb - [[api-610-lateral-analysis]] documents it as API 610's actual "critically damped" definition (equivalent to a damping ratio xi >= 0.2), used inside its real pass/fail pipeline.

For the actual binding pipeline - whether a lateral analysis is even required in the first place, the natural-frequency sweep range, the separation-margin/damping acceptance check, and the allowable-displacement and shop-verification tolerances - see [[api-610-lateral-analysis]]. Always check the current edition of the standard for the exact criteria.

## Amplification factor

AF is estimated from the response peak with the half-power (3 dB) method: AF = N_c / (N_2 - N_1), where N_c is the speed at the response peak and N_1, N_2 the speeds at 0.707 of the peak amplitude. High AF means little damping at that mode - expect a sharp, dangerous resonance. Journal bearings usually damp the first bending mode strongly; stiff rolling-element supports do not.

## Practical workflow with this copilot

1. Ask for the machine's critical speeds - the copilot runs the FEA and reports them with a run citation.
2. Compare against the [[api-610-lateral-analysis]] pipeline: is a lateral analysis even required (SS5.2.4.1.1), and if so, do the reported critical speeds and AF clear the separation-margin/damping check (Appendix I)?
3. If the margin is thin, the classic levers are bearing stiffness/damping (bearing type, clearance, preload), shaft diameter (stiffness scales with d^4), span, and disk placement.

## Sources

* API Standard 610, "Centrifugal Pumps for Petroleum, Heavy Duty Chemical, and Gas Industry Services", SS5.2.4 and Appendix I - binding lateral-analysis criteria (summarized, not reproduced; see [[api-610-lateral-analysis]]).
* D. Childs, *Turbomachinery Rotordynamics*, Wiley, 1993 - modal analysis and amplification factors.
* Original summary notes for this project.

## Related pages

* [API 610 - Lateral Analysis Requirements (5.2.4 / Appendix I)](api-610-lateral-analysis.md)
* [Journal Bearing Theory](journal-bearing-theory.md)
* [Rotordynamics Glossary](rotordynamics-glossary.md)
