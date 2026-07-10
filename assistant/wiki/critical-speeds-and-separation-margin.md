---
type: Practice Note
title: Critical Speeds and Separation Margin
description: Critical speeds, Campbell diagrams, damped vs undamped analysis, and the API 684 separation-margin concept summarized in original words; how the engine identifies criticals.
tags: [rotordynamics, critical-speed, campbell, api-684, separation-margin]
timestamp: 2026-07-10T00:00:00Z
---

## Overview

A critical speed is a rotating speed at which the frequency of a synchronous excitation - almost always residual unbalance at exactly 1x running speed - coincides with a natural frequency of the rotor-bearing system. Passing through or dwelling near a critical produces amplified vibration limited only by the system damping. These notes summarize the concepts *in my own words*; consult API Standard 684 (the rotordynamics tutorial) and API 610/617 for the binding requirements - the standard text itself is copyrighted and is not reproduced here.

## Campbell diagram

The Campbell diagram plots the system natural frequencies against running speed. For journal-bearing machines the natural frequencies themselves change with speed (bearing coefficients are speed-dependent) and split into forward and backward whirl branches under gyroscopic effects. Overlaying the synchronous line (frequency = running speed, the "1X line") identifies the critical speeds as intersections. This is exactly how the engine finds them: it solves the eigenvalue problem at every speed step, tracks the natural-frequency lines, and interpolates each crossing of the 1X line (see the "Frequency Response & Campbell Diagram" plot of any run report).

## Separation margin - the concept

Operating continuously *at* a critical is unacceptable; standards therefore require the operating speed range to be separated from critical speeds by a margin. In API-style practice:

* The margin is defined between every critical speed and the limits of the operating speed range (minimum to maximum continuous speed, including trip range considerations).
* How much margin is required depends on how strongly the machine responds at that critical, quantified through the amplification factor (AF) measured or predicted at resonance: sharply-tuned, lightly-damped criticals (high AF) demand a wide margin; heavily damped criticals (low AF, roughly AF below 2.5) are considered critically damped enough that no margin is required.
* Typical figures quoted from API-style rules of thumb: for criticals *below* the minimum operating speed, a margin of the order of 15-16 percent; for criticals *above* the maximum continuous speed, of the order of 20-26 percent, increasing with AF. Always check the current edition for the exact formulae - the numbers here are indicative, not normative.

## Amplification factor

AF is estimated from the response peak with the half-power (3 dB) method: AF = N_c / (N_2 - N_1), where N_c is the speed at the response peak and N_1, N_2 the speeds at 0.707 of the peak amplitude. High AF means little damping at that mode - expect a sharp, dangerous resonance. Journal bearings usually damp the first bending mode strongly; stiff rolling-element supports do not.

## Practical workflow with this copilot

1. Ask for the machine's critical speeds - the copilot runs the FEA and reports them with a run citation.
2. Compare the intended operating speed against each critical: margin = (nearest critical - operating speed) / operating speed.
3. If the margin is thin, the classic levers are bearing stiffness/damping (bearing type, clearance, preload), shaft diameter (stiffness scales with d^4), span, and disk placement.

## Sources

* API Standard 684, "API Standard Paragraphs Rotordynamic Tutorial" - concept definitions (summarized, not reproduced).
* D. Childs, *Turbomachinery Rotordynamics*, Wiley, 1993 - modal analysis and amplification factors.
* Original summary notes for this project.

## Related pages

* [Journal Bearing Theory](journal-bearing-theory.md)
* [Rotordynamics Glossary](rotordynamics-glossary.md)
