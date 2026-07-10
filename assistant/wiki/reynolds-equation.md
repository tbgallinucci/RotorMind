---
type: Theory Note
title: Reynolds Equation
description: The Reynolds equation of hydrodynamic lubrication - assumptions, physical meaning, short-bearing (Ocvirk) solution used by the engine, and the resulting load capacity; original notes.
tags: [rotordynamics, lubrication, reynolds-equation, short-bearing]
timestamp: 2026-07-10T00:00:00Z
---

## Overview

The Reynolds equation is the governing equation of hydrodynamic lubrication: a reduced form of the Navier-Stokes equations for a thin viscous film, it relates the pressure field p(x, z) in the oil film to the film thickness h, the viscosity mu, and the relative motion of the surfaces. Every journal-bearing force, stiffness, and damping value in this project ultimately comes from a solution of this equation.

## Assumptions

The classical (incompressible, isoviscous) form assumes: the film is thin compared with the bearing radius, so pressure is constant across the film thickness; flow is laminar; inertia terms are negligible next to viscous ones; the lubricant is Newtonian with constant viscosity; and the surfaces are rigid. Under these, for a journal bearing with circumferential coordinate x = R theta and axial coordinate z:

    d/dx ( h^3/(12 mu) dp/dx ) + d/dz ( h^3/(12 mu) dp/dz ) = (U/2) dh/dx + dh/dt

The right-hand side contains the two physical pumping mechanisms: the *wedge* term (U/2) dh/dx (rotation dragging oil into a converging gap) and the *squeeze* term dh/dt (radial approach of the surfaces). The wedge carries the static load; the squeeze film provides the damping.

## Film thickness

For a journal with eccentricity ratio epsilon inside radial clearance delta:

    h(theta) = delta (1 + epsilon cos(theta))

The gap converges over half the circumference and diverges over the other half; in the divergent region the film cavitates, and practical solutions set pressure to ambient there (Gumbel / half-Sommerfeld condition).

## Short-bearing (Ocvirk) solution

For a bearing whose length L is small compared with its diameter D (roughly L/D below about 0.5), axial pressure flow dominates circumferential flow, and dropping the circumferential pressure term yields a closed-form pressure profile - parabolic in z. Integrating it gives compact expressions for the radial and tangential film forces; the ones this engine solves at each speed are:

    F_t proportional to  mu omega (D/2) L^3 / delta^2 * pi epsilon / (4 (1 - epsilon^2)^(3/2))
    F_r proportional to  mu omega (D/2) L^3 / delta^2 * epsilon^2 / (1 - epsilon^2)^2

Setting F_t and F_r in equilibrium with the static bearing reaction gives the operating (epsilon, phi); differentiating the forces about that point gives the eight linearized coefficients used in the FE model (see [Journal Bearing Theory](journal-bearing-theory.md)). The strong (1 - epsilon^2) powers explain the film's hardening behaviour: load capacity grows without bound as epsilon approaches 1.

## What to remember

* Load capacity scales with mu omega L^3 / delta^2 - clearance is the most sensitive design variable.
* The same film that carries load also produces the cross-coupled stiffness responsible for oil whirl; you cannot have one without the other in a plain cylindrical bore.
* Short-bearing theory is an approximation: good for trends and educational rigs, increasingly wrong for L/D near 1, grooved bores, or turbulent films.

## Sources

* O. Reynolds, "On the Theory of Lubrication...", Phil. Trans. Royal Society, 1886 - the original derivation (public domain).
* D. Childs, *Turbomachinery Rotordynamics*, Wiley, 1993 - short-bearing force expressions and coefficient derivation.
* Original summary notes; no copyrighted text reproduced.

## Related pages

* [Journal Bearing Theory](journal-bearing-theory.md)
* [Rotordynamics Glossary](rotordynamics-glossary.md)
