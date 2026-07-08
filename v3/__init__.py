"""v3 — physics-anchored RUL track (damage-state-first, time-second).

The NN never sees a time-like target: it predicts the physical damage state
(fault class + ordinal severity + crack length), and the Paris-law engine
converts damage → RUL downstream. See docs/ and the approved plan for the
benchmark rationale (DARPA ESP chain, Wiener first-passage RUL).
"""
