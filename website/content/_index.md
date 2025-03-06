---
date: '2025-03-06T14:46:36+01:00'
draft: false
title: 'Homepage'
---

# SpecHLS: A Speculative HLS Framework

SpecHLS is a source-to-source compiler flow aimed at synthesizing speculative hardware designs using High-Level
Synthesis (HLS). SpecHLS accepts a subset of C++ as input and produces transformed C++ code supporting speculative loop
pipelining. Vitis HLS can then exploit the latter to obtain an accelerator design. The key idea of speculative
loop pipelining (SLP) is to consider speculation as a parallelizing transformation rather than a backend optimization.
Consequently, SpecHLS does not address the issue of scheduling individual operations in the pipeline and delegates that
task to the HLS tool it uses as a backend.
