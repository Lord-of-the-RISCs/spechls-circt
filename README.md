# SpecHLS: A Speculative HLS Framework

SpecHLS is a source-to-source compiler flow aimed at synthesizing speculative hardware designs using High-Level
Synthesis (HLS). SpecHLS accepts a subset of C++ as input and produces transformed C++ code supporting speculative loop
pipelining. Vitis HLS can then exploit the latter to obtain an accelerator design.

## License

All the code in this repository is released under the `Apache 2.0 License with LLVM Exceptions`. See the
[LICENSE](LICENSE) file for more details.
