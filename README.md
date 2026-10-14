> **Status: fixed.** No longer reproduces with ONNX Runtime 1.31 (`ort` 2.0.0-rc.14), fixed upstream by [#32587](https://github.com/microsoft/onnxruntime/pull/32587). Verified on macOS 26.5 (Apple Silicon): all 144 nodes placed on the WebGPU EP, concurrent runs complete without crashing. This repo is archived.

cargo run --release -- version-RFB-640.onnx



https://github.com/microsoft/onnxruntime/issues/27592
https://github.com/microsoft/onnxruntime/issues/31627
