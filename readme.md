<div align="center">

# Codeformer Tensorrt

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.3-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-8.6-green)](https://developer.nvidia.com/tensorrt)

</div>

<p align="center">
  <img src="demo.png" height="128" />
</p>

- This repo provides a minimal TensorRT implementation of [Codeformer](https://github.com/sczhou/CodeFormer) in Python, enabling fast face restoration on images
- This implementation does not include preprocessing (face detection/alignment/cropping) and postprocessing (pasting the restored face on the original image)
- The model only performs inference on preprocessed images (e.g [input.png](./input.png)), which need to be 512 x 512, with face fully visible

## ⏱️ Performance

| Device | Model Input (WxH) | Inference Time(ms) |
| :----: | :---------------: | :----------------: |
|  A10G  |     512 x 512     |         23         |

> [!NOTE]
> Inference was conducted using `FP16` precision, with a warm-up period of 10 frames. The reported time corresponds to the last inference.

## 🛠️ Building Tensorrt Engine

1. Download the [codeformer onnx model](https://huggingface.co/yuvraj108c/codeformer-onnx/tree/main)
2. Run the following command

   ```bash
   trtexec --onnx=codeformer.onnx --saveEngine=codeformer.engine --fp16
   ```

## ⚡ Inference

```bash
git clone https://github.com/yuvraj108c/Codeformer-Tensorrt.git
pip install -r requirements.txt
python inference.py --input ./input.png --engine ./codeformer.engine --output ./output.png
```

## 🤖 Environment tested

- Ubuntu 22.04 LTS, Cuda 12.3, Tensorrt 8.6.1, Python 3.10, A10G GPU
- Windows (Not tested)

## 👏 Credits

- [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer)
