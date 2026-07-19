# CoastSeg-lite

> **Parent-project users:** run setup and imagery work from `multimethod-sdb`. The
> parent project creates this runtime, gets the model, prepares RGB input, restores
> map coordinates, and applies QA. Use the commands below only to test or maintain
> CoastSeg-lite itself.

CoastSeg-lite is the segmentation and coastal-measurement boundary extracted from
CoastSeg for the `multimethod-sdb` project. It runs one pinned four-class global RGB
SegFormer and provides minimal array-only commands for shoreline extraction,
quality-controlled transect intersections, and CoastSat beach-slope estimation. It
deliberately contains no imagery acquisition, radiometric rendering, project catalog,
sessions, maps, notebooks, or user interface.

This fork derives from
[`SatelliteShorelines/CoastSeg@5066f259`](https://github.com/SatelliteShorelines/CoastSeg/tree/5066f2594cff55e5b5143ac2b4d244e5a11288ff).
This fork keeps the original Git history and GPL-3.0 license. It pins the model
`global_segformer_RGB_4class_14036903`, BEST v2, from Hugging Face revision
`4882d70f4952ab4f38d396d2a37d1b6f4f4a0e68`; the code checks every downloaded file
against the SHA-256 values in the bundled manifest.

## Environment

The inference runtime is intentionally isolated because TensorFlow 2.15 requires
Python 3.11 and NumPy 1.x, while `multimethod-sdb` uses Python 3.12 and NumPy 2.x.
[`uv`](https://docs.astral.sh/uv/) creates and locks the environment:

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen
```

Commands run through `uv run`, so users need no shell activation.

## Model setup

Model weights are runtime data and are not committed:

```bash
UV_CACHE_DIR=.uv-cache uv run coastseg-lite fetch-model \
  --output .models/global_segformer_RGB_4class_14036903
```

Users may run the command again. The code accepts an existing file only when its hash
matches the manifest.

## Inference input and output

The input is a three-channel, 8-bit RGB image already rendered into the CoastSeg model
domain by an upstream adapter. CoastSeg-lite does not interpret reflectance, bands,
cloud masks, CRS, affine transforms, or nodata.

```bash
UV_CACHE_DIR=.uv-cache uv run coastseg-lite infer \
  --input rendered-rgb.png \
  --output classes.npy \
  --model-dir .models/global_segformer_RGB_4class_14036903
```

`classes.npy` is a `uint8[height,width]` array on the input pixel grid:

| Value | Class |
|---:|---|
| 0 | water |
| 1 | whitewater |
| 2 | sediment |
| 3 | other |

A same-stem JSON record stores input and output hashes, model identity, class mapping,
and the exact steps before and after the model. It contains no
nodata value. The parent geospatial adapter is responsible for applying its authoritative
QA mask and writing the final classified COG.

## Coastal measurements

The `shoreline`, `intersections`, and `slopes` commands expose the smallest useful
headless subset of CoastSeg/CoastSat. Their contracts use NumPy arrays and archives;
the parent project owns CRS-aware files and orchestration. Module docstrings retain
the exact upstream module and revision provenance. Shorelines merge class 0 water
with class 1 whitewater and contour the binary boundary at 0.5. Slope estimation
retains CoastSat's Lomb–Scargle method and SciPy Simpson integration.

The preserved CoastSeg behavior is:

1. Pillow RGB decode;
2. bilinear resize to 512 x 512;
3. whole-image mean/adjusted-standard-deviation standardization;
4. HWC-to-CHW transpose;
5. SegFormer inference;
6. bilinear logits resize to the source image dimensions;
7. `argmax` class assignment.

The distributed file named `*_fullmodel.h5` is a weights-only HDF5 file. CoastSeg-lite
bundles the exact SegFormer B0 setup from pinned
`nvidia/mit-b0` revision `80983a413c30d36a39c20203974ae7807835e2b4` and constructs
the model locally before loading the verified weights. Inference never downloads an
architecture or pretrained weights implicitly.

## Development

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

Set `COASTSEG_LITE_MODEL_DIR` to run the real-model regression test.
