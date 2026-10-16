MOSS-Image Base model integration

DiT mathematics and original tensor names derive from OpenMOSS-Team/MOSS-Image-Base,
revision 5f40f07a304541840b6019d2d34d30f966756787, pipeline_moss_image.py.
https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/5f40f07a304541840b6019d2d34d30f966756787

Copyright the OpenMOSS contributors. Licensed under Apache License 2.0 (LICENSE).
Changes: padded true-batch masked attention, activation checkpointing, local component
validation, training-family lifecycle, and the stage-three timestep sampler from the
MOSS-Image Technical Report, Appendix B.1, equation (4).

The Qwen3.5 encoder and AutoencoderKLFlux2 are supplied by Transformers and Diffusers;
this package does not execute model repository Python files.
