# Model-only lossless weights

Every independent ZIP was decoded and compared against every original uint8 tensor. Native base/config/tokenizer are included under base/. No network or other package is needed.

`python tools/scripts/archive_model_weights.py --threads 2 decode --archive weights/ENDPOINT.xor.zip --base base/MODEL-final/model.safetensors --output NEW.pt`

D aliases reference exact source-bound B4 best weight members and retain their own metadata sidecars. All original nonweight/nonoptimizer metadata is in metadata/*.source_metadata.pt. Optimizer/Adam resume is excluded; separate ROOT durable publication is required.
