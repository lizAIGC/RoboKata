# Third-party notices

This supplement adapts Zero-WAM and The-Imitator-Game components.
Third-party copyright notices and licenses are retained in their original files.

- Backend license: `backend/LICENSE`.
- Zero-WAM license: `backend/examples/baselines/zero_wam/zero_wam/wan_va/LICENSE.txt`.
- Zero-WAM source attribution and modifications:
  `backend/examples/baselines/zero_wam/zero_wam/wan_va/UPSTREAM.md`.

Third-party source links identify external dependencies, not this submission's
authors. Model and dataset assets are not included in this supplement.

## Pretrained backbone

The main model is initialized from compatible Zero-WAM pretraining weights.
The `MODEL` path in the training guide refers to that local pretrained-model
directory, not an arbitrary video model. The architecture and action-model
backend build on the attributed upstream components; RoboKata adds its
demonstration-reading, semantic-foresight and indexing adaptations.
Upstream module names and paths remain intact for checkpoint and import compatibility.
