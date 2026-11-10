"""Error vocabulary for the orn_g4 package.

Every exception class below carries an `# Errors:` note in its docstring so that
callers can reason about failure shapes without reading implementation. None of
these classes wrap third-party errors silently; each names the producer, the
stream (where applicable), and the operator-facing remediation.
"""


class OrnG4Error(Exception):
    """Base exception for all orn_g4 failures.

    # Errors: raised only by this package; subclasses refine the shape.
    """


class PackageResolutionError(OrnG4Error):
    """The `orn_g4` package could not be imported from any candidate root.

    # Errors: raised by :func:`orn_g4.utils.resolve_package_root` when none of
    ORN_G4_ROOT, /content/orn_g4, /content/Azrael-g4, or a Drive run-directory
    snapshot resolves. Operator remediation: upload the Azrael-g4 folder into
    Colab (or set ORN_G4_ROOT) — see Azrael-g4/README.md.
    """


class ModelSourceError(OrnG4Error):
    """The Kaggle artifact could not be imported or does not match expectations.

    # Errors: raised by :func:`orn_g4.modeling.import_model` when KaggleHub is
    unavailable, the download fails, `config.json` is absent, or the declared
    architecture cannot be discovered. Operator remediation: confirm network +
    Kaggle credentials; the artifact handle and version are recorded verbatim.
    """


class TemplateError(OrnG4Error):
    """The native chat template is missing, inconsistent, or unreadable.

    # Errors: raised by :func:`orn_g4.template.inspect_template` when the
    artifact carries no template file, or when template/dataset token-level
    inconsistency is detected and the operator has not opted into silent repair.
    """


class DatasetError(OrnG4Error):
    """The Google Drive dataset import failed or is structurally unmappable.

    # Errors: raised by :func:`orn_g4.data.import_dataset` when the ZIP is
    absent, fails integrity, lacks `DATASET_RELEASE_MANIFEST.json`, or a shard
    cannot be mapped to arrays+masks. Never guesses the shard layout; names the
    keys actually found so the operator can correct the mapping.
    """


class DataAlignmentError(OrnG4Error):
    """Token/label alignment check failed.

    # Errors: raised by :func:`orn_g4.data.validate_alignment` when input ids,
    attention masks, and label masks disagree in length or when assistant-only
    masking cannot be reconstructed.
    """


class LoRAError(OrnG4Error):
    """LoRA application, coverage measurement, or virtual-merge parity failed.

    # Errors: raised by :func:`orn_g4.lora.apply_lora`,
    :func:`orn_g4.lora.measure_coverage`, or :func:`orn_g4.lora.virtual_merge_parity`
    when target modules are not found, frozen modules receive grads, or the
    adapted matrix diverges beyond tolerance.
    """


class TernaryError(OrnG4Error):
    """Ternary conversion or reconstruction failed.

    # Errors: raised by :func:`orn_g4.ternary.group_scales`,
    :func:`orn_g4.ternary.hard_ternary_forward`, or
    :func:`orn_g4.ternary.sequential_gptq` when group axes are inconsistent,
    codes leave {-1,0,+1}, or reconstruction exceeds the declared budget.
    """


class RotationError(OrnG4Error):
    """Hadamard rotation or its inverse failed parity.

    # Errors: raised by :func:`orn_g4.rotation.apply_hadamard` or
    :func:`orn_g4.rotation.verify_inverse` when rotation+inverse does not
    reproduce the input within tolerance, or when the rotation cannot be
    represented faithfully by the inference path.
    """


class RecoveryError(OrnG4Error):
    """Distillation, residual correction, or selective QAT failed.

    # Errors: raised by :func:`orn_g4.recovery.scale_distillation`,
    :func:`orn_g4.recovery.low_rank_residual`, or
    :func:`orn_g4.recovery.selective_qat` when gradients become non-finite,
    held-out behavior regresses beyond tolerance, or absorption claims cannot
    be reproduced by the hard-ternary export.
    """


class ExportError(OrnG4Error):
    """Artifact export or reload failed.

    # Errors: raised by :func:`orn_g4.export.write_canonical_checkpoint`,
    :func:`orn_g4.export.load_effective_ternary`, or
    :func:`orn_g4.export.pack_ptq1_0` when the packed/ ternary checkpoint does
    not round-trip, or when unchanged FP source weights would be serialized
    under a ternary filename (R-26).
    """


class ControllerError(OrnG4Error):
    """Execution controller state machine failed.

    # Errors: raised by :class:`orn_g4.controller.ExecutionController` when a
    stage budget is exceeded without a checkpoint, a completed-stage checksum
    mismatches on resume, or a stage is requested out of the fixed order.
    """


class BudgetExceededError(ControllerError):
    """A stage exceeded its time/memory/step budget.

    # Errors: the controller checkpoints and advances to the strongest viable
    remaining deliverable; this exception records the budget class and the
    stage that triggered it.
    """


class ChecksumMismatchError(ControllerError):
    """A resumed artifact's checksum did not match the manifest.

    # Errors: raised during resume when a previously completed stage's artifact
    no longer hashes to its recorded value; the stage is re-run rather than
    trusted.
    """


class SourceContractError(OrnG4Error):
    """A source, checkpoint, or dataset contract differs from the run lock.

    # Errors: raised by :func:`orn_g4.source_contract.resolve_hub_sha`,
    :func:`orn_g4.source_contract.enforce_source_lock`,
    :func:`orn_g4.source_contract.validate_release`, or
    :func:`orn_g4.source_contract.assert_template_alignment` when a Hub ref
    resolves to no immutable commit SHA, a pinned input changed inside the
    same run, a training shard escapes the training root or enters
    ``eval_firewall``, a shard checksum is missing or mismatched, or the
    pre-tokenized dataset chat template does not match the model's.
    Operator remediation: pin a new immutable source in a new run directory,
    or restore the originally locked revision.
    """


__all__ = [
    "OrnG4Error",
    "PackageResolutionError",
    "ModelSourceError",
    "TemplateError",
    "DatasetError",
    "DataAlignmentError",
    "LoRAError",
    "TernaryError",
    "RotationError",
    "RecoveryError",
    "ExportError",
    "ControllerError",
    "BudgetExceededError",
    "ChecksumMismatchError",
    "SourceContractError",
]
