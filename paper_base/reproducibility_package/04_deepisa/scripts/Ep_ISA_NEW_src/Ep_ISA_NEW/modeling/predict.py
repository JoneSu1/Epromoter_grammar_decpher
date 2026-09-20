import numpy as np
from loguru import logger
from Ep_ISA_NEW.utils import one_hot_encode


def _detect_input_format(model):
    """
    Detect whether model expects (N, 4, L) or (N, L, 4).
    Returns 'channels_first' or 'channels_last'.
    """
    candidates = []
    if hasattr(model, 'input_shape') and model.input_shape:
        candidates.append(model.input_shape)
    if hasattr(model, 'inputs') and model.inputs:
        for inp in model.inputs:
            candidates.append(inp.shape)
    if hasattr(model, 'layers') and model.layers:
        first = model.layers[0]
        if hasattr(first, 'input_shape') and first.input_shape:
            candidates.append(first.input_shape)

    for ishape in candidates:
        shape = tuple(s for s in (ishape if isinstance(ishape, (list, tuple)) else [ishape]))
        if len(shape) == 3:
            if shape[1] == 4:
                return 'channels_first'
            if shape[2] == 4:
                return 'channels_last'

    return 'channels_last'


def _as_2d_prediction(x):
    arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _normalize_model_output(preds):
    if isinstance(preds, dict):
        keys = sorted(preds.keys())
        parts = [_as_2d_prediction(preds[k]) for k in keys]
        return np.concatenate(parts, axis=1)

    if isinstance(preds, (list, tuple)):
        parts = [_as_2d_prediction(p) for p in preds]
        return np.concatenate(parts, axis=1)

    return _as_2d_prediction(preds)


def compute_predictions(model, seqs, device=None, batch_size=1024, tracks=[0]):
    """
    TF/Keras inference for DNA sequences. Auto-detects input format
    (channels_first/last) and output format (single/multi-task/dict).

    device param is accepted but ignored (TF manages GPU placement).
    """
    seqs = list(seqs)
    if len(seqs) == 0:
        return np.empty((0, len(tracks)), dtype=float)

    seq_lens = {len(str(s)) for s in seqs}
    if len(seq_lens) != 1:
        raise ValueError(f"All sequences in a prediction batch must have the same length. Got lengths: {sorted(seq_lens)}")

    x_np = one_hot_encode(seqs)

    input_format = _detect_input_format(model)
    if input_format == 'channels_last':
        x_np = np.transpose(x_np, (0, 2, 1))
    elif input_format == 'channels_first':
        logger.info("Model uses channels_first (N,4,L) input")

    x_np = x_np.astype('float32')

    all_preds = []
    for i in range(0, len(seqs), batch_size):
        batch_x = x_np[i : i + batch_size]
        preds = model(batch_x, training=False)

        preds = _normalize_model_output(preds)
        if preds.shape[0] != len(batch_x):
            raise ValueError(
                f"Model returned {preds.shape[0]} predictions for batch size {len(batch_x)}."
            )
        all_preds.append(preds)

    result = np.concatenate(all_preds, axis=0)

    if result.shape[1] == 1:
        invalid = [t for t in tracks if t != 0]
        if invalid:
            raise ValueError(
                f"Requested tracks {tracks} from a single-output model. Only track 0 is valid."
            )
        return result

    invalid_tracks = [t for t in tracks if t < 0 or t >= result.shape[1]]
    if invalid_tracks:
        raise ValueError(
            f"Requested tracks {tracks}, but model returned {result.shape[1]} outputs."
        )

    return result[:, tracks]


def audit_model_io(model, seqs, tracks):
    preds = compute_predictions(model, seqs, tracks=tracks, batch_size=max(1, len(seqs)))
    return {
        "input_shape": getattr(model, "input_shape", None),
        "output_shape": getattr(model, "output_shape", None),
        "n_sequences": len(seqs),
        "sequence_lengths": sorted({len(str(s)) for s in seqs}),
        "tracks": list(tracks),
        "prediction_shape": tuple(preds.shape),
        "prediction_min": float(np.nanmin(preds)) if preds.size else np.nan,
        "prediction_max": float(np.nanmax(preds)) if preds.size else np.nan,
    }
