"""TensorBoard integration for FUSION (optional dependency)."""

from __future__ import annotations

from typing import Optional, Sequence

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    raise ImportError(
        "TensorBoard is required for this integration. "
        "Install it with: pip install tensorboard"
    ) from e


class TensorBoardLogger:
    """Log scalars, embeddings, and images to TensorBoard."""

    def __init__(self, log_dir: str = "runs/fusion") -> None:
        """Initialize the TensorBoard event writer."""
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value at the specified training step."""
        self.writer.add_scalar(tag, value, global_step=step)

    def log_embedding(
        self,
        embeddings,
        metadata: Optional[Sequence] = None,
        step: int = 0,
    ) -> None:
        """Log embeddings with optional metadata."""
        self.writer.add_embedding(
            embeddings,
            metadata=metadata,
            global_step=step,
        )

    def log_images(self, tag: str, images, step: int) -> None:
        """Log a batch of images at the specified training step."""
        self.writer.add_images(tag, images, global_step=step)

    def flush(self) -> None:
        """Flush pending events to disk."""
        self.writer.flush()

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()

    def __enter__(self) -> TensorBoardLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()