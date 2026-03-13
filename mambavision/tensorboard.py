import torch
from tensorboardX import SummaryWriter

class TensorboardLogger(object):
    """Logs data to TensorBoard.

    Attributes:
        writer: SummaryWriter instance used for logging.
        step: Current step in the training process.
    """
def __init__(self, log_dir: str):
        """Initializes a SummaryWriter for tensorboard logging.

        Args:
            log_dir (str): The directory where the logs will be written.

        Returns:
            None

        Raises:
            None
        """
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

def set_step(self, step: int = None) -> None:
        """Sets the step value. If no step is provided, increments the current step.

        Args:
            step (int, optional): The new step value. If None, the current step is incremented. Defaults to None.

        Returns:
            None.

        Raises:
            TypeError: If step is provided and is not an integer.

        """
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("Step must be an integer or None.")
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()