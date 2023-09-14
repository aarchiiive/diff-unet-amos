from .base_dataset import BaseDataset


class AMOSDataset(BaseDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        