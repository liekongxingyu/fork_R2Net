import torch
from torch.utils.data import DataLoader

class MSDataLoader(DataLoader):
    """
    A PyTorch-2.x compatible DataLoader that preserves the original R2Net batch format:
        (lr, hr, filename, idx_scale)
    It also supports multi-scale training by dynamically setting dataset.set_scale().
    In denoising tasks (scale=1), idx_scale will always be 0.
    """

    def __init__(self, args, dataset, batch_size=1, shuffle=False,
                 sampler=None, batch_sampler=None,
                 collate_fn=None, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):

        # Keep scale list (e.g., [1] for denoising, or [2,3,4] for SR)
        self.scale = args.scale
        self.dataset = dataset

        # Default collate function
        if collate_fn is None:
            collate_fn = self._collate_with_scale

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=args.n_threads,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )

    # -----------------------------------------------------------
    # Multi-scale aware collate_fn
    # -----------------------------------------------------------
    def _collate_with_scale(self, batch):
        # batch is a list of elements returned by dataset[i]
        # each element = (lr, hr, filename, idx_scale)
        import random
        
        # For SR: randomly pick a scale
        if len(self.scale) > 1 and getattr(self.dataset, "train", True):
            idx_scale = random.randrange(0, len(self.scale))
            if hasattr(self.dataset, "set_scale"):
                self.dataset.set_scale(idx_scale)
        else:
            idx_scale = 0
        
        # Standard collate (lr, hr, filename, scale) → tensor batch
        lr_batch = torch.stack([b[0] for b in batch])
        hr_batch = torch.stack([b[1] for b in batch])
        filenames = [b[2] for b in batch]
        
        # keep idx_scale to match trainer
        return lr_batch, hr_batch, filenames, idx_scale
