"""Box-selected target buffers and fixed-shape QFormer execution contexts."""

from pathlib import Path
import torch


class CompactFastRunner:
    def __init__(self, path, runner_class):
        self.path = Path(path).with_name("qformer_compact_fp.engine")
        self.runner_class = runner_class
        self.buckets = {}
        self.target_counts = []
        self.bucket_counts = []
        # Allocate all execution contexts before timing/robot observations, including rare full boxes.
        for k in [32, 64, 128, 256]:
            runner = self.runner_class(self.path)
            host_idx = torch.full((7, k), 256, dtype=torch.int32, pin_memory=True)
            host_pad = torch.zeros((7, k), dtype=torch.bool, pin_memory=True)
            feed = {
                "features": torch.zeros(
                    (7, 256, 896), dtype=torch.float16, device="cuda"
                ),
                "target_indices": torch.full(
                    (7, k), 256, dtype=torch.int32, device="cuda"
                ),
                "target_padding": torch.zeros((7, k), dtype=torch.bool, device="cuda"),
            }
            self.buckets[k] = (runner, host_idx, host_pad, feed)
            for _ in range(3):
                runner(feed)
            torch.cuda.synchronize()

    def __call__(self, inputs):
        from deploy.dobottrack.compact_boxes import pack_box_indices, pad_to_bucket

        boxes = inputs["boxes"]
        valid = inputs["valid"]
        if boxes.device.type != "cpu" or valid.device.type != "cpu":
            raise RuntimeError("CompactFastRunner requires CPU history box hook")
        idx, pad = pack_box_indices(boxes.float().numpy(), valid.numpy())
        self.target_counts.append(int(idx.shape[1]))
        idx, pad = pad_to_bucket(idx, pad)
        k = idx.shape[1]
        self.bucket_counts.append(k)
        if k not in self.buckets:
            runner = self.runner_class(self.path)
            host_idx = torch.empty(idx.shape, dtype=torch.int32, pin_memory=True)
            host_pad = torch.empty(pad.shape, dtype=torch.bool, pin_memory=True)
            feed = {
                "features": torch.empty_like(inputs["features"]),
                "target_indices": torch.empty(
                    idx.shape, dtype=torch.int32, device="cuda"
                ),
                "target_padding": torch.empty(
                    pad.shape, dtype=torch.bool, device="cuda"
                ),
            }
            self.buckets[k] = (runner, host_idx, host_pad, feed)
        runner, host_idx, host_pad, feed = self.buckets[k]
        host_idx.copy_(torch.from_numpy(idx))
        host_pad.copy_(torch.from_numpy(pad))
        feed["features"].copy_(inputs["features"])
        feed["target_indices"].copy_(host_idx, non_blocking=True)
        feed["target_padding"].copy_(host_pad, non_blocking=True)
        return runner(feed)
