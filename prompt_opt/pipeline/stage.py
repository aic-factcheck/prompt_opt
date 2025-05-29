from loguru import logger
import numpy as np

class PipelineStage:
    def __init__(self, cfg):
        logger.info(f"loading {self.__class__.__name__}...")
        self.cfg = cfg

    def _execute_batch(self, sample_fn, input_, load_fn, save_fn, context_init_fn=lambda: {}, rng=None):
        if rng:
            base_seed = rng.randint(0, np.iinfo(np.int32).max//2, dtype=np.int32)
        old_output, old_meta = load_fn()
        # logger.debug(input_)
        # logger.debug(old_output)
        # logger.debug(old_debug_data)
        if old_output:
            if not old_meta:
                old_meta = [None] * len(old_output)
            else:
                assert len(old_output) == len(old_meta), f"inconsistent output/meta: {len(old_output)} != {len(old_meta)}"

            if len(old_output) >= len(input_):
                logger.debug("all samples already processed...")
                return old_output[: len(input_)], old_meta[: len(input_)]
            
            logger.debug(f"{len(old_output)} samples already processed...")

        else:
            old_output = []
            old_meta = []

        context = context_init_fn()
        start = len(old_output)
        idx = start + 1
        output = old_output
        meta = old_meta
        for sample in input_[start:]:
            logger.info(f"processing sample {idx}/{len(input_)}")
            if rng:
                o, m = sample_fn(sample, context, np.random.RandomState(base_seed + idx)) # type: ignore
            else:
                o, m = sample_fn(sample, context)
            output.append(o)
            meta.append(m)
            save_fn(output, meta)
            idx += 1
        return output, meta
