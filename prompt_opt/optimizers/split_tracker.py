from collections import defaultdict
from copy import deepcopy
from loguru import logger

from ..utils import *

class SplitTracker:
    """Represents cross-validation-like (target) splits based on a source split if xval_trn_and_dev or just "trn" split is present, 
otherwise just serves the source split.
REMOVED: It also registers all split configurations (split2indices) for each candidate."""
    def __init__(self, dataset_loader, xval_trn_and_dev: bool, rng):
        self.data = dataset_loader.get_data()
        self.xval_trn_and_dev = xval_trn_and_dev
        self.candidate_id2indices = {}
        self.rng = rng
        
        
    def sample_indices(self, source_split, trn_size):
        if self.xval_trn_and_dev:
            # takes the source split (typically "trn") and randomly divides it to two target splits "trn" and "dev")
            assert trn_size < len(self.data[source_split]), (trn_size, len(self.data[source_split]))
            indices = self.rng.permutation(len(self.data[source_split])).tolist()
            return {
                "trn": {"source": source_split, "indices": indices[:trn_size]},
                "dev": {"source": source_split, "indices": indices[trn_size:]},
                }
        else:
            return None  
            
            
    # TODO: remove after checking that resume works
    # def register_candidate_indices(self, candidate_id, split2indices):
    #     if self.xval_trn_and_dev:
    #         self.candidate_id2indices[candidate_id] = split2indices
    
    
    def get_source(self, source_split):
        return self.data[source_split]
    
    
    def get_samples(self, split, split2indices):
        # if xval_trn_and_dev then provide samples for the corresponding target split, otherwise serve the source split 
        if self.xval_trn_and_dev and split in split2indices:
            source_split = split2indices[split]["source"]
            indices = split2indices[split]["indices"]
            ld(f"indices={indices}")
            data = self.get_source(source_split)
            return [data[idx] for idx in indices]
        else:
            # if self.xval_trn_and_dev: # temporary for testing, REMOVE!
                # assert split == "tst", split
            return self.get_source(split)
        
    # TODO: remove after checking that resume works
    # def get_candidate_samples(self, split, candidate_id):
    #     if self.xval_trn_and_dev:
    #         split2indices = self.candidate_id2indices[candidate_id]
    #         return self.get_samples(split, split2indices)
    #     else:
    #         return self.get_source(split)
        
        
    def resample_splits_for_candidate(self, candidate):
        # creates a copy of the candidate and returns its version with resampled splits
        # importantly all sample evaluations are reordered accordingly
        # TODO this may be cleaned up a bit
        candidate = deepcopy(candidate)
        split2indices = candidate["split2indices"]
        ld(f"original split2indices: {split2indices}")
        source2evals = defaultdict(dict) # collect evaluation keyed by the source splits 
        for target_split, rec in split2indices.items():
            source = rec["source"]
            indices = rec["indices"]
            evals = candidate["split"][target_split]
            assert len(indices) == len(evals), (len(indices), len(evals))
            for idx, ev in zip(indices, evals):
                source2evals[source][idx] = ev
        source2indices = {source: self.rng.permutation(len(evals)).tolist() for source, evals in source2evals.items()}
        ld(f'source2indices: {source2indices}')
        source2cnt = defaultdict(lambda: 0)
        for target_split, rec in split2indices.items():
            source = rec["source"]
            indices = rec["indices"]
            start = source2cnt[source]
            new_indices = source2indices[source][start:start+len(indices)]
            ld(f'new_indices: {new_indices}')
            new_evals = [source2evals[source][idx] for idx in new_indices]
            rec["indices"] = new_indices
            candidate["split"][target_split] = new_evals
            source2cnt[source] += len(indices)
        ld(f'new split2indices: {candidate["split2indices"]}')
        return candidate