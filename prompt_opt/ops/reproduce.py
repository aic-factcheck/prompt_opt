from ..utils import *

class ReproduceMutateOnly:
    def __init__(self, cfg, exp_path, mutate_op):
        logger.info("loading ReproduceMutateOnly...")
        self.cfg_op = cfg
        self.mutate_op = mutate_op
          
          
    def reproduce(self, parents, n_neighbors):
        offspring = []
        for parent in parents:
            op_results = self.mutate_op.mutate(parent, n_neighbors=n_neighbors)
            for neighbor_idx, op_result in enumerate(op_results):
                neighbor = op_result["candidate"]
                if op_result["skipped"]:
                    ld(f"mutation skipped for candidate id={neighbor['id']} {neighbor_idx+1}/{n_neighbors}")
                else:
                    li(f"mutated candidate id={neighbor['id']} {neighbor_idx+1}/{n_neighbors}")
                offspring.append(neighbor)
        return offspring
