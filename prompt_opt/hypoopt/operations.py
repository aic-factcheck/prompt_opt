from pathlib import Path
import time

from loguru import logger
import numpy as np

from aic_nlp_utils.json import read_json, write_json, read_jsonl, write_jsonl

from prompt_opt.hypoopt.fix_hypo import fix_hypothesis
from prompt_opt.hypoopt.initial_hypo import generate_initial_hypo
from prompt_opt.hypoopt.correct_hypo import correct_hypothesis
from prompt_opt.hypoopt.evaluate import hypothesis_to_instructions, hypothesis_to_instructions2, compute_string_accuracy, compute_object_aligner_accuracy, get_ethos_accs, predict_samples
from prompt_opt.hypoopt.update_hypo import update_hypo_with_new_samples
from prompt_opt.hypoopt.generalize_hypo import generalize_hypo
from prompt_opt.hypoopt.hypothesis import update_hypothesis

class Operations:
    def __init__(self, trn_data, tst_data, llm_opt, llm_tgt, score_samples, predict_schema, n_jobs=10, seed=1234):
        self.trn_data = trn_data
        self.tst_data = tst_data
        self.llm_opt = llm_opt
        self.llm_tgt = llm_tgt
        self.score_samples = score_samples
        self.predict_schema = predict_schema
        self.n_jobs = n_jobs
        self.rng = np.random.RandomState(seed)

    def _evaluate(self, instructions, split, indices=None):
        assert split in ["trn", "tst"]
        eval_data = self.tst_data if split == "tst" else self.trn_data
        indices = list(indices) if indices else list(range(len(eval_data)))
        eval_data = [eval_data[idx] for idx in indices]

        logger.info(f"evaluation on \"{split}\": {len(eval_data)} samples")
        st = time.time()
        preds, messages = predict_samples(self.llm_tgt, instructions, eval_data, predict_schema=self.predict_schema, n_jobs=self.n_jobs)
        accuracy, correct_indices, wrong_indices = self.score_samples(preds, eval_data)
        duration = time.time() - st

        samples = [{"idx": idx, "pred": p, "gold": e["answer"]} for idx, p, e in zip(indices, preds, eval_data)]
        return {
            split: {
                "accuracy": accuracy,
                "samples": samples,
                "duration": duration,
                "messages": messages,
            }
        }

    def initial(self, id_: int, eval_indices=None, init_template_src="modular_classify_v1e"):
        logger.info("initial candidate: generate")
        st = time.time()
        hypo, messages = generate_initial_hypo(self.llm_opt, self.trn_data, init_template_src=init_template_src)
        instructions = hypothesis_to_instructions2(hypo)
        duration = time.time() - st
        ev_trn = self._evaluate(instructions, split="trn", indices=eval_indices)
        ev_tst = self._evaluate(instructions, split="tst")
        candidate = {
            "id": id_,
            "parent": None,
            "op": "init",
            "eval": {
                **ev_trn,
                **ev_tst
                },
            "duration": duration,
            "hypo": hypo,
            "instructions": instructions,
            "messages": messages
        }
        logger.info(f"initial candidate id={id_} created in {duration:.2f}s")
        logger.info(f"TRN acc={ev_trn['trn']['accuracy']:.3f} in {ev_trn['trn']['duration']:.2f}s")
        logger.info(f"TST acc={ev_tst['tst']['accuracy']:.3f} in {ev_tst['tst']['duration']:.2f}s")
        return candidate
    

    def generalize(self, id_, parent, eval_indices=None, template_src="generalize_classify_v2"):
        logger.info("candidate: generalize")
        st = time.time()
        parent_hypo = parent["hypo"]
        hypothesis_generalized, messages = generalize_hypo(self.llm_opt, parent_hypo["rules"], template_src=template_src)
        
        duration = time.time() - st
        
        instructions = hypothesis_to_instructions2(hypothesis_generalized)
        ev_trn = self._evaluate(instructions, split="trn", indices=eval_indices)
        ev_tst = self._evaluate(instructions, split="tst")
        candidate = {
            "id": id_,
            "parent": parent["id"],
            "op": "generalize",
            "eval": {
                **ev_trn,
                **ev_tst
                },
            "duration": duration,
            "hypo": hypothesis_generalized,
            "instructions": instructions,
            "messages": messages
        }
        
        parent_trn_acc = parent["eval"]["trn"]["accuracy"]
        parent_tst_acc = parent["eval"]["tst"]["accuracy"]
        nrules = len(hypothesis_generalized["rules"])
        logger.info(f"generalized candidate id={id_}: {nrules} rules, created in {duration:.2f}s")
        logger.info(f"TRN acc={parent_trn_acc:.3f} -> {ev_trn['trn']['accuracy']:.3f} in {ev_trn['trn']['duration']:.2f}s")
        logger.info(f"TST acc={parent_tst_acc:.3f} -> {ev_tst['tst']['accuracy']:.3f} in {ev_tst['tst']['duration']:.2f}s")
        return candidate
    

    def update(self, id_, parent, start_idx, bsize, eval_indices=None, template_src="update_classify_v2"):
        logger.info("candidate: update")
        st = time.time()
        parent_hypo = parent["hypo"]
        hypothesis_update, messages = update_hypo_with_new_samples(
            self.llm_opt,
            self.trn_data[start_idx:start_idx+bsize],
            parent_hypo["rules"],
            template_src=template_src
        )
        hypo = update_hypothesis(parent_hypo, hypothesis_update)
        duration = time.time() - st
        
        instructions = hypothesis_to_instructions2(hypo)
        ev_trn = self._evaluate(instructions, split="trn", indices=eval_indices)
        ev_tst = self._evaluate(instructions, split="tst")
        candidate = {
            "id": id_,
            "parent": parent["id"],
            "op": "update",
            "eval": {
                **ev_trn,
                **ev_tst
                },
            "duration": duration,
            "hypo": hypo,
            "instructions": instructions,
            "messages": messages
        }
        
        parent_trn_acc = parent["eval"]["trn"]["accuracy"]
        parent_tst_acc = parent["eval"]["tst"]["accuracy"]
        nrules = len(hypothesis_update["rules"])
        logger.info(f"updated candidate id={id_}: {nrules} rules, created in {duration:.2f}s")
        logger.info(f"TRN acc={parent_trn_acc:.3f} -> {ev_trn['trn']['accuracy']:.3f} in {ev_trn['trn']['duration']:.2f}s")
        logger.info(f"TST acc={parent_tst_acc:.3f} -> {ev_tst['tst']['accuracy']:.3f} in {ev_tst['tst']['duration']:.2f}s")
        return candidate
    
    
    def fix(self, id_, parent, eval_indices=None, template_src="fix_classify_v1", max_wrong=3, max_correct=0):
        logger.info("candidate: fix")
        st = time.time()
        parent_hypo = parent["hypo"]

        correct = [] # correct trn prediction
        wrong = [] # wrong predictions
        for ex in parent["eval"]["trn"]['samples']:
            assert ex["gold"] == self.trn_data[ex["idx"]]["answer"]
            query = self.trn_data[ex["idx"]]["query"]
            sample = {"query": query, **ex}
            if sample["pred"] == sample["gold"]:
                correct.append(sample)
            else:
                wrong.append(sample)
                
        correct_samples = list(self.rng.choice(correct, size=min(max_correct, len(correct)), replace=False))
        wrong_samples = list(self.rng.choice(wrong, size=min(max_wrong, len(wrong)), replace=False))
        samples = self.rng.permutation(correct_samples + wrong_samples)
        logger.debug(f"#total: {len(samples)}, #correct: {len(correct_samples)}/{len(correct)}, #wrong: {len(wrong_samples)}/{len(wrong)}")                

        hypothesis_fix, messages = fix_hypothesis(
            self.llm_opt,
            samples,
            parent_hypo["rules"],
            template_src=template_src
        )
        
        hypo = update_hypothesis(parent_hypo, hypothesis_fix)
        duration = time.time() - st
        
        instructions = hypothesis_to_instructions2(hypo)
        ev_trn = self._evaluate(instructions, split="trn", indices=eval_indices)
        ev_tst = self._evaluate(instructions, split="tst")
        candidate = {
            "id": id_,
            "parent": parent["id"],
            "op": "fix",
            "eval": {
                **ev_trn,
                **ev_tst
                },
            "duration": duration,
            "hypo": hypo,
            "instructions": instructions,
            "messages": messages
        }
        
        parent_trn_acc = parent["eval"]["trn"]["accuracy"]
        parent_tst_acc = parent["eval"]["tst"]["accuracy"]
        nrules = len(hypothesis_fix["rules"])
        logger.info(f"fixed candidate id={id_}: {nrules} rules, created in {duration:.2f}s")
        logger.info(f"TRN acc={parent_trn_acc:.3f} -> {ev_trn['trn']['accuracy']:.3f} in {ev_trn['trn']['duration']:.2f}s")
        logger.info(f"TST acc={parent_tst_acc:.3f} -> {ev_tst['tst']['accuracy']:.3f} in {ev_tst['tst']['duration']:.2f}s")
        return candidate