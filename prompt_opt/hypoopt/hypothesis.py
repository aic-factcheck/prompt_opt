from copy import deepcopy
from loguru import logger

def update_hypothesis(hypothesis, delta):
    new_hypothesis = deepcopy(hypothesis)
    old_id2idx = {r["rule_id"]: idx for idx, r in enumerate(hypothesis["rules"])}
    n_added, n_updated, n_deleted = 0, 0, 0
    for new_rule in delta["rules"]:
        new_id = new_rule["rule_id"]
        if new_id in old_id2idx:
            if new_rule["title"] == "" or new_rule["rule"] == "":
                logger.debug(f'deleting rule {new_rule["rule_id"]}')
                current_size = len(new_hypothesis["rules"])
                new_hypothesis["rules"] = [r for r in new_hypothesis["rules"] if r["rule_id"] != new_rule["rule_id"]]
                n_deleted += 1
                new_size = len(new_hypothesis["rules"])
                if new_size == current_size:
                    logger.warning("rule not found, nothing deleted!")
                elif new_size != current_size-1:
                    logger.warning(f"more than one rules deleted ({current_size-new_size}), the rules were not unique!")
            else:
                logger.debug(f'updating rule {new_rule["rule_id"]}')
                new_hypothesis["rules"][old_id2idx[new_id]] = new_rule
                n_updated += 1
        else:
            logger.debug(f'adding rule {new_rule["rule_id"]}')
            new_hypothesis["rules"].append(new_rule)
            n_added += 1
    logger.info(f"#rules: {len(new_hypothesis['rules'])}; added: {n_added}, updated: {n_updated}, deleted: {n_deleted}")
    return new_hypothesis
