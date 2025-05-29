from pathlib import Path

import numpy as np

from prompt_opt.utils import pf, li, ld, lw, le, candidate2prompt_dseek, candidate2prompt_md

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl


def collect_key_values(data, target_key, values=None):
    if values is None:
        values = set()

    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                values.add(int(value))
            collect_key_values(value, target_key, values)

    elif isinstance(data, list):
        for item in data:
            collect_key_values(item, target_key, values)

    return values


def descend(dct, path):
    if not path:
        return dct
    ret = dct
    for k in path.split("."):
        ret = ret[k]
    return ret


def descend_and_set(dct, path, v):
    parent = None
    ret = dct
    for k in path.split("."):
        parent = ret
        ret = ret[k]
    assert parent
    parent[k] = v # type: ignore


def select(obj, path):
    # gives shallow copy of `obj` part
    # ld(f"descend obj={pf(obj)} path={path}")
    idx = path.find(".")
    if idx == -1:
        if path in obj:
            yield obj[path]
        else:
            lw(f"path {path} not found!")
    else:
        key = path[:idx]
        rest = path[idx + 1 :]
        # ld(f"key={key} rest={rest}")
        if key.endswith("[]"):
            for e in obj[key[:-2]]:
                yield from select(e, rest)
        else:
            yield from select(obj[key], rest)


def get_best_candidate(root_dir, metric_name, split="tst"):
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise ValueError(f"Path does not exist: {str(root_dir)}")

    best_score = 0
    best_candidate = None
    best_archive_file = None
    best_idx = -1

    for archive_file in root_dir.rglob("archive.jsonl"):
        archive = read_jsonl(archive_file)
        ld(f"archive_file: {archive_file}")
        ld(f"# candidates: {len(archive)}")
        for idx, candidate in enumerate(archive):
            if "split" not in candidate or split not in candidate["split"]:
                continue
            try:
                scores = [sample["eval"][metric_name]["score"] for sample in candidate["split"][split]]
            except:
                ld("skipping incomplete candidate")
                continue
            # pprint(scores)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_candidate = candidate
                best_archive_file = archive_file
                best_idx = idx
    li(f"best score: {best_score} for candidate idx: {best_idx}, archive: {best_archive_file} for root: {str(root_dir)}")
    return best_candidate


def get_best_prompt(root_dir, metric_name, split="tst", prompt_type="dseek"):
    assert prompt_type in ["dseek", "md"]
    candidate2prompt = {"dseek": candidate2prompt_dseek, "md": candidate2prompt_md}[prompt_type]
    best_candidate = get_best_candidate(root_dir=root_dir, metric_name=metric_name, split=split, prompt_type=prompt_type)
    prompt = candidate2prompt(best_candidate)
    return prompt


def source_configurator(sources):
    date_cfg = {"source": "date"}
    text_cfg = {"source": "text"}
    people_cfg = {"element": "person", "items": {"id": "@id", "name": "@name", "roles": "role"}}
    locations_cfg = {
        "element": "loc",
        "items": {"id": "@id", "name": "@name", "abbreviation": "@abbreviation", "type": "@type"},
    }
    organizations_cfg = {
        "element": "org",
        "items": {"id": "@id", "name": "@name", "abbreviation": "@abbreviation", "type": "@type"},
    }
    events_cfg = {
        "element": "event",
        "items": {
            "id": "@id",
            "event": "@text",
            "subevents": {"element": "subevent", "items": {"subid": "@subid", "event": "@text"}},
        },
    }
    events_only_cfg = {
        "element": "event",
        "items": {"id": "@id", "event": "@text"},
    }
    source_cfg_template = {
        "date": {"cfg": date_cfg},
        "text": {"cfg": text_cfg},
        "people": {"cfg": people_cfg},
        "locations": {"cfg": locations_cfg},
        "organizations": {"cfg": organizations_cfg},
        "events": {"cfg": events_cfg},
        "events_only": {"cfg": events_only_cfg, "key": "events"},
    }
    return {source_cfg_template[k].get("key", k): source_cfg_template[k]["cfg"] for k in sources}
