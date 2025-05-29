from dateutil import parser
from collections import Counter
from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl

def remove_cro_tweets(txt):
    lines = txt.splitlines()
    result_lines = []
    in_removal_section = False

    for line in lines:
        if in_removal_section:
            if line.startswith("--"):
                in_removal_section = False
            continue
        
        if "pic.twitter" in line:
            in_removal_section = True
            continue

        result_lines.append(line)

    return "\n".join(result_lines)


def remove_cro_boilerplate(d):
    d = d.replace("##", "")

    ld = d.lower()
    if "twitter.com" in ld:
        d = remove_cro_tweets(d)
    if "Číst článek" in d:
        d = d.replace("Číst článek", "")
    while "\n\n\n" in d:
        d = d.replace("\n\n\n", "\n\n")
    return d


def import_cro(data_path, prepend_date=False):
    # import time
    import locale
    locale.setlocale(locale.LC_TIME, "cs_CZ")

    recs = read_jsonl(data_path)
    kws = Counter()
    for r in recs:
        txt = f"{r['title']}\n\n{r['abstract']}\n\n{r['text']}"
        txt = remove_cro_boilerplate(txt)

        if prepend_date:
            date_txt = parser.parse(r['date']).strftime("%a, %d %b %Y %H:%M:%S")
            txt = f"{date_txt}\n\n{txt}"
        
        r["ftext"] = txt
        r["keywords"] = Counter([kw.strip() for kw in r["tags"].split(";") if len(kw.strip()) > 0])
        kws.update(r["keywords"])
    return recs, kws


def select_recs(recs, kw_or=[], txt_subs=[], min_len=2000, max_len=10000):
    kw_or = set(kw_or)
    sel = []
    for r in recs:
        txt = r["ftext"].lower()
        if len(txt) > max_len or len(txt) < min_len:
            continue

        for kw in kw_or:
            if kw in r["keywords"]:
                sel.append(r)
                continue
        for sub in txt_subs:
            if sub.lower() in txt:
                sel.append(r)
                continue

    return sel