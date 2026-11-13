"""Training-only official contrast cohorts with fixed label/span budgets."""
from collections import Counter, defaultdict
import math
import re
from contractnli_v3_common import *

STOP = set('the a an of to and or in on by at from for with without all some any only shall may not be is are was were have has it its such this that these those each receiving disclosing party parties confidential information agreement including circumstances'.split())
CONDITION = re.compile(r'\b(if|unless|except|provided|subject to|upon|when|only|prior|after|before|need to know)\b', re.I)
PERMISSION_HIDS = {'nda-5','nda-7','nda-17','nda-20','nda-12','nda-13'}


def terms(text):
    return {w[:5] for w in re.findall(r'[a-z]+', text.lower()) if len(w)>2 and w not in STOP}


def contradiction_rule(hid, text):
    """Positive lexical screening, not a claim of new expert label validation."""
    t = re.sub(r'\s+', ' ', text).lower()
    if hid == 'nda-2' and re.search(r'\b(financial|finance|marketing|customers?|business|commercial|payroll|pricing|employees?|personnel)\b', t):
        return 'explicit_nontechnical_scope'
    if hid == 'nda-1' and re.search(r'whether or not|regardless of whether|need not|not (?:be |necessarily )?(?:marked|designated|identified)|should (?:reasonably )?(?:know|understand)|reasonably (?:be )?(?:understood|known)|reasonably (?:expect|understand)', t):
        return 'identification_not_required'
    if hid in {'nda-7','nda-5','nda-17'}:
        action = {'nda-7':r'third.part|any (?:person|individual|other)',
                  'nda-5':r'employees|any (?:person|individual)|another person',
                  'nda-17':r'cop(?:y|ies|ied)|reproduc'}[hid]
        # Avoid lexical candidates that grant a standing purpose/need exception.
        standing = re.search(r'(?:except|other than|unless).{0,300}(?:as required to|reasonably necessary|to the extent necessary|need.to.know|for the purpose|as permitted|as provided|to (?:its |their |the )?representatives)', t)
        if re.search(r'\b(not|no|neither|without)\b', t) and re.search(action, t) and not standing:
            return 'explicit_permission_restriction'
    if hid == 'nda-20' and re.search(r'(?:not|no|without).{0,25}retain|retain.{0,25}(?:no|not)|destroy.{0,80}all.{0,30}cop|return.{0,80}all.{0,30}cop', t) and not re.search(r'except|provided|may retain', t):
        return 'explicit_no_retention'
    if hid == 'nda-10' and re.search(r'may (?:notify|disclose)|free to disclose', t):
        return 'explicit_disclosure_permission'
    if hid == 'nda-4' and re.search(r'(?:free to use|may use).{0,40}any purpose', t):
        return 'explicit_broader_use'
    return None


def take_round_robin(items, count, rank):
    groups = defaultdict(list)
    for item, feat in items:
        groups[feat['hypothesis_id']].append((item, feat))
    for group in groups.values():
        group.sort(key=lambda x:rank(x[1]))
    chosen = []; level = 0
    order = sorted(groups, key=ordered_key)
    while len(chosen) < count:
        batch = [groups[h][level] for h in order if len(groups[h]) > level]
        assert batch, f'Not enough official candidates for requested quota {count}'
        chosen.extend(batch[:count-len(chosen)]); level += 1
    return chosen


def select(candidates):
    old = read_rows(ROOT/'data/curated/contractnli-v1/train.jsonl')
    previous = read_rows(ROOT/'data/curated/contractnli-v2/train.jsonl')
    oldids = {r['id'] for r in old}; previousids = {r['id'] for r in previous}
    byid = {x[0]['id']:x for x in candidates}; assert oldids <= set(byid)
    docs = {x[0]['paper_id']:x[0]['paragraphs'] for x in candidates}
    pt = {d:[terms(p['text']) for p in ps] for d,ps in docs.items()}
    df = Counter(t for parts in pt.values() for t in set().union(*parts))
    idf = {t:math.log((1+len(docs))/(1+n))+1 for t,n in df.items()}
    features = {}; pools = defaultdict(list)
    for item in candidates:
        row, native, gold = item; hid = row['id'].rsplit(':',1)[1]
        parts = {p['id']:p['text'] for p in row['paragraphs']}
        ev = gold['evidence_ids']; text = ' '.join(parts[s] for s in ev)
        q = terms(row['question']); denominator = sum(idf.get(t,1.) for t in q) or 1.
        overlaps = [sum(idf.get(t,1.) for t in q & ts)/denominator for ts in pt[row['paper_id']]]
        category = 'none' if not ev else 'single' if len(ev)==1 else 'multi'
        nums = {int(s[1:]) for s in ev}
        fragment = any(i+1 in nums and not re.search(r'[.!?;:]\s*$',parts[f's{i:04d}'].strip()) for i in nums)
        cond = len(CONDITION.findall(text))
        f = {'id':row['id'],'paper_id':row['paper_id'],'hypothesis_id':hid,'official_label':gold['label'],
            'evidence_kind':category,'evidence_count':len(ev),'condition_markers':cond,
            'conditional_permission_candidate':gold['label']=='Entailment' and hid in PERMISSION_HIDS and cond>0,
            'contradiction_rule':contradiction_rule(hid,text) if gold['label']=='Contradiction' else None,
            'lexical_overlap':round(max(overlaps,default=0.),6),'adjacent_fragment_pair':fragment,
            'in_v1':row['id'] in oldids,'in_v2':row['id'] in previousids,'target_modified':False}
        features[row['id']] = f
        if row['id'] not in oldids:
            pools[(gold['label'],category)].append((item,f))
    selected = [byid[r['id']] for r in old]
    # Final totals: 680 single + 680 multi supported examples and 680 empty-label examples.
    targets = {('Contradiction','single'):300,('Contradiction','multi'):210,
               ('Entailment','single'):380,('Entailment','multi'):470,('NotMentioned','none'):680}
    prior = Counter((features[x[0]['id']]['official_label'],features[x[0]['id']]['evidence_kind']) for x in selected)
    pool_counts = {}; added = []
    for (label,kind), total in targets.items():
        pool = pools[(label,kind)]
        if label == 'Contradiction':
            pool = [(x,f) for x,f in pool if f['contradiction_rule']]
            rank = lambda f:(ordered_key(f['id']),)
        elif label == 'Entailment':
            rank = lambda f:(-int(f['conditional_permission_candidate']),-min(f['condition_markers'],3),
                -int(f['in_v2']),-int(f['adjacent_fragment_pair']),ordered_key(f['id']))
        else:
            rank = lambda f:(-f['lexical_overlap'],-int(f['in_v2']),ordered_key(f['id']))
        pool_counts[label+'_'+kind] = len(pool)
        picked = take_round_robin(pool,total-prior[(label,kind)],rank)
        selected.extend(x for x,f in picked); added.extend(f for x,f in picked)
    selected.sort(key=lambda x:ordered_key(x[0]['id']))
    assert len(selected)==len({x[0]['id'] for x in selected})==2040
    final = [features[x[0]['id']] for x in selected]
    assert Counter((x['official_label'],x['evidence_kind']) for x in final)==targets
    native = {x[0]['id']:{**{k:x[0][k] for k in ['id','paper_id']},'answerable':x[2]['label']!='NotMentioned',
        'reference_index':0,'human_review_status':'pending',**x[1]} for x in selected}
    assert all(native[x['id']]==x for x in old), 'Original v1 row changed'
    assert all(native[x['id']]==x for x in previous if x['id'] in native), 'Shared v2 target changed'
    # Auditable same-hypothesis triads. These are ordinary separate SFT rows, not synthetic text or contrastive loss.
    groups = defaultdict(list)
    for f in final:groups[(f['hypothesis_id'],f['official_label'])].append(f)
    triads = []
    for c in [f for f in added if f['official_label']=='Contradiction']:
        es = groups[(c['hypothesis_id'],'Entailment')]; ns = groups[(c['hypothesis_id'],'NotMentioned')]
        assert es and ns, 'New contradiction lacks selected same-hypothesis comparison'
        e = min(es,key=lambda f:(-int(f['conditional_permission_candidate']),-min(f['condition_markers'],3),ordered_key(c['id']+f['id'])))
        n = min(ns,key=lambda f:(-f['lexical_overlap'],ordered_key(c['id']+f['id'])))
        triads.append({'hypothesis_id':c['hypothesis_id'],'contradiction_id':c['id'],
            'entailment_id':e['id'],'notmentioned_id':n['id'],'contradiction_rule':c['contradiction_rule'],
            'paired_entailment_has_conditional_permission_cue':e['conditional_permission_candidate']})
    assert len(triads)==170
    save(REPORT/'selection-audit.json',{'method':'Official train only; retain v1 core, fixed label/span quotas, lexical explicit-contradiction candidates and conditional-permission priority; round-robin hypotheses.',
        'limitation':'Rule-based candidates, not expert re-annotation; labels and evidence stay official. Same-hypothesis triads may reuse comparator rows and are not independent new examples.',
        'candidate_pool_counts':pool_counts,'targets':{k+'_'+v:n for (k,v),n in targets.items()},
        'retained_v1':1020,'retained_v2':sum(f['in_v2'] for f in final),'new_to_v2':sum(not f['in_v2'] for f in final),
        'selected':final,'added_beyond_v1':added,'contrast_triads':triads,
        'conditional_permission_examples':sum(f['conditional_permission_candidate'] for f in final),
        'conditional_permission_triads':sum(t['paired_entailment_has_conditional_permission_cue'] for t in triads)})
    return selected,added
