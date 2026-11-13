"""Select only official training records, using lexical/structural heuristics."""
from collections import Counter,defaultdict
import math
import re
from contractnli_v2_common import *

STOP=set('the a an of to and or in on by at from for with without all some any only shall may not be is are was were have has it its such this that these those each receiving disclosing party parties confidential information agreement including circumstances'.split())
CONDITION=re.compile(r'\b(if|unless|except|provided|subject to|upon|when|only|prior|after|before)\b',re.I)

def terms(text):
    return {w[:5] for w in re.findall(r'[a-z]+',text.lower()) if len(w)>2 and w not in STOP}

def select(candidates):
    old=read_rows(ROOT/'data/curated/contractnli-v1/train.jsonl');oldids={r['id'] for r in old}
    byid={x[0]['id']:x for x in candidates};assert oldids<=set(byid)
    selected=[byid[r['id']] for r in old];audit=[]
    # IDF is computed from official training contract text only, without dev/test data.
    documents={x[0]['paper_id']:x[0]['paragraphs'] for x in candidates}
    vocab={d:terms(' '.join(p['text'] for p in ps)) for d,ps in documents.items()}
    df=Counter(t for ts in vocab.values() for t in ts)
    idf={t:math.log((1+len(vocab))/(1+n))+1 for t,n in df.items()}
    paragraph_terms={d:[terms(p['text']) for p in ps] for d,ps in documents.items()}
    pools={label:defaultdict(list) for label in ('NotMentioned','Entailment')}
    for item in candidates:
        row,rendered,gold=item
        if row['id'] in oldids or gold['label'] not in pools:continue
        hid=row['id'].rsplit(':',1)[1];q=terms(row['question']);den=sum(idf.get(t,1.) for t in q) or 1.
        overlaps=[sum(idf.get(t,1.) for t in q&s)/den for s in paragraph_terms[row['paper_id']]]
        indices=sorted(range(len(overlaps)),key=lambda i:(-overlaps[i],i))[:3]
        overlap=max(overlaps,default=0.)
        paragraphs={p['id']:p['text'] for p in row['paragraphs']};ev=gold['evidence_ids']
        evtext=' '.join(paragraphs[s] for s in ev)
        evnums={int(s[1:]) for s in ev}
        fragment=any(i+1 in evnums and not re.search(r'[.!?;:]\s*$',row['paragraphs'][i]['text'].strip()) for i in evnums)
        feat={'id':row['id'],'paper_id':row['paper_id'],'hypothesis_id':hid,'official_label':gold['label'],
            'lexical_overlap':round(overlap,6),'lexical_candidate_spans':[f's{i:04d}' for i in indices if overlaps[i]>0],
            'official_evidence_count':len(ev),'multi_span':len(ev)>=2,'adjacent_fragment_pair':fragment,
            'condition_marker_count':len(CONDITION.findall(evtext)),'target_modified':False}
        if gold['label']=='NotMentioned':
            rank=(-overlap,-sum(bool(CONDITION.search(row['paragraphs'][i]['text'])) for i in indices),ordered_key(row['id']))
        else:
            rank=(-int(len(ev)>=2),-int(fragment),-min(feat['condition_marker_count'],4),-min(len(ev),5),ordered_key(row['id']))
        pools[gold['label']][hid].append((rank,item,feat))
    for label,groups in pools.items():
        for group in groups.values():group.sort(key=lambda x:x[0])
        chosen=[];hypotheses=sorted(groups,key=ordered_key);level=0
        while len(chosen)<510:
            batch=[groups[h][level] for h in hypotheses if len(groups[h])>level]
            assert batch,'Insufficient unused official training samples'
            chosen.extend(batch[:510-len(chosen)]);level+=1
        selected.extend(x[1] for x in chosen);audit.extend(x[2] for x in chosen)
    selected.sort(key=lambda x:ordered_key(x[0]['id']))
    assert len(selected)==len({x[0]['id'] for x in selected})==2040
    assert Counter(x[2]['label'] for x in selected)=={'Entailment':850,'NotMentioned':850,'Contradiction':340}
    native={x[0]['id']:{**{k:x[0][k] for k in ['id','paper_id']},'answerable':x[2]['label']!='NotMentioned',
        'reference_index':0,'human_review_status':'pending',**x[1]} for x in selected}
    assert all(native[x['id']]==x for x in old),'Original training row changed'
    save(REPORT/'added-selection-audit.json',{'examples':audit,'method':'Training-only lexical overlap for NotMentioned; official multi-span/fragment/condition structure for Entailment; round-robin over hypotheses.',
        'limitation':'Heuristic candidates, not newly human/AI-validated labels. Official labels/evidence are unchanged. Lexically related text is never inserted as evidence for NotMentioned.'})
    return selected,audit
