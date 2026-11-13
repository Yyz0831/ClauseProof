'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {catalog:null,example:null,mode:'sample',health:null,job:null,filter:'all',page:'review',experiment:null,scoreMode:'strict',poll:null,submitting:false};
const labels = {Entailment:'Supports',Contradiction:'Contradicts',NotMentioned:'Not mentioned'};
const pct = n => `${(n * 100).toFixed(1)}%`;

async function api(path,body) {
  const options = body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
  const response = await fetch(path,options);
  let data;
  try { data=await response.json(); } catch { throw new Error('The local service returned an unreadable response.'); }
  if(!response.ok){const error=new Error(data.error?.message || 'The request failed.');error.status=response.status;throw error;}
  return data;
}
function showError(message){$('error-banner').textContent=message;$('error-banner').hidden=false;}
function toast(message){$('toast').textContent=message;$('toast').hidden=false;setTimeout(()=>{$('toast').hidden=true;},4000);}
function selected(){return [...document.querySelectorAll('.hypothesis-option input:checked')].map(el=>el.value);}
function updateRunState(){
  const count=selected().length, running=state.job?.status==='running';
  const hasText=state.mode==='sample' ? !!state.example : $('doc-text').value.trim().length>=30;
  $('run-review').disabled=state.submitting||state.health?.status!=='ready'||running||!count||!hasText;
  $('run-review').querySelector('span').textContent=running?'Review in progress…':'Review agreement';
  $('run-hint').textContent=running?'You can stop after the current statement.':state.health?.status!=='ready'?'The GPU model needs to be ready for live review.':`${count} statement${count===1?'':'s'} selected · one pass per statement`;
  $('toggle-all').textContent=count===17?'Clear all':'Select all';
}
async function refreshHealth(){
  try {
    state.health=await api('/api/health');const h=state.health;
    $('model-status').className=`model-status ${h.status==='ready'?'ready':h.status==='error'||h.status==='offline'?'error':''}`;
    $('model-status').lastElementChild.textContent=h.status==='ready'?(h.busy?'Aya v2 · reviewing':'Aya v2 · GPU ready'):h.status==='loading'?'Aya v2 · loading model':h.status==='offline'?'GPU inference offline':'Aya v2 · unavailable';
  } catch {state.health=null;$('model-status').className='model-status error';$('model-status').lastElementChild.textContent='Local service disconnected';}
  updateRunState();
}
function setSourceMode(mode){
  state.mode=mode;
  for(const name of ['sample','paste']){const b=$(`${name}-mode`);b.classList.toggle('selected',name===mode);b.setAttribute('aria-pressed',String(name===mode));$(`${name}-fields`).hidden=name!==mode;}
  updateRunState();
}
async function restoreEditor(snapshot){
  const ids=new Set(snapshot.hypotheses.map(h=>h.id));
  document.querySelectorAll('.hypothesis-option input').forEach(el=>{el.checked=ids.has(el.value);});
  if(snapshot.origin.kind==='official_example'){
    $('sample-select').value=snapshot.origin.document_id;setSourceMode('sample');await loadExample();
  }else{
    $('doc-title').value=snapshot.title;$('doc-text').value=snapshot.text;
    $('char-count').textContent=`${snapshot.text.length.toLocaleString()} characters`;setSourceMode('paste');
  }
}
async function loadExample(){
  state.example=null;updateRunState();
  const chosen=$('sample-select').value;
  try {
    const example=await api(`/api/examples/${encodeURIComponent(chosen)}`);
    if($('sample-select').value!==chosen)return;
    state.example=example;
    $('sample-meta').textContent=`ContractNLI dev · document ${example.id} · ${example.spans.length} source spans. Previously used in evaluation.`;
  } catch(e){showError(e.message);}
  updateRunState();
}
function badge(label,valid=true){return `<span class="label-badge ${valid?escapeHTML(label):'invalid'}">${valid?escapeHTML(labels[label]||label):'Invalid output'}</span>`;}
function renderJob(){
  const j=state.job;if(!j)return;
  const running=j.status==='running';$('empty-state').hidden=true;$('review-summary').hidden=false;
  $('progress-wrap').hidden=!running;
  $('progress-bar').max=j.total;$('progress-bar').value=j.completed;
  $('progress-text').textContent=j.cancel_requested?`Stopping after the current statement · ${j.completed}/${j.total}`:`Checking statements · ${j.completed} of ${j.total}`;
  $('cancel-review').disabled=!!j.cancel_requested;
  $('result-count').textContent=running?`${j.completed}/${j.total}`:j.status==='completed'?`${j.completed} checked`:j.status==='cancelled'?`Stopped · ${j.completed}/${j.total}`:`Incomplete · ${j.completed}/${j.total}`;
  const counts={Entailment:0,Contradiction:0,NotMentioned:0,invalid_output:0};
  for(const r of j.results)counts[r.prediction?.label||'invalid_output']++;
  $('count-all').textContent=j.completed;$('count-support').textContent=counts.Entailment;$('count-conflict').textContent=counts.Contradiction;$('count-unknown').textContent=counts.NotMentioned;$('count-invalid').textContent=counts.invalid_output;
  $('invalid-filter').hidden=!counts.invalid_output;
  $('snapshot-label').textContent=`${j.document.title} · ${j.document.origin.kind==='official_example'?'Public development sample':'Your text'} · ${j.document.paragraphs.length} spans · ${new Date(j.created_at_utc).toLocaleString()}${j.elapsed_seconds!=null?` · ${j.elapsed_seconds.toFixed(1)}s`:''}. Findings stay attached to this document snapshot.`;
  for(const b of document.querySelectorAll('.summary-chip')){b.classList.toggle('selected',b.dataset.filter===state.filter);b.setAttribute('aria-pressed',String(b.dataset.filter===state.filter));}
  const visible=j.results.filter(r=>state.filter==='all'||(r.prediction?.label||'invalid_output')===state.filter);
  // Keep open raw-output disclosures when progress adds another result.
  const open=new Set([...document.querySelectorAll('.raw-details[open]')].map(el=>el.dataset.id));
  $('finding-list').innerHTML=visible.map(r=>{
    const h=state.catalog.hypotheses.find(x=>x.id===r.hypothesis_id);
    return `<article class="finding"><div class="finding-top"><h3 class="finding-title"><span class="finding-id">${escapeHTML(r.hypothesis_id)}</span>${escapeHTML(h?.title||r.hypothesis_id)}</h3>${badge(r.prediction?.label,r.status==='valid')}</div><p class="finding-question">${escapeHTML(r.hypothesis)}</p>${r.citations.length?`<div class="evidence-row"><span>Source evidence</span>${r.citations.map(c=>`<button class="evidence-link" type="button" data-citation="${escapeHTML(c.id)}" data-result="${escapeHTML(r.hypothesis_id)}" title="Read cited passage ${escapeHTML(c.id)}">${escapeHTML(c.id)} ↗</button>`).join('')}</div>`:r.status==='valid'?'<p class="no-evidence">The model returned insufficient information. No evidence was cited.</p>':'<p class="no-evidence">No validated decision. The original output is preserved below.</p>'}<details class="raw-details" data-id="${escapeHTML(r.hypothesis_id)}" ${open.has(r.hypothesis_id)?'open':''}><summary>Original model output & details</summary><pre>${escapeHTML(r.raw_output)}</pre><p class="diagnostic-line">${r.diagnostics.input_tokens.toLocaleString()} input tokens · ${r.diagnostics.generated_tokens} generated · ${r.diagnostics.generation_seconds}s${r.diagnostics.hit_token_limit?' · Output token limit reached':''}</p>${r.parse_error?`<p class="diagnostic-line">Validation: ${escapeHTML(r.parse_error)}</p>`:''}</details></article>`;
  }).join('');
  $('no-filter-results').hidden=visible.length>0||running;
  $('export-report').disabled=running||!j.completed;$('export-json').disabled=running||!j.completed;
  if(j.error)showError(j.error.message);
  updateRunState();
}
async function runReview(){
  if(state.submitting)return;
  state.submitting=true;
  $('error-banner').hidden=true;
  const payload={hypothesis_ids:selected()};
  if(state.mode==='sample')payload.example_id=state.example.id;
  else{payload.text=$('doc-text').value;payload.title=$('doc-title').value;}
  $('run-review').disabled=true;$('run-hint').textContent='Checking document length…';
  try{
    state.job=await api('/api/jobs',payload);state.filter='all';
    try{sessionStorage.setItem('clauseproof-job',state.job.id);}catch{}
    renderJob();
    $('result-count').scrollIntoView({behavior:'smooth',block:'nearest'});
    pollJob(state.job.id);
  }catch(e){showError(e.message);}finally{state.submitting=false;updateRunState();}
}
async function pollJob(id){
  clearTimeout(state.poll);
  try {
    const j=await api(`/api/jobs/${id}`);
    if(state.job?.id!==id)return;
    state.job=j;renderJob();
    if(j.status==='running')state.poll=setTimeout(()=>pollJob(id),1000);
    else{refreshHealth();toast(j.status==='completed'?`${j.completed} statements checked. Review the cited evidence.`:j.status==='cancelled'?'Review stopped. Completed findings are available.':'Review ended with an error. Completed findings are available.');}
  }catch(e){
    if(e.status===404||e.status===410){
      try{sessionStorage.removeItem('clauseproof-job');}catch{}
      state.job.status='unavailable';state.job.error={message:e.message};renderJob();
    }else{showError(`${e.message} Reconnect to resume checking this review.`);$('progress-text').textContent='Connection interrupted. Waiting to reconnect…';state.poll=setTimeout(()=>pollJob(id),5000);}
  }
}
function showSource(snapshot,highlightIds=[],focusId=null){
  $('source-title').textContent=snapshot.title;
  $('source-caption').textContent=`${snapshot.origin?.span_policy||'Official ContractNLI spans'} · ${snapshot.paragraphs.length} spans${highlightIds.length?` · ${highlightIds.length} cited`:''}. Highlighting shows what was cited; it does not independently validate the conclusion.`;
  $('source-body').replaceChildren();
  const chars=[...(snapshot.text||'')];let last=0;
  for(const p of snapshot.paragraphs){
    if(p.start!=null&&p.start>last){const gap=document.createElement('div');gap.className='source-gap';gap.textContent=chars.slice(last,p.start).join('');$('source-body').append(gap);}
    const article=document.createElement('div');article.className='source-paragraph'+(highlightIds.includes(p.id)?' highlight':'');article.id=`source-${p.id}`;
    const code=document.createElement('span');code.className='source-id';code.textContent=p.id;
    article.append(code,document.createTextNode(p.text));$('source-body').append(article);
    last=p.end??last;
  }
  if(last&&last<chars.length){const gap=document.createElement('div');gap.className='source-gap';gap.textContent=chars.slice(last).join('');$('source-body').append(gap);}
  if(!$('source-dialog').open)$('source-dialog').showModal();
  if(focusId)requestAnimationFrame(()=>document.getElementById(`source-${focusId}`)?.scrollIntoView({block:'center'}));
  else $('source-body').scrollTop=0;
}
function previewExample(){
  const ex=state.example;if(!ex)return;
  const snapshot={title:ex.title,text:ex.text,paragraphs:[],origin:{span_policy:'Official ContractNLI spans'}};
  // JavaScript uses UTF-16 offsets; Python/data offsets are Unicode code points.
  const chars=[...ex.text];snapshot.paragraphs=ex.spans.map(([start,end],i)=>({id:`s${String(i).padStart(4,'0')}`,start,end,text:chars.slice(start,end).join('')}));
  showSource(snapshot);
}
function download(content,name,mime){
  const url=URL.createObjectURL(new Blob([content],{type:mime}));const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);
}
function exportReport(json=false){
  const j=state.job;if(!j||!j.completed||j.status==='running')return;
  const name=`clauseproof-${j.id.slice(0,8)}`;
  const exportData={application:'ClauseProof v0.1',exported_at_utc:new Date().toISOString(),...j,
    interpretation:'Model judgments, not independent verification. NotMentioned is insufficient information, not permission. Invalid outputs were not repaired.',
    completeness:`${j.completed} of ${j.total} selected statements completed; status ${j.status}.`};
  if(json){download(JSON.stringify(exportData,null,2),`${name}.json`,'application/json');return;}
  const e=escapeHTML;
  const content=`<!doctype html><html lang="en"><meta charset="utf-8"><title>ClauseProof — ${e(j.document.title)}</title><style>body{max-width:900px;margin:50px auto;padding:0 25px;font:14px/1.7 system-ui;color:#243d30}h1{font:38px Georgia}h2{margin-top:35px}h3{font-size:15px}article{break-inside:avoid;border-top:1px solid #d7dfd2;padding:18px 0}blockquote{margin:10px 0;padding:12px;background:#eef3e8;white-space:pre-wrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;padding:12px;background:#f3f4ef}.meta{color:#697762;font-size:12px}.source{white-space:pre-wrap}@media print{body{margin:0;font-size:11px}h1{font-size:27px}details{display:block}}</style><body><p class="meta">CLAUSEPROOF / LOCAL REVIEW REPORT</p><h1>${e(j.document.title)}</h1><p>${e(exportData.completeness)}</p><p class="meta">${e(j.created_at_utc)} · Aya Expanse 8B / ContractNLI v2</p><p>${e(exportData.interpretation)}</p><p class="meta">Source: ${e(j.document.origin.kind)} · SHA-256 ${e(j.document.source_sha256)}</p>${j.results.map(r=>`<article><h3>${e(r.hypothesis)}</h3><p><strong>${e(r.prediction?labels[r.prediction.label]:'Invalid output')}</strong></p>${r.citations.map(c=>`<blockquote><strong>${e(c.id)}</strong><br>${e(c.text)}</blockquote>`).join('')}${!r.citations.length?'<p>No validated citation supplied.</p>':''}<p class="meta">Original output — unchanged</p><pre>${e(r.raw_output)}</pre>${r.parse_error?`<p>Validation error: ${e(r.parse_error)}</p>`:''}</article>`).join('')}<h2>Full source snapshot</h2><div class="source">${e(j.document.text)}</div><h2>Model provenance</h2><pre>${e(JSON.stringify(j.model,null,2))}</pre><p class="meta">Exported ${e(exportData.exported_at_utc)}. Open this file in a browser and use Print to save a PDF.</p></body></html>`;
  download(content,`${name}.html`,'text/html');toast('Report downloaded. Open it in a browser to print or save as PDF.');
}
async function navigate(page,push=true){
  state.page=page;
  $('review-page').hidden=page!=='review';$('experiments-page').hidden=page!=='experiments';
  for(const a of document.querySelectorAll('nav [data-page]')){a.classList.toggle('active',a.dataset.page===page);if(a.dataset.page===page)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');}
  if(push)history.pushState({},'',page==='experiments'?'/experiments':'/');
  document.title=page==='experiments'?'Experiments — ClauseProof':'ClauseProof — Every conclusion, a clause.';
  if(page==='experiments'&&!state.experiment){
    try{state.experiment=await api('/api/experiments');renderExperiment();renderCaseList();}
    catch(e){toast(e.message);}
  }
}
function metricValue(m,key){return key==='evidence_f1'?m.evidence_on_gold_supported.f1:m[key];}
function renderExperiment(){
  const x=state.experiment;if(!x)return;const arms=x.modes[state.scoreMode],v2=arms.v2,base=arms.base;
  for(const mode of ['strict','content']){const b=$(`${mode}-mode`);b.classList.toggle('selected',mode===state.scoreMode);b.setAttribute('aria-pressed',String(mode===state.scoreMode));}
  $('scoring-description').textContent=state.scoreMode==='strict'?'Malformed outputs remain failures; nothing is repaired.':'Diagnostic only: remove one complete Markdown fence. Invalid JSON, labels and citation IDs still fail.';
  const specs=[['Label accuracy','label_accuracy','All 306 decisions'],['Balanced class F1','label_macro_f1','Mean F1 across 3 labels'],['Evidence overlap','evidence_f1','176 reference-evidence cases'],['Label + exact evidence','joint_exact_success_rate','All 306 decisions']];
  $('metric-cards').innerHTML=specs.map(([name,key,scope])=>`<div class="metric-card"><span class="metric-title">${name} · v2</span><strong>${pct(metricValue(v2,key))}</strong><p><span class="change">+${((metricValue(v2,key)-metricValue(base,key))*100).toFixed(1)} pp</span> vs original</p><p>${scope}</p></div>`).join('');
  const metric=$('metric-select').value;
  $('model-chart').innerHTML=Object.entries(arms).map(([arm,m])=>`<div class="bar-row ${arm==='v2'?'chosen':''}"><div class="bar-label">${escapeHTML(x.models[arm].title)}${arm==='v2'?'<span>IN USE</span>':''}</div><div class="bar-track"><div class="bar-fill" style="width:${metricValue(m,metric)*100}%"></div></div><span class="bar-value">${pct(metricValue(m,metric))}</span></div>`).join('');
  $('chart-note').textContent=metric==='evidence_f1'?'Evidence F1 is official span overlap across 176 non-neutral cases. It does not establish semantic support.':metric==='joint_exact_success_rate'?'Joint exact requires a valid output, the correct label, and the exact official evidence set. This is a conservative mechanical metric.':'Same 306 decisions from 18 contracts, generated once per model. No automatic retries.';
  $('metrics-body').innerHTML=Object.entries(arms).map(([arm,m])=>`<tr class="${arm==='v2'?'chosen':''}"><td>${escapeHTML(x.models[arm].title)}</td><td>${m.format_valid}/306</td><td>${pct(m.label_accuracy)}</td><td>${pct(m.label_macro_f1)}</td><td>${pct(m.evidence_on_gold_supported.f1)}</td><td>${m.joint_exact_success}/306</td></tr>`).join('');
  $('experiment-limits').innerHTML=x.limitations.map(t=>`<li>${escapeHTML(t)}</li>`).join('');
}
function renderCaseList(){
  const x=state.experiment;if(!x)return;const filter=$('case-filter').value;
  const cases=x.cases.filter(c=>filter==='all'||filter==='improved'&&c.outputs.v2.label_correct&&!c.outputs.base.label_correct||filter==='regressed'&&!c.outputs.v2.label_correct&&c.outputs.base.label_correct||filter==='v2_wrong'&&!c.outputs.v2.label_correct||filter==='citation'&&c.outputs.v2.label_correct&&!c.outputs.v2.joint_exact);
  $('case-select').innerHTML=cases.map(c=>`<option value="${escapeHTML(c.id)}">Document ${escapeHTML(c.document_id)} · ${escapeHTML(c.id.split(':').at(-1))} · ${escapeHTML(c.hypothesis)}</option>`).join('');
  $('case-count').textContent=`${cases.length} of 306 cases · strict scoring`;
  renderCase();
}
function renderCase(){
  const c=state.experiment.cases.find(c=>c.id===$('case-select').value);
  if(!c){$('case-detail').textContent='No cases match this filter.';return;}
  $('case-detail').innerHTML=`<h3 class="case-question">${escapeHTML(c.hypothesis)}</h3><p class="reference-line">Official reference: <strong>${escapeHTML(labels[c.reference.label])}</strong> · evidence: ${escapeHTML(c.reference.evidence_ids.join(', ')||'none')} · document ${escapeHTML(c.document_id)}</p><div class="case-outputs">${Object.entries(c.outputs).map(([arm,r])=>`<article class="case-output ${arm==='v2'?'chosen':''}"><div class="case-output-top"><strong>${escapeHTML(state.experiment.models[arm].title)}</strong><span>${arm==='v2'?'IN USE':''}</span></div>${badge(r.prediction?.label,r.format_valid)}<p class="case-state">${r.joint_exact?'Label + official evidence match':r.label_correct?'Label matches · evidence differs':'Strict label score: incorrect / invalid'}</p><pre>${escapeHTML(r.raw)}</pre>${r.parse_error?`<p class="case-state">${escapeHTML(r.parse_error)}</p>`:''}</article>`).join('')}</div><details class="case-source-details"><summary>Read cited and reference source spans (${c.source_spans.length})</summary>${c.source_spans.map(p=>`<p><strong>${escapeHTML(p.id)}</strong><br>${escapeHTML(p.text)}</p>`).join('')}</details>`;
}

document.addEventListener('click',event=>{
  const nav=event.target.closest('[data-page]');if(nav){event.preventDefault();navigate(nav.dataset.page);window.scrollTo({top:0});return;}
  const cite=event.target.closest('[data-citation]');if(cite&&state.job){const r=state.job.results.find(x=>x.hypothesis_id===cite.dataset.result);showSource(state.job.document,r.citations.map(p=>p.id),cite.dataset.citation);return;}
  const filter=event.target.closest('[data-filter]');if(filter){state.filter=filter.dataset.filter;renderJob();}
});
$('sample-mode').onclick=()=>setSourceMode('sample');$('paste-mode').onclick=()=>setSourceMode('paste');$('sample-select').onchange=loadExample;
$('toggle-all').onclick=()=>{const all=selected().length===17;document.querySelectorAll('.hypothesis-option input').forEach(el=>{el.checked=!all;});updateRunState();};
$('hypothesis-list').onchange=updateRunState;$('doc-text').oninput=()=>{$('char-count').textContent=`${$('doc-text').value.length.toLocaleString()} characters`;updateRunState();};
$('file-input').onchange=async()=>{const file=$('file-input').files[0];if(!file)return;if(!file.name.toLowerCase().endsWith('.txt')||file.size>700000){showError('Choose a .txt file smaller than 700 KB.');return;}try{$('doc-text').value=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());if(!$('doc-title').value)$('doc-title').value=file.name.replace(/\.txt$/i,'');$('doc-text').dispatchEvent(new Event('input'));}catch{showError('The text file could not be read. Use UTF-8 encoded plain text.');}finally{$('file-input').value='';}};
$('run-review').onclick=runReview;$('preview-source').onclick=previewExample;$('close-source').onclick=()=>$('source-dialog').close();
$('cancel-review').onclick=async()=>{if(!state.job)return;try{state.job=await api(`/api/jobs/${state.job.id}/cancel`,{});renderJob();}catch(e){showError(e.message);}};
$('export-report').onclick=()=>exportReport(false);$('export-json').onclick=()=>exportReport(true);
$('strict-mode').onclick=()=>{state.scoreMode='strict';renderExperiment();};$('content-mode').onclick=()=>{state.scoreMode='content';renderExperiment();};
$('metric-select').onchange=renderExperiment;$('case-filter').onchange=renderCaseList;$('case-select').onchange=renderCase;
window.addEventListener('popstate',()=>navigate(location.pathname==='/experiments'?'experiments':'review',false));
async function init(){
  await refreshHealth();setInterval(refreshHealth,10000);
  try{state.catalog=await api('/api/catalog');
    $('hypothesis-list').innerHTML=state.catalog.hypotheses.map(h=>`<label class="hypothesis-option" title="${escapeHTML(h.hypothesis)}"><input type="checkbox" value="${escapeHTML(h.id)}" checked><span>${escapeHTML(h.title)}</span></label>`).join('');
    $('sample-select').innerHTML=state.catalog.examples.map(d=>`<option value="${escapeHTML(d.id)}">${escapeHTML(d.title)} · document ${escapeHTML(d.id)}</option>`).join('');
    await loadExample();
    let lastJob=null;try{lastJob=sessionStorage.getItem('clauseproof-job');}catch{}
    if(lastJob){
      try{state.job=await api(`/api/jobs/${encodeURIComponent(lastJob)}`);await restoreEditor(state.job.document);renderJob();if(state.job.status==='running')pollJob(lastJob);}
      catch{try{sessionStorage.removeItem('clauseproof-job');}catch{}toast('The previous review is unavailable. Start a new review.');}
    }
  }catch(e){showError(e.message);}
  navigate(location.pathname==='/experiments'?'experiments':'review',false);
}
init();
