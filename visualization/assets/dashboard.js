'use strict';
const reports = JSON.parse(document.getElementById('report-data').textContent);
const $ = id => document.getElementById(id);
const colors = ['#087f76','#ba713a','#546bb1','#985c93','#687b39','#337f9c','#b64f62','#7761a5','#716050','#30846a','#7d778a'];
const descriptions = {
  nDCG: 'Ranking quality: rewards relevant documents near the top, normalized against the ideal ranking.',
  AP: 'Average precision across relevant results. The average over queries is MAP; full-ranking AP uses the submitted retrieval depth.',
  RR: 'Reciprocal rank of the first relevant result. The average over queries is MRR.',
  P: 'Precision: relevant results divided by the requested cutoff, even when fewer results are returned.',
  R: 'Recall: fraction of known relevant documents retrieved.',
  Success: 'Fraction of queries with at least one relevant result.',
  Judged: 'Judgment coverage: fraction of returned documents with judgments. This measures assessment coverage, not retrieval quality.',
  Bpref: 'Preference for judged relevant over judged nonrelevant documents. Limited information when judgments are positive-only.',
  Rprec: 'Precision at the number of known relevant documents for each query.',
  SetP: 'Precision of the complete submitted result set, ignoring order.',
  SetR: 'Recall of the complete submitted result set, ignoring order.',
  SetF: 'F1 of the complete submitted result set, ignoring order.'
};
let report, selected = new Set(), tab = 'overview', sortKey = 'score', descending = true;
let page = 0, queryRows = [], queryGeneration = 0;
const cache = new Map(), pending = new Map();
const fmt = value => value.toFixed(6);
const deltaText = value => (value > 0 ? '+' : '') + fmt(value);
const shortName = name => name.replace(/\.(trec|jsonl)(\.gz)?$/, '');
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function options(id, entries, preferred) {
  const select = $(id);
  select.replaceChildren(...entries.map(([value, label]) => {
    const option = element('option', label); option.value = value; return option;
  }));
  if (entries.some(([value]) => value === preferred)) select.value = preferred;
}
function currentMetric() { return $('metric').value; }
function baseline() { return report.runs.find(run => run.name === $('baseline').value); }
function activeRuns() { return report.runs.filter(run => selected.has(run.name)); }
function color(run) { return colors[report.runs.indexOf(run) % colors.length]; }
function differenceCell(value) {
  return element('td', deltaText(value), value > 0 ? 'positive' : value < 0 ? 'negative' : '');
}
function setDataset() {
  const entries = reports.map((r, i) => [String(i), `${r.split || 'External judgments'} · ${r.source}`])
    .filter(([i]) => (reports[Number(i)].dataset || 'External judgments') === $('dataset').value);
  options('report', entries);
  setReport();
}
function setReport() {
  report = reports[Number($('report').value)];
  selected = new Set(report.runs.map(run => run.name));
  options('metric', report.metrics.map(m => [m, m]), report.metrics.includes(currentMetric()) ? currentMetric() : 'nDCG@10');
  const names = report.runs.map(run => [run.name, shortName(run.name)]);
  options('baseline', names, names[0][0]);
  const leader = [...report.runs].sort((a,b) => b.aggregate[currentMetric()] - a.aggregate[currentMetric()])[0];
  options('candidate', names, leader.name);
  $('models').replaceChildren(...report.runs.map(run => {
    const label = element('label');
    const checkbox = element('input'); checkbox.type = 'checkbox'; checkbox.checked = true;
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) selected.add(run.name); else selected.delete(run.name);
      render();
    });
    label.append(checkbox, document.createTextNode(shortName(run.name))); return label;
  }));
  cache.clear();
  render();
}
function render() {
  const metric = currentMetric(), base = baseline(), runs = activeRuns();
  const family = metric.split('@')[0];
  $('description').textContent = `${descriptions[family] || 'Saved evaluation metric.'} ${metric.includes('@') ? 'Cutoff: ' + metric.split('@')[1] + '.' : ''}`;
  $('comparison-hint').textContent = (family === 'Judged' ? 'Higher scores mean more judgment coverage, not better retrieval. ' : 'Higher scores are better. ') + 'Differences are absolute changes from the baseline.';
  $('selection-count').textContent = `${runs.length} / ${report.runs.length} selected`;
  const best = [...runs].sort((a,b) => b.aggregate[metric] - a.aggregate[metric])[0];
  const coverage = base.coverage;
  const cards = [
    [metric === 'Judged' || family === 'Judged' ? 'Highest judgment coverage' : `Highest ${metric}`, best ? fmt(best.aggregate[metric]) : '—', best ? shortName(best.name) : 'Select a model to compare'],
    ['Judged queries', String(coverage.judged_queries), `${report.dataset || 'External dataset'} · ${report.split || 'External split'}`],
    ['Baseline score', fmt(base.aggregate[metric]), shortName(base.name)]
  ];
  $('cards').replaceChildren(...cards.map(([label,value,note]) => {
    const card = element('div', undefined, 'card');
    card.append(element('span', label, 'label'),element('strong', value),element('span',note,'note')); return card;
  }));
  renderOverview(); renderDepth(); renderDetails();
  if (tab === 'queries') loadComparison();
}
function renderOverview() {
  const metric = currentMetric(), baseValue = baseline().aggregate[metric];
  const runs = activeRuns();
  const chartRuns = [...runs].sort((a,b) => b.aggregate[metric] - a.aggregate[metric]);
  const max = Math.max(...chartRuns.map(run => run.aggregate[metric]), 0.000001);
  $('bars').replaceChildren(...chartRuns.map(run => {
    const row = element('div', undefined, 'bar-row');
    const track = element('div', undefined, 'bar-track');
    const fill = element('div', undefined, 'bar-fill');
    fill.style.width = `${Math.max(0,run.aggregate[metric]) / max * 100}%`; fill.style.background = color(run);
    track.append(fill);
    row.append(element('span',shortName(run.name),'bar-label'),track,element('span',fmt(run.aggregate[metric]),'bar-value'));
    return row;
  }));
  if (!runs.length) $('bars').append(element('p','Select at least one model to display results.','muted'));
  runs.sort((a,b) => {
    const result = sortKey === 'name' ? a.name.localeCompare(b.name) : a.aggregate[metric] - b.aggregate[metric];
    return descending ? -result : result;
  });
  $('ranking').replaceChildren(...runs.map(run => {
    const row = element('tr'), c = run.coverage;
    row.append(element('td',shortName(run.name)),element('td',fmt(run.aggregate[metric])),differenceCell(run.aggregate[metric]-baseValue),element('td',`${c.evaluated_queries_with_results} / ${c.judged_queries}`),element('td',`${c.minimum_depth}–${c.maximum_depth}`)); return row;
  }));
}
function svgElement(tag, attributes, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg',tag);
  for (const [key,value] of Object.entries(attributes)) node.setAttribute(key,String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}
function renderDepth() {
  const family = currentMetric().split('@')[0], runs = activeRuns();
  const metrics = report.metrics.filter(m => m.startsWith(family+'@')).sort((a,b) => Number(a.split('@')[1])-Number(b.split('@')[1]));
  $('curve').replaceChildren(); $('legend').replaceChildren(); $('depth-table').replaceChildren();
  if (!metrics.length || !runs.length) {
    $('curve').append(element('p', !runs.length ? 'Select a model to display curves.' : `No saved cutoff values for ${family}. Choose a metric such as nDCG@10 or R@100.`,'muted')); return;
  }
  const cutoffs = metrics.map(m => Number(m.split('@')[1]));
  const maximum = Math.max(0.01,...runs.flatMap(run => metrics.map(m => run.aggregate[m])));
  const low = Math.log10(cutoffs[0]), high = Math.log10(cutoffs[cutoffs.length-1]);
  const x = cutoff => high === low ? 490 : 60+(Math.log10(cutoff)-low)/(high-low)*860;
  const y = value => 310-value/maximum*265;
  const svg = svgElement('svg',{viewBox:'0 0 960 365',role:'img','aria-label':`${family} across retrieval cutoffs. Exact values appear in the table below.`});
  for (let i=0;i<=4;i++) {
    const value = maximum*i/4, position = y(value);
    svg.append(svgElement('line',{x1:60,x2:920,y1:position,y2:position,stroke:'#e1e8e4'}));
    svg.append(svgElement('text',{x:48,y:position+4,'text-anchor':'end'},value.toFixed(3)));
  }
  for (const cutoff of cutoffs) svg.append(svgElement('text',{x:x(cutoff),y:333,'text-anchor':'middle'},String(cutoff)));
  svg.append(svgElement('text',{x:490,y:358,'text-anchor':'middle'},'Retrieval cutoff K (log scale)'));
  for (const run of runs) {
    svg.append(svgElement('polyline',{points:metrics.map((m,i)=>`${x(cutoffs[i])},${y(run.aggregate[m])}`).join(' '),fill:'none',stroke:color(run),'stroke-width':2}));
    metrics.forEach((m,i) => {
      const dot = svgElement('circle',{cx:x(cutoffs[i]),cy:y(run.aggregate[m]),r:4,fill:color(run)});
      dot.append(svgElement('title',{},`${shortName(run.name)} · ${m}: ${fmt(run.aggregate[m])}`)); svg.append(dot);
    });
    const label = element('span'); const swatch = element('span',undefined,'swatch'); swatch.style.background=color(run);
    label.append(swatch,document.createTextNode(shortName(run.name))); $('legend').append(label);
  }
  $('curve').append(svg);
  const table = element('table'), head=element('thead'), header=element('tr'), body=element('tbody');
  header.append(element('th','Model'),...metrics.map(m=>element('th',m))); head.append(header);
  for (const run of runs) {const row=element('tr');row.append(element('td',shortName(run.name)),...metrics.map(m=>element('td',fmt(run.aggregate[m]))));body.append(row);}
  table.append(head,body); $('depth-table').append(table);
}
function renderDetails() {
  const fields = [
    ['Dataset',report.dataset || 'External judgments'],['Prepared split',report.split || 'External judgments'],
    ['Split interpretation',report.dataset === 'jurifindit' && report.split === 'test' ? 'Original JuriFindIT validation split' : ['msmarco','mmarco-it'].includes(report.dataset) && report.split === 'test' ? 'Original MARCO dev split' : 'As recorded by the evaluator'],
    ['Evaluation timestamp',report.created_at],['Report source',report.source],['Metric count',report.metrics.length],
    ['Relevant grade threshold',report.relevance_level],['Query policy',report.query_policy],['Tie policy',report.tie_policy],['nDCG gain',report.ndcg_gain],
    ['Metric providers',(report.providers || []).join(', ')],['Software versions',JSON.stringify(report.versions)],['Python version',report.python_version],
    ['Baseline coverage',JSON.stringify(baseline().coverage)],['Interpretation','Unjudged documents count as nonrelevant for conventional metrics. Validation comparisons do not establish held-out performance.']
  ];
  $('metadata').replaceChildren(...fields.flatMap(([key,value])=>[element('dt',key),element('dd',String(value ?? 'Not recorded'))]));
}
window.receiveQueries = rows => {
  const file = document.currentScript.dataset.file;
  if (pending.has(file)) pending.get(file).rows = rows;
};
function loadQueries(file) {
  if (cache.has(file)) return Promise.resolve(cache.get(file));
  if (pending.has(file)) return pending.get(file).promise;
  const job = {};
  job.promise = new Promise((resolve,reject)=>{
    const script = element('script'); script.src=file; script.dataset.file=file;
    script.onload=()=>{
      pending.delete(file);script.remove();
      if (!job.rows) {reject(new Error('Query payload is missing data.'));return;}
      cache.set(file,job.rows);
      while (cache.size>2) cache.delete(cache.keys().next().value);
      resolve(job.rows);
    };
    script.onerror=()=>{pending.delete(file);script.remove();reject(new Error('Could not load query data. Keep the generated query files beside index.html.'));};
    pending.set(file,job);document.head.append(script);
  });
  return job.promise;
}
async function loadComparison() {
  const generation = ++queryGeneration;
  const currentReport = report, metric = currentMetric();
  const base = baseline(), candidate = report.runs.find(run=>run.name===$('candidate').value);
  queryRows=[];page=0;renderQueries();$('query-status').textContent='Loading selected models…';
  try {
    const [left,right] = await Promise.all([loadQueries(base.query_file),loadQueries(candidate.query_file)]);
    if (generation !== queryGeneration || currentReport !== report || tab !== 'queries') return;
    const column = report.metrics.indexOf(metric)+1;
    const baselineValues = new Map(left.map(row=>[row[0],row[column]]));
    queryRows=right.map(row=>({id:row[0],base:baselineValues.get(row[0]),candidate:row[column],delta:row[column]-baselineValues.get(row[0])}));
    renderQueries();
  } catch(error) {
    if (generation===queryGeneration) $('query-status').textContent=error.message;
  }
}
function renderQueries() {
  const wins=queryRows.filter(row=>row.delta>1e-12).length;
  const losses=queryRows.filter(row=>row.delta < -1e-12).length;
  const ties=queryRows.length-wins-losses;
  const outcome=$('outcome').value, search=$('search').value.toLowerCase();
  const rows=queryRows.filter(row=>row.id.toLowerCase().includes(search) && (outcome==='all' || outcome==='wins' && row.delta>1e-12 || outcome==='losses' && row.delta < -1e-12 || outcome==='ties' && Math.abs(row.delta)<=1e-12));
  const order=$('query-sort').value;
  rows.sort((a,b)=>order==='id' ? a.id.localeCompare(b.id) : order==='gain' ? b.delta-a.delta : order==='loss' ? a.delta-b.delta : Math.abs(b.delta)-Math.abs(a.delta));
  const pages=Math.max(1,Math.ceil(rows.length/50)); page=Math.min(page,pages-1);
  $('query-status').textContent=`${wins} wins · ${ties} ties · ${losses} losses across ${queryRows.length} queries. ${rows.length} match the filters. Ties use a tolerance of 0.000000000001; displayed values are rounded.`;
  $('query-rows').replaceChildren(...rows.slice(page*50,(page+1)*50).map(row=>{
    const tr=element('tr');tr.append(element('td',row.id),element('td',fmt(row.base)),element('td',fmt(row.candidate)),differenceCell(row.delta));return tr;
  }));
  $('page').textContent=`Page ${page+1} of ${pages}`;$('previous').disabled=page===0;$('next').disabled=page>=pages-1;
}
$('dataset').addEventListener('change',setDataset);$('report').addEventListener('change',setReport);
$('metric').addEventListener('change',render);$('baseline').addEventListener('change',render);
$('candidate').addEventListener('change',loadComparison);
for (const id of ['outcome','query-sort']) $(id).addEventListener('change',()=>{page=0;renderQueries();});
$('search').addEventListener('input',()=>{page=0;renderQueries();});
$('previous').addEventListener('click',()=>{page--;renderQueries();});$('next').addEventListener('click',()=>{page++;renderQueries();});
for (const button of document.querySelectorAll('[data-tab]')) button.addEventListener('click',()=>{
  tab=button.dataset.tab;++queryGeneration;
  for (const item of document.querySelectorAll('[data-tab]')) {const active=item.dataset.tab===tab;item.setAttribute('aria-pressed',String(active));$(item.dataset.tab).hidden=!active;}
  if (tab==='queries') loadComparison();
});
for (const button of document.querySelectorAll('[data-sort]')) button.addEventListener('click',()=>{
  if (sortKey===button.dataset.sort) descending=!descending; else {sortKey=button.dataset.sort;descending=sortKey!=='name';}
  renderOverview();
});
$('download').addEventListener('click',()=>{
  const metric=currentMetric(), base=baseline();
  const rows=[['Model',metric,'Difference from '+base.name],...activeRuns().map(run=>[run.name,run.aggregate[metric],run.aggregate[metric]-base.aggregate[metric]])];
  const csv=rows.map(row=>row.map(value=>'"'+String(value).replace(/"/g,'""')+'"').join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  const anchor=element('a');anchor.href=url;anchor.download='model-comparison.csv';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
options('dataset',[...new Set(reports.map(r=>r.dataset || 'External judgments'))].map(value=>[value,value]));
setDataset();