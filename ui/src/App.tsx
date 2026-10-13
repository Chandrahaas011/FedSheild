import { useEffect, useRef, useState } from 'react'
import Home from './components/Home'
import Lifecycle from './components/Lifecycle'
import WhiteboardBioluminescentField from './components/WhiteboardBioluminescentField'
import { useTheme } from './theme'
import { Moon, Sun } from '@phosphor-icons/react'
import { ArrowRight, ArrowSquareOut, CheckCircle, DownloadSimple, Fingerprint, Flask, Gauge, Info, Play, ShieldCheck, UploadSimple, WarningCircle, Waveform, X } from '@phosphor-icons/react'

type Recovery = { source: string; target: string; seed: string; shadow_decision: string; shadow_label_budget: string; shadow_unsafe_resumption: string; selected_audit_macro_f1: string }
type Evidence = { title: string; experimentCount: number; figureCount: number; tableCount: number; figures: string[]; recovery: Recovery[]; recoverySummary: {runs: number; resumed: number; unsafe: number}; experiments: { id: string; model: string; seed: number; protocol: string; status: string }[] }
type Schema = {features: string[]; windowSize: number; threshold: number; experimentId: string; artifactSha256: string; modelBytes: number; protocol: string}
type WindowResult = {index: number; shift_detected: boolean; score: number; threshold: number; contributors: {feature: string; score: number}[]}
type Analysis = {origin: string; inputRows: number; scoredRows: number; pendingRows: number; counts: Record<string, number>; windows: WindowResult[]; rows: {row: number; decision: string}[]; elapsedMs: number; timingScope: string; note: string; experimentId: string; artifactSha256: string}
const names = { source: 'Source audit', target: 'Unseen testbed', mixture: '50% domain mixture' }
const sourceName = (s: string) => s.includes('unsw') ? 'NF-UNSW-NB15-v2' : 'NF-ToN-IoT-v2'
const stateLabel = (s: string) => ({BENIGN: 'Benign', MALICIOUS: 'Malicious', UNKNOWN: 'Unknown', DOMAIN_SHIFT_DETECTED: 'Domain shift'}[s] || s)
async function api<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : undefined)
  const data = await response.json()
  if (!response.ok) throw new Error(data.error || 'The local API could not complete this request.')
  return data
}
function download(name: string, content: string, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([content], {type}))
  const link = document.createElement('a'); link.href = url; link.download = name; link.click(); URL.revokeObjectURL(url)
}
function Badge({state}: {state: string}) {return <span className={`badge ${state === 'DOMAIN_SHIFT_DETECTED' || state === 'RECALIBRATION_REQUIRED' ? 'warning' : state === 'MALICIOUS' ? 'danger' : ['BENIGN','IN_DOMAIN','SAFE_TO_RESUME'].includes(state) ? 'good' : 'neutral'}`}>{stateLabel(state).replaceAll('_', ' ').toLowerCase()}</span>}

export default function App() {
  const {theme, toggleTheme} = useTheme()
  const [motion,setMotion] = useState<'system'|'on'|'off'>(()=>{
    try { const saved=localStorage.getItem('know-motion'); if(saved==='on'||saved==='off')return saved } catch { /* Use the accessible default. */ }
    return 'system'
  })
  useEffect(()=>{try {localStorage.setItem('know-motion',motion)}catch { /* Storage is optional. */ }},[motion])
  const [page, setPage] = useState('home')
  const [evidence, setEvidence] = useState<Evidence | null>(null)
  const [schema, setSchema] = useState<Schema | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [scenario, setScenario] = useState<keyof typeof names>('source')
  const [source, setSource] = useState('nf_unsw_nb15_v2')
  const [offset, setOffset] = useState(0)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [search, setSearch] = useState('')
  const [figure, setFigure] = useState<string | null>(null)
  const file = useRef<HTMLInputElement>(null)
  useEffect(() => {Promise.all([api<Evidence>('/api/evidence'), api<Schema>('/api/schema')]).then(([e, s]) => {setEvidence(e); setSchema(s)}).catch(e => setError(e.message))}, [])
  useEffect(() => {if (!figure) return; const handler = (e: KeyboardEvent) => {if (e.key === 'Escape') setFigure(null)}; window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler)}, [figure])
  async function run(csv?: string) {
    setBusy(true); setError('')
    try {setAnalysis(await api<Analysis>('/api/analyze', csv === undefined ? {source, scenario, offset} : {source, csv}))}
    catch (e) {setError(e instanceof Error ? e.message : 'Analysis failed')}
    finally {setBusy(false)}
  }
  const shifted = analysis?.windows.some(w => w.shift_detected)
  const navigation = [{id:'home',label:'Home',icon:ShieldCheck},{id:'lifecycle',label:'Start-to-finish demo',icon:Play},{id:'monitor', label:'Results workbench', icon:Waveform}, {id:'recovery', label:'Recovery evidence', icon:ShieldCheck}, {id:'research', label:'Experiment library', icon:Flask}, {id:'guide', label:'Demonstration guide', icon:Info}]
  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => {e.preventDefault();setPage('home')}}><span className="brand-symbol"><ShieldCheck size={27} weight="duotone"/></span><span>KNOW<span className="brand-sub">WHEN TO RESUME</span></span></a>
      <div className="sidebar-label">RESEARCH CONSOLE</div>
      <nav aria-label="Main navigation">{navigation.map(({id,label,icon:Icon}) => <button key={id} className={`nav-item ${page===id?'active':''}`} aria-current={page===id?'page':undefined} onClick={() => setPage(id)}><Icon size={21}/>{label}{page===id && <span className="nav-dot"/>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="local-mark"><span/> LOCAL WORKSPACE</div><p>Saved models. Real traffic features.<br/>No cloud. No packet interception.</p><span className="version">Research demonstration / v1.0</span></div>
    </aside>
    <main className="main-content">
      <WhiteboardBioluminescentField theme={theme} motion={motion}/>
      <header className="topbar"><div className="breadcrumb">IoT intrusion detection <span>/</span> <strong>{navigation.find(n=>n.id===page)?.label}</strong></div><div className="topbar-actions"><span className={`connection ${evidence ? '' : 'offline'}`}><span/>{evidence?'Local API connected':'Connecting to local API'}</span><label className="motion-control"><Waveform size={18}/><span className="sr-only">Background motion</span><select aria-label="Background motion" value={motion} onChange={e=>setMotion(e.target.value as 'system'|'on'|'off')}><option value="system">Motion: system</option><option value="on">Motion: on</option><option value="off">Motion: off</option></select></label><button className="theme-toggle" onClick={toggleTheme} aria-label={`Switch to ${theme==='light'?'dark':'light'} mode`} title={`Switch to ${theme==='light'?'dark':'light'} mode`}>{theme==='light'?<Moon size={18}/>:<Sun size={18}/>}<span>{theme==='light'?'Dark':'Light'} mode</span></button></div></header>
      <div className={`page-content max-w-[1440px] mx-auto ${page==='home'?'home-page-content':''}`}>
        {page==='home'?<Home navigate={setPage}/>:<>
        <div className="page-heading"><div><div className="eyebrow">VALIDATION-GATED · RESOURCE-BOUNDED</div><h1>{page==='monitor'?'A prediction is a decision.':page==='lifecycle'?'From traffic to a locked decision.':page==='recovery'?'Resume only with evidence.':page==='research'?'Every claim has a record.':'Show the decision, not just the score.'}</h1><p>{page==='monitor'?'Inspect traffic, measure domain mismatch, and see when the system withholds a prediction.':page==='lifecycle'?'Execute each stage live, pause to explain, then reveal audit outcomes only after validation-based selection.':page==='recovery'?'Selection and shadow validation decide. Untouched audit measures the consequences.':page==='research'?'Browse saved experiment records and publication figures without altering the research ledger.':'A short, honest walkthrough for your college presentation.'}</p></div><span className="study-tag"><Flask size={17}/>RESEARCH / NOT A FIREWALL</span></div>
        {error && <div className="error-box" role="alert"><WarningCircle size={22}/><span>{error}</span><button aria-label="Dismiss error" onClick={()=>setError('')}><X size={18}/></button></div>}
        <section className="stat-grid grid grid-cols-2 xl:grid-cols-4 gap-4">
          <Stat label="EXPERIMENT RECORDS" value={evidence?.experimentCount.toString() ?? '—'} detail="Read from the master results ledger" icon={<Flask size={21}/>}/>
          <Stat label="INPUT FEATURE SPACE" value={schema?.features.length.toString() ?? '—'} suffix="features" detail="Identifier-excluded NetFlow-v2" icon={<Fingerprint size={21}/>}/>
          <Stat label="GUARD WINDOW" value={schema?.windowSize.toString() ?? '—'} suffix="flows" detail="Frozen artifact-compatible window" icon={<Waveform size={21}/>}/>
          <Stat label="SHADOW-GATE RESUMPTIONS" value={evidence?`${evidence.recoverySummary.resumed}/${evidence.recoverySummary.runs}`:'—'} detail={`${evidence?.recoverySummary.unsafe ?? '—'} observed unsafe · saved clean-label cohort`} icon={<ShieldCheck size={21}/>}/>
        </section>
        {page==='lifecycle'&&<Lifecycle/>}
        {page==='monitor' && <>
          <section className="decision-rail" aria-label="System decision sequence">{['Inspect domain','Classify or withhold','Review shifted traffic','Validate before resuming'].map((step,i)=><div key={step}><span className={i===0?'rail-number current':'rail-number'}>{String(i+1).padStart(2,'0')}</span><span>{step}</span>{i<3&&<ArrowRight size={17}/>}</div>)}</section>
          <div className="analysis-layout grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_340px] gap-5">
            <section className="panel">
              <div className="panel-heading"><div><h2>Traffic workbench</h2><p>Run inference with a saved seed-0 guard.</p></div><span className="small-label">OFFLINE REPLAY</span></div>
              <div className="workbench-controls grid grid-cols-1 md:grid-cols-2 gap-4"><label>Training testbed<select value={source} disabled={busy} onChange={e=>{setSource(e.target.value);setAnalysis(null);setOffset(0)}}><option value="nf_unsw_nb15_v2">NF-UNSW-NB15-v2</option><option value="nf_ton_iot_v2">NF-ToN-IoT-v2</option></select></label><label>Audit / replay window<select value={offset} disabled={busy} onChange={e=>{setOffset(Number(e.target.value));setAnalysis(null)}}>{Array.from({length:20},(_,i)=><option key={i} value={i}>Window {i+1} of 20</option>)}</select></label></div>
              <fieldset className="scenario-picker"><legend>Traffic scenario</legend>{Object.entries(names).map(([id,label])=><button key={id} aria-pressed={scenario===id} className={scenario===id?'selected':''} disabled={busy} onClick={()=>{setScenario(id as keyof typeof names);setAnalysis(null)}}>{label}<span>{id==='source'?'Disjoint source validation':id==='target'?'Label-blind target window':'Source + target features'}</span></button>)}</fieldset>
              <div className="workbench-actions"><button className="primary-button" disabled={busy || !schema} onClick={()=>run()}><Play size={17} weight="fill"/>{busy?'Analyzing traffic…':'Analyze window'}{!busy&&<ArrowRight size={17}/>}</button><button className="secondary-button" disabled={busy || !schema} onClick={()=>file.current?.click()}><UploadSimple size={18}/>Upload feature CSV</button><input ref={file} type="file" accept=".csv,text/csv" className="hidden" onChange={async e=>{const f=e.target.files?.[0]; if(f){if(f.size>3*1024*1024)setError('CSV must be at most 3 MiB.');else await run(await f.text())}e.target.value=''}}/></div>
              <div className="sample-download"><button className="text-button" disabled={busy || !schema} onClick={async()=>{try{const response=await fetch(`/api/sample/${source}.csv`);if(!response.ok)throw new Error('Sample could not be loaded. Check local data/artifacts.');download('source-audit-features.csv',await response.text(),'text/csv')}catch(e){setError(e instanceof Error?e.message:'Sample failed')}}}><DownloadSimple size={15}/>Download valid sample CSV</button><span>512 raw-feature rows · no labels</span></div>
              <div className="scope-note"><Info size={17}/><span>Historical class-capped protocol. Replay order is not natural time. No threshold is fitted in this console.</span></div>
              {analysis ? <div aria-live="polite" className="result-area">
                <div className={`decision-banner ${shifted?'withheld':'accepted'}`}><span className="decision-icon">{shifted?<WarningCircle size={27}/>:<CheckCircle size={27}/>}</span><div><span className="eyebrow">{shifted?'DEPLOYMENT INTERLOCK':'DOMAIN CHECK PASSED'}</span><h3>{shifted?'Automatic predictions withheld':'Classification permitted'}</h3><p>{shifted?'Domain mismatch is not evidence of an attack. Review this traffic before recovery.':'Per-flow conformal rejection still applies; unknown is not benign.'}</p></div></div>
                <div className="output-counts grid grid-cols-2 md:grid-cols-4 gap-3">{Object.entries(analysis.counts).map(([state,count])=><div key={state}><Badge state={state}/><strong>{count.toLocaleString()}</strong><span>flows</span></div>)}</div>
                <div className="section-row"><h3>Window diagnostics</h3><button className="text-button" onClick={()=>download('demo-session.json',JSON.stringify(analysis,null,2))}><DownloadSimple size={16}/>Export session</button></div>
                <div className="table-wrap"><table><thead><tr><th>Window</th><th>Mean JSD</th><th>Frozen threshold</th><th>Decision</th></tr></thead><tbody>{analysis.windows.map(w=><tr key={w.index}><td>{String(w.index).padStart(2,'0')}</td><td className="mono">{w.score.toFixed(5)}</td><td className="mono">{w.threshold.toFixed(5)}</td><td><Badge state={w.shift_detected?'DOMAIN_SHIFT_DETECTED':'IN_DOMAIN'}/></td></tr>)}</tbody></table></div>
                <p className="measurement-note">{analysis.scoredRows.toLocaleString()} scored · {analysis.pendingRows} unscored remainder · {analysis.elapsedMs.toFixed(1)} ms server analysis (not a benchmark). Incomplete rows are not retained between requests.</p>
                <details><summary>Inspect first {analysis.rows.length} flow decisions and provenance</summary><div className="flow-grid">{analysis.rows.map(r=><span key={r.row} title={`Flow ${r.row}: ${r.decision}`} className={`flow-dot ${r.decision.toLowerCase()}`}/>)}</div><p className="mono provenance">{analysis.experimentId}<br/>SHA-256: {analysis.artifactSha256}</p><p className="measurement-note">{analysis.note}</p></details>
              </div>:<div className="empty-workbench"><Waveform size={47} weight="duotone"/><h3>Ready to inspect a traffic window</h3><p>Start with a source audit window, then replay an unseen testbed. Compare the domain score—not just the predicted label.</p><div className="empty-rule"><span/> Source-fitted bins → frozen threshold → decision</div></div>}
            </section>
            <aside className="right-column"><section className="panel signal-panel"><div className="panel-heading"><h2>Domain signal</h2><Gauge size={21}/></div>{analysis?<><div className="signal-value">{analysis.windows[0].score.toFixed(4)}<span>mean JSD / first window</span></div><div className="score-track"><div style={{width:`${Math.min(100,analysis.windows[0].score*100)}%`}}/><span style={{left:`${analysis.windows[0].threshold*100}%`}}/></div><div className="track-labels"><span>0 · matching marginals</span><span>1 · disjoint</span></div><p className="measurement-note">Threshold {analysis.windows[0].threshold.toFixed(5)}. Alarm if score exceeds this value.</p><h3 className="contributor-title">Largest marginal divergences</h3>{analysis.windows[0].contributors.map(c=><div className="contributor" key={c.feature}><div><span title={c.feature}>{c.feature}</span><strong>{c.score.toFixed(3)}</strong></div><div className="mini-track"><span style={{width:`${c.score*100}%`}}/></div></div>)}<p className="measurement-note">Distribution differences, not causal explanations of maliciousness.</p></>:<div className="signal-placeholder"><span className="signal-value">—</span><p>Analyze traffic to compare observed marginals with the source reference.</p></div>}</section><section className="panel glossary"><h2>Read the outputs</h2>{[['BENIGN','A singleton benign prediction in an accepted domain.'],['MALICIOUS','A singleton malicious prediction—not automatic prevention.'],['UNKNOWN','The rejector cannot support a single class.'],['DOMAIN_SHIFT_DETECTED','The batch is outside the accepted source distribution.']].map(([s,d])=><div key={s}><Badge state={s}/><p>{d}</p></div>)}</section></aside>
          </div>
        </>}
        {page==='recovery'&&<section className="panel"><div className="panel-heading"><div><h2>Prospective shadow-validation cohort</h2><p>20 saved runs · seeds 15–24 · clean labels · both transfer directions</p></div><Badge state="READ_ONLY"/></div><div className="recovery-callout"><ShieldCheck size={30}/><div><h3>{evidence?.recoverySummary.resumed ?? '—'} models passed the two-stage gate</h3><p>{evidence?.recoverySummary.unsafe ?? '—'} unsafe resumptions observed in this cohort. This is not a deployment guarantee; separate directional label-noise experiments contain failures.</p></div></div><div className="scope-note"><Info size={18}/><span>These are saved experimental decisions. This page does not retrain, approve new traffic, or replace the source artifact.</span></div><div className="table-wrap"><table><thead><tr><th>Transfer direction</th><th>Seed</th><th>Gate decision</th><th>Labels</th><th>Audit macro-F1</th><th>Unsafe</th></tr></thead><tbody>{evidence?.recovery.map(r=><tr key={r.source+r.seed}><td>{sourceName(r.source)}<br/><span className="muted">→ {sourceName(r.target)}</span></td><td>{r.seed}</td><td><Badge state={r.shadow_decision}/></td><td>{Number(r.shadow_label_budget).toLocaleString()}</td><td className="mono">{r.selected_audit_macro_f1?Number(r.selected_audit_macro_f1).toFixed(4):'Not selected'}</td><td>{r.shadow_unsafe_resumption==='1'?'Yes':'No'}</td></tr>)}</tbody></table></div></section>}
        {page==='research'&&<><section className="panel"><div className="panel-heading"><div><h2>Publication figure library</h2><p>Original saved outputs, not hand-entered values.</p></div><span className="small-label">{evidence?.figureCount ?? '—'} FIGURES</span></div><div className="figure-grid grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">{evidence?.figures.map(f=><button key={f} className="figure-card" onClick={()=>setFigure(f)}><img src={`/api/figures/${f}`} loading="lazy" alt={f.replace('.png','').replaceAll('_',' ')}/><span>{f.replace(/^research_figure\d+_/, '').replace('.png','').replaceAll('_',' ')}<ArrowSquareOut size={17}/></span></button>)}</div></section><section className="panel mt-5"><div className="panel-heading"><h2>Experiment index</h2><label className="search-label"><span className="sr-only">Search experiment records</span><input placeholder="Filter by experiment, model or protocol" value={search} onChange={e=>setSearch(e.target.value)}/></label></div><div className="table-wrap experiment-table"><table><thead><tr><th>Experiment ID</th><th>Model</th><th>Seed</th><th>Status</th></tr></thead><tbody>{evidence?.experiments.filter(e=>`${e.id} ${e.model} ${e.protocol}`.toLowerCase().includes(search.toLowerCase())).map(e=><tr key={e.id}><td className="mono">{e.id}</td><td>{e.model}</td><td>{e.seed ?? '—'}</td><td>{e.status}</td></tr>)}</tbody></table></div></section></>}
        {page==='guide'&&<div className="guide-layout grid grid-cols-1 lg:grid-cols-[1.5fr_1fr] gap-5"><section className="panel guide"><h2>A five-minute demonstration</h2>{[['Frame the problem','High benchmark performance does not establish reliability on a different traffic testbed. Our system makes unsupported predictions visible.'],['Show the source domain','Open Results workbench. Select Source audit and analyze Window 1. Show the score, threshold and per-flow outputs. If a source window alarms, explain that source false alarms are possible.'],['Change the traffic domain','Choose Unseen testbed and analyze the same window. Explain why DOMAIN_SHIFT_DETECTED withholds all flow decisions without declaring every flow malicious.'],['Explain the path to recovery','Open Recovery evidence. Distinguish review labels, selection validation, independent shadow validation and final untouched audit. A failed gate means keep withholding.'],['Show honest limitations','Open the figure library. Discuss negative transfer and label-noise failures. This demo does not capture live packets, prevent attacks, or prove deployment safety.']].map(([title,text],i)=><div className="guide-step" key={title}><span>{String(i+1).padStart(2,'0')}</span><div><h3>{title}</h3><p>{text}</p></div></div>)}</section><section className="panel guide"><h2>What the title means</h2><h3>Know When to Resume</h3><p>Resume automatic classification after domain mismatch only when independent validation supports doing so.</p><h3>Validation-Gated</h3><p>Model selection and shadow validation govern the decision. Final audit reports the outcome and must not choose the model.</p><h3>Resource-Bounded</h3><p>Recovery investigates limited review-label budgets and compact CPU models. Timing is specific to the measured hardware.</p><h3>Lightweight IoT Intrusion Detection</h3><p>Classify extracted traffic-flow features and expose uncertainty. Detection is not prevention.</p><div className="scope-note"><Info size={18}/><span>Full setup and presentation instructions: docs/UI_DEMONSTRATION_GUIDE.md</span></div></section></div>}
        <footer className="footer"><span>Know When to Resume <span>/</span> Reproducible IoT IDS research</span><span>Evidence first. No automatic blocking.</span></footer>
        </>}
      </div>
    </main>
    {figure&&<FigureView figure={figure} close={()=>setFigure(null)}/>}
  </div>
}
function Stat({label,value,suffix,detail,icon}: {label:string;value:string;suffix?:string;detail:string;icon:React.ReactNode}) {return <div className="stat-card"><div>{label}{icon}</div><strong>{value}<span>{suffix}</span></strong><p>{detail}</p></div>}
function FigureView({figure,close}: {figure:string;close:()=>void}) {
  const ref=useRef<HTMLDialogElement>(null)
  useEffect(()=>{ref.current?.showModal()},[])
  return <dialog ref={ref} aria-label="Publication figure" className="figure-dialog" onCancel={close} onClick={e=>{if(e.target===ref.current){const b=ref.current!.getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)close()}}}><div><h2>{figure.replace('.png','').replaceAll('_',' ')}</h2><button autoFocus onClick={close} aria-label="Close figure"><X size={23}/></button></div><img src={`/api/figures/${figure}`} alt={figure.replaceAll('_',' ')}/><a href={`/api/figures/${figure}`} target="_blank" rel="noreferrer">Open original figure <ArrowSquareOut size={16}/></a></dialog>
}
