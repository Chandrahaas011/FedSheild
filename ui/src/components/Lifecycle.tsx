import { ArrowRight, CheckCircle, DownloadSimple, Play, WarningCircle } from '@phosphor-icons/react'
import { useEffect, useState } from 'react'

type Session = {id:string;source:string;target:string;seed:number;nextStage:number;status:string;events:{stage:number;title:string;durationSeconds:number;result:Record<string,unknown>}[]}
const stages=['Prepare disjoint data','Train source and classify','Detect changed domain','Reveal review labels','Train recovery candidates','Validate independently','Lock resumption decision','Reveal untouched audit']
const explanations=[
  'Allocate separate review, selection validation, shadow validation, monitoring and final-audit roles. Verify disjointness.',
  'Fit a fresh lightweight source model. Fit histogram bins and threshold using source-only roles. Classify an independent source-audit window.',
  'Inspect unseen-testbed features without labels. Domain mismatch can withhold every flow decision without calling every flow an attack.',
  'Reveal dataset ground-truth labels for a limited balanced review pool. This is a labeled-dataset rehearsal, not live analyst annotation.',
  'Actually fit probability recalibration and compact retraining under nested 100, 500 and 2,000-label budgets; retain no adaptation as a baseline.',
  'Apply fixed performance, calibration, Wilson FPR, size and timing checks to independent selection and shadow-validation roles. Audit remains sealed.',
  'Lock a validation-only selection. If no candidate passes, remain withheld. A passing target model may classify within this session only.',
  'Only now evaluate untouched audit labels. Report any unsafe resumption honestly; never change the locked selection to improve its audit score.'
]
async function request(path:string,body:unknown):Promise<Session>{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw new Error(data.error||'Demo failed');return data}
function exportSession(session:Session){const url=URL.createObjectURL(new Blob([JSON.stringify(session,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='guided-demo-session.json';a.click();URL.revokeObjectURL(url)}
export default function Lifecycle(){
  const [session,setSession]=useState<Session|null>(()=>{try{return JSON.parse(sessionStorage.getItem('know-guided-demo')||'null')}catch{return null}})
  const [source,setSource]=useState('nf_unsw_nb15_v2')
  const [busy,setBusy]=useState(false)
  const [error,setError]=useState('')
  const [inspected,setInspected]=useState<number|null>(()=>session?.events.length?session.events.length-1:null)
  useEffect(()=>{if(session)sessionStorage.setItem('know-guided-demo',JSON.stringify(session));else sessionStorage.removeItem('know-guided-demo')},[session])
  async function act(start=false){setBusy(true);setError('');try{const next=await request(start?'/api/lifecycle/start':'/api/lifecycle/next',start?{source}:{id:session!.id,expectedStage:session!.nextStage});setSession(next);setInspected(next.events.length?next.events.length-1:null)}catch(e){setError(e instanceof Error?e.message:'Demo failed')}finally{setBusy(false)}}
  const current=session?.nextStage??0
  const event=inspected===null?undefined:session?.events[inspected]
  const elapsed=session?.events.reduce((n,e)=>n+e.durationSeconds,0)??0
  return <div className="lifecycle-page">
    <div className="scope-note"><WarningCircle size={19}/><span>Fresh in-memory demonstration, separate from saved paper results. Fixed seed 15, clean dataset labels. No network blocking, artifact replacement or production activation. A failure to resume is a valid result.</span></div>
    <div className="lifecycle-layout grid grid-cols-1 lg:grid-cols-[290px_minmax(0,1fr)] gap-5">
      <section className="panel lifecycle-stages"><h2>Execution stages</h2><p className="measurement-note">Advance manually. Explain each stage before continuing.</p><ol>{stages.map((stage,i)=><li key={stage}><button disabled={i>=current} onClick={()=>setInspected(i)} className={inspected===i?'viewing':''}><span className={i<current?'stage-done':i===current?'stage-current':''}>{i<current?<CheckCircle size={20}/>:String(i+1).padStart(2,'0')}</span><span>{stage}</span></button></li>)}</ol><div className="lifecycle-clock">Measured computation: <strong>{elapsed.toFixed(1)} s</strong><p>Presenter pauses are not included.</p></div></section>
      <section className="panel lifecycle-main">
        <div className="panel-heading"><div><h2>{session?'Guided recovery session':'Begin a fresh demonstration'}</h2><p>{session?`Seed ${session.seed} · ${session.status==='complete'?'Complete':`${current}/8 stages completed`}`:'Saved results remain unchanged.'}</p></div><span className="badge neutral">{session?.status==='complete'?'completed':'live execution'}</span></div>
        {!session?<><label className="lifecycle-source">Source training testbed<select value={source} onChange={e=>setSource(e.target.value)} disabled={busy}><option value="nf_unsw_nb15_v2">NF-UNSW-NB15-v2</option><option value="nf_ton_iot_v2">NF-ToN-IoT-v2</option></select></label><div className="lifecycle-intro"><Play size={40} weight="duotone"/><h3>Nothing has been trained yet.</h3><p>This path executes the process rather than replaying a saved recovery result. Each button runs the next real operation.</p><p>Plan 8–12 minutes for an explained presentation. Computation time is measured per stage on your machine.</p></div></>:<>
          {event&&<div className="stage-evidence" aria-live="polite"><div className="section-row"><h3>{event.title}</h3><span className="small-label">{event.durationSeconds.toFixed(2)} s COMPUTE</span></div><p>{explanations[event.stage]}</p><StageResult result={event.result}/><details><summary>Inspect exact stage output</summary><pre>{JSON.stringify(event.result,null,2)}</pre></details></div>}
          {session.status!=='complete'&&<div className="next-stage"><span className="eyebrow">NEXT OPERATION / {current+1} OF 8</span><h3>{stages[current]}</h3><p>{explanations[current]}</p></div>}
          {session.status==='complete'&&<div className="complete-note"><CheckCircle size={24}/><span>Audit revealed after decision lock. Export this demonstration separately from paper evidence.</span></div>}
        </>}
        {error&&<div className="error-box" role="alert"><WarningCircle size={21}/><span>{error} Restart the session if execution failed.</span></div>}
        <div className="workbench-actions">
          {!session?<button className="primary-button" disabled={busy} onClick={()=>act(true)}><Play size={17}/>{busy?'Starting…':'Create demo session'}</button>:session.status!=='complete'&&<button className="primary-button" disabled={busy} onClick={()=>act()}><Play size={17}/>{busy?`Running ${stages[current].toLowerCase()}…`:`Run stage ${current+1}`}<ArrowRight size={17}/></button>}
          {session&&<><button className="secondary-button" disabled={busy} onClick={()=>{setSession(null);setInspected(null);setError('')}}>New session</button><button className="text-button" disabled={busy} onClick={()=>exportSession(session)}><DownloadSimple size={17}/>Export demo</button></>}
        </div>
      </section>
    </div>
  </div>
}
function StageResult({result:r}:{result:Record<string,unknown>}){
  const roleRows=r.roleRows as Record<string,number>|undefined
  const counts=r.counts as Record<string,number>|undefined
  const actions=r.actions as {action:string;label_budget:number;gate_passed:boolean;validation_metrics:{macro_f1:number};shadow_metrics:{macro_f1:number};validationFprUpper:number;shadowFprUpper:number;efficiency:{serialized_model_bytes:number;single_record_latency_ms_p95:number}}[]|undefined
  const assessment=r.assessment as {score:number;threshold:number;shift_detected:boolean}|undefined
  const selected=r.selected as {macro_f1:number;false_positive_rate:number}|null|undefined
  const baseline=r.baseline as {macro_f1:number}|undefined
  return <>
    {typeof r.decision==='string'&&<div className={`decision-banner ${r.decision==='SAFE_TO_RESUME'?'accepted':'withheld'}`}><strong>{r.decision.replaceAll('_',' ')}</strong><span>Decision locked before audit</span></div>}
    {roleRows&&<div className="role-grid grid grid-cols-2 md:grid-cols-3 gap-3">{Object.entries(roleRows).map(([role,n])=><div key={role}><span>{role.replaceAll('_',' ')}</span><strong>{n.toLocaleString()}</strong></div>)}</div>}
    {assessment&&<div className="lifecycle-metrics"><div><span>Mean JSD</span><strong>{assessment.score.toFixed(5)}</strong></div><div><span>Source-only threshold</span><strong>{assessment.threshold.toFixed(5)}</strong></div><div><span>Domain check</span><strong>{assessment.shift_detected?'Withhold':'Accepted'}</strong></div></div>}
    {counts&&<div className="output-counts grid grid-cols-2 md:grid-cols-4 gap-3">{Object.entries(counts).map(([state,n])=><div key={state}><span className={`badge ${state==='DOMAIN_SHIFT_DETECTED'?'warning':'neutral'}`}>{state.replaceAll('_',' ').toLowerCase()}</span><strong>{n.toLocaleString()}</strong><span>flows</span></div>)}</div>}
    {actions&&<div className="table-wrap"><table><thead><tr><th>Action / labels</th><th>Selection F1</th><th>Shadow F1</th><th>FPR upper / selection, shadow</th><th>KiB / p95 ms</th><th>Gate</th></tr></thead><tbody>{actions.map(a=><tr key={a.action+a.label_budget}><td>{a.action.replaceAll('_',' ')} / {a.label_budget}</td><td>{a.validation_metrics.macro_f1.toFixed(4)}</td><td>{a.shadow_metrics.macro_f1.toFixed(4)}</td><td>{a.validationFprUpper.toFixed(3)} / {a.shadowFprUpper.toFixed(3)}</td><td>{(a.efficiency.serialized_model_bytes/1024).toFixed(1)} / {a.efficiency.single_record_latency_ms_p95.toFixed(3)}</td><td><span className={`badge ${a.gate_passed?'good':'warning'}`}>{a.gate_passed?'Pass':'Withhold'}</span></td></tr>)}</tbody></table></div>}
    {baseline&&<div className="lifecycle-metrics"><div><span>Baseline audit macro-F1</span><strong>{baseline.macro_f1.toFixed(4)}</strong></div><div><span>Selected audit macro-F1</span><strong>{selected?selected.macro_f1.toFixed(4):'Not resumed'}</strong></div><div><span>Unsafe resumption</span><strong>{r.unsafeResumption?'Observed':'Not observed'}</strong></div></div>}
    {typeof r.audit==='string'&&<p className="audit-state">{r.audit}</p>}
    {typeof r.note==='string'&&<div className="scope-note">{r.note}</div>}
  </>
}
