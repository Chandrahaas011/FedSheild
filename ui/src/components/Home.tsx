import { ArrowRight, Flask, Play, ShieldCheck } from '@phosphor-icons/react'

export default function Home({navigate}: {navigate:(page:string)=>void}) {
  return <section className="console-home">
    <div className="home-content">
      <div className="home-kicker"><ShieldCheck size={18}/> KNOW WHEN TO RESUME / RESEARCH CONSOLE</div>
      <h1>Know before you predict.<br/><span>Resume with evidence.</span></h1>
      <p className="home-intro">A lightweight intrusion detector should recognize the limits of its own predictions. Explore the completed research, or walk through a fresh recovery demonstration one decision at a time.</p>
      <div className="home-paths grid grid-cols-1 lg:grid-cols-2 gap-5">
        <button onClick={()=>navigate('monitor')} className="home-path"><Flask size={28} weight="duotone"/><span className="home-path-label">01 / SAVED RESEARCH & TRAFFIC WORKBENCH</span><h2>Explore finished results</h2><p>Inspect saved experiments, publication figures, recovery decisions, and inference with existing model artifacts.</p><span className="home-path-action">Open results console <ArrowRight size={20}/></span></button>
        <button onClick={()=>navigate('lifecycle')} className="home-path"><Play size={28} weight="duotone"/><span className="home-path-label">02 / LIVE, PRESENTER-PACED EXECUTION</span><h2>Run from start to finish</h2><p>Prepare data, train a source model, detect shift, review labels, fit candidates, validate, lock a decision, and reveal the audit.</p><span className="home-path-action">Start guided demonstration <ArrowRight size={20}/></span></button>
      </div>
      <div className="home-boundary"><span/>Local CPU execution. No cloud services. No packet capture or automatic blocking.</div>
    </div>
  </section>
}
