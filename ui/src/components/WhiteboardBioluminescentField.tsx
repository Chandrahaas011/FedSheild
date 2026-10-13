import { lazy, Suspense, useEffect, useState } from 'react'
import type { Theme } from '../theme'

const Fluid = lazy(() => import('./WhiteboardBioluminescent'))
// Adapted from the supplied Next.js field wrapper for this Vite application.
const COLORS = {
  light: ['#c9e2e6', '#82b9c2', '#54aaa5', '#176c74'],
  dark: ['#142c45', '#174b5a', '#237e81', '#7cd5cf'],
}

export default function WhiteboardBioluminescentField({ theme, motion }: { theme: Theme; motion: 'system' | 'on' | 'off' }) {
  const [reduced, setReduced] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches)
  useEffect(() => {
    const media = matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setReduced(media.matches)
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])
  return <div className="bioluminescent-field" aria-hidden="true">
    <div className="bioluminescent-fallback" />
    {(motion === 'on' || (motion === 'system' && !reduced)) && <Suspense fallback={null}><Fluid colors={COLORS[theme]}
      mouseForce={26} cursorSize={140} resolution={0.22} iterationsPoisson={16}
      autoDemo autoSpeed={0.65} autoIntensity={3.2} takeoverDuration={0.25}
      autoResumeDelay={1500} autoRampDuration={0.6}
      className="bioluminescent-canvas" /></Suspense>}
  </div>
}
