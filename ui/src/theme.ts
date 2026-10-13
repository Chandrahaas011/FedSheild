import { useEffect, useState } from 'react'
export type Theme = 'light' | 'dark'
export const nextTheme = (theme: Theme): Theme => theme === 'light' ? 'dark' : 'light'
const key = 'know-theme'
export function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(key)
    if (saved === 'light' || saved === 'dark') return saved
  } catch { /* Storage may be unavailable in privacy modes. */ }
  return 'light'
}
export function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme
  document.documentElement.style.colorScheme = theme
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#0d1f30' : '#f3f6f9')
}
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(initialTheme)
  useEffect(() => {
    applyTheme(theme)
  }, [theme])
  function toggleTheme() {
    setTheme(current => {
      const next = nextTheme(current)
      try { localStorage.setItem(key, next) } catch { /* Keep working without persistence. */ }
      return next
    })
  }
  return {theme, toggleTheme}
}
