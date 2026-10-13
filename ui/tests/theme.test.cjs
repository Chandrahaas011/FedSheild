const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')
// Exercise the exported pure helpers without mounting React or mutating the OS.
function load({saved=null,systemDark=false,blocked=false}={}) {
  const source = fs.readFileSync(path.join(__dirname,'../src/theme.ts'),'utf8')
  const exports = {}
  const meta = {}
  const document = {documentElement:{dataset:{},style:{}},querySelector:()=>({setAttribute:(k,v)=>meta[k]=v})}
  vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText, {
    exports, require:()=>({}), document,
    localStorage:{getItem:()=>{if(blocked)throw Error('denied');return saved}},
    matchMedia:()=>({matches:systemDark}),
  })
  return {api:exports,document,meta}
}
test('saved light overrides a dark system',()=>assert.equal(load({saved:'light',systemDark:true}).api.initialTheme(),'light'))
test('saved dark overrides a light system',()=>assert.equal(load({saved:'dark'}).api.initialTheme(),'dark'))
test('missing or invalid preference defaults to light regardless of system',()=>{
  assert.equal(load({systemDark:true}).api.initialTheme(),'light')
  assert.equal(load({saved:'garbage'}).api.initialTheme(),'light')
})
test('blocked storage defaults to light',()=>assert.equal(load({blocked:true,systemDark:true}).api.initialTheme(),'light'))
test('toggle is reversible',()=>{const {api}=load();assert.equal(api.nextTheme('light'),'dark');assert.equal(api.nextTheme('dark'),'light')})
test('theme applies to root, native controls and browser chrome',()=>{
  const {api,document,meta}=load();api.applyTheme('dark')
  assert.equal(document.documentElement.dataset.theme,'dark')
  assert.equal(document.documentElement.style.colorScheme,'dark')
  assert.equal(meta.content,'#0d1f30')
  api.applyTheme('light');assert.equal(meta.content,'#f3f6f9')
})
