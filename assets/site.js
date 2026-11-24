(()=>{'use strict';
const DATA=window.ROBOKATA_DATA;
const asset=s=>'assets/'+s;let heroPair,heroIndex=0;
class Pair{
 constructor(el,item){this.el=el;this.state='idle';this.running=false;this.ready=false;this.item=item;this.el.dataset.phase='idle';this.el.innerHTML=`<div class="pair-head"><span>Demo</span><span>Execution</span></div><div class="pair-panes"><video muted playsinline preload="auto" poster="${asset(item.dp)}" aria-label="Demonstration"></video><video muted playsinline preload="auto" poster="${asset(item.qp)}" aria-label="Robot execution"></video></div><div class="pair-bar"><button class="toggle" aria-label="Pause videos"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z" fill="currentColor"/></svg></button><input aria-label="Video progress" type="range" min="0" max="1000" value="0"><span class="time">0:00</span><button class="pair-fullscreen" aria-label="Fullscreen">⛶</button></div>`;
 [this.d,this.q]=el.querySelectorAll('video');this.button=el.querySelector('.toggle');this.seek=el.querySelector('input');this.clock=el.querySelector('.time');this.button.onclick=()=>this.running?this.pause():this.play();this.d.ontimeupdate=()=>this.update();this.q.ontimeupdate=()=>this.update();this.seek.oninput=()=>this.scrub();el.querySelector('.pair-fullscreen').onclick=()=>{if(el.requestFullscreen)el.requestFullscreen().catch(()=>{})};for(const v of [this.d,this.q])v.onerror=()=>{this.running=false;this.setControl('Retry');this.button.title='Video could not load'}
  this.state = 'demo';
  this.d.onended = () => {
   if (this.state !== 'demo' || !this.d.ended) return;
   this.state = 'execution';
   if (this.running) this.play();
  };
  this.q.onended = () => {
   if (this.state !== 'execution' || !this.q.ended) return;
   if (this !== heroPair && this.running) {
    this.state = 'done';
    this.play();
    return;
   }
   this.pause();
   this.state = 'done';
   this.setControl('Replay');
   const finished = this;
   setTimeout(() => {
    if (heroPair === finished && this.state === 'done') hero(heroIndex + 1);
   }, 1200);
  };
 }
 setControl(mode){this.button.innerHTML=mode==='Pause'?'<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M7 5h4v14H7zm6 0h4v14h-4z"/></svg>':'<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M8 5v14l11-7z"/></svg>';this.button.setAttribute('aria-label',mode);this.button.title=mode}
 load(){if(this.ready)return;this.d.src=asset(this.item.demo);this.q.src=asset(this.item.query);this.d.load();this.q.load();this.ready=true}
 pause(){this.running=false;this.d.pause();this.q.pause();this.setControl('Play')}
 play() {
  this.load();
  if (this.state === 'done') {
   this.d.currentTime = this.q.currentTime = 0;
   this.state = 'demo';
  }
  this.running = true;
  this.el.dataset.phase = this.state;
  this.setControl('Pause');
  const video = this.state === 'demo' ? this.d : this.q;
  video.play().catch(error => {
   if (error.name !== 'AbortError' && this.running) this.pause();
  });
 }
 update() {
  const total = this.d.duration + this.q.duration;
  if (!Number.isFinite(total) || !total) return;
  const time = this.state === 'demo' ? this.d.currentTime : this.d.duration + this.q.currentTime;
  this.el.dispatchEvent(new CustomEvent('pairtime', {detail: {
   demoProgress: this.d.currentTime / this.d.duration,
   executionProgress: this.q.currentTime / this.q.duration
  }}));
  this.seek.value = 1000 * time / total;
  this.clock.textContent = Math.floor(time / 60) + ':' + String(Math.floor(time % 60)).padStart(2, '0');
 }
 scrub() {
  const total = this.d.duration + this.q.duration;
  if (!Number.isFinite(total)) return;
  const time = total * Number(this.seek.value) / 1000;
  this.d.pause();
  this.q.pause();
  this.state = time < this.d.duration ? 'demo' : 'execution';
  this.d.currentTime = Math.min(time, this.d.duration);
  this.q.currentTime = Math.max(0, time - this.d.duration);
  this.el.dataset.phase = this.state;
  if (this.running) this.play();
  else this.setControl('Play');
  this.update();
 }
}
function mountAll(root=document){root.querySelectorAll('[data-pair]:not([data-mounted])').forEach(el=>{el.dataset.mounted='1';new Pair(el,DATA.items[el.dataset.pair]).play()})}
function hero(i){if(heroPair)heroPair.pause();heroIndex=(i+DATA.hero.length)%DATA.hero.length;let id=DATA.hero[heroIndex],it=DATA.items[id];document.getElementById('hero-pair').classList.toggle('landscape',it.section==='robotwin');heroPair=new Pair(document.getElementById('hero-pair'),it);heroPair.play();document.getElementById('hero-task').textContent=it.title;document.querySelectorAll('.dots button').forEach((b,n)=>b.setAttribute('aria-current',String(n===heroIndex)));updateHeroPreviews()}
function updateHeroPreviews(){
 const current=DATA.items[DATA.hero[heroIndex]];
 document.querySelector('.carousel-stage').dataset.format=current.section==='robotwin'?'landscape':'square';
 for(const [selector,delta] of [['.peek-prev',-1],['.peek-next',1]]){
  const it=DATA.items[DATA.hero[(heroIndex+delta+DATA.hero.length)%DATA.hero.length]];
  document.querySelectorAll(selector+' img').forEach((im,j)=>{im.src=asset(j?it.qp:it.dp)});
 }
 const slide=document.querySelector('.hero-slide');slide.classList.remove('enter');void slide.offsetWidth;slide.classList.add('enter');
}
document.querySelector('.peek-prev').onclick=()=>hero(heroIndex-1);
document.querySelector('.peek-next').onclick=()=>hero(heroIndex+1);
let touchStart=null;
document.querySelector('.carousel-stage').addEventListener('touchstart',e=>{if(e.target.closest('input,button'))return;touchStart=[e.touches[0].clientX,e.touches[0].clientY]},{passive:true});
document.querySelector('.carousel-stage').addEventListener('touchend',e=>{if(!touchStart)return;const dx=e.changedTouches[0].clientX-touchStart[0],dy=e.changedTouches[0].clientY-touchStart[1];touchStart=null;if(Math.abs(dx)>65&&Math.abs(dx)>Math.abs(dy)*1.5)hero(heroIndex+(dx<0?1:-1))},{passive:true});
document.querySelector('.carousel-navigation').addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();hero(heroIndex+(e.key==='ArrowRight'?1:-1))}});
document.querySelectorAll('[data-hero-index]').forEach(b=>b.onclick=()=>hero(Number(b.dataset.heroIndex)));
document.getElementById('prev-slide').onclick=()=>hero(heroIndex-1);document.getElementById('next-slide').onclick=()=>hero(heroIndex+1);document.querySelectorAll('.dots button').forEach((b,i)=>b.onclick=()=>hero(i));hero(0);mountAll();
const tabs=[...document.querySelectorAll('[data-axis]')];function axis(button){tabs.forEach(b=>{const on=b===button;b.setAttribute('aria-selected',on);b.tabIndex=on?0:-1;document.getElementById('axis-'+b.dataset.axis).hidden=!on});document.getElementById('axis-description').textContent=DATA.axes[button.dataset.axis]}
tabs.forEach((b,i)=>{b.onclick=()=>axis(b);b.onkeydown=e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();const next=tabs[(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length];axis(next);next.focus()}}});
document.querySelectorAll('[data-trajectory]').forEach(b=>b.onclick=()=>{const el=document.getElementById(b.dataset.trajectory);el.hidden=!el.hidden;b.setAttribute('aria-expanded',String(!el.hidden))});
// Cycle through the method in presentation order while it is visible.
const spot=document.querySelector('.method-spot');
const methodSteps=[...document.querySelectorAll('.method-steps [data-region]')];
let methodIndex=0,methodTimer=null,methodVisible=false;
function showMethodStep(index){
 methodIndex=index;
 methodSteps.forEach((button,i)=>{
  button.classList.toggle('active',i===index);
  button.setAttribute('aria-pressed',String(i===index));
 });
 const [x,y,w,h]=methodSteps[index].dataset.region.split(',');
 Object.assign(spot.style,{left:x+'%',top:y+'%',width:w+'%',height:h+'%',opacity:1});
}
function startMethodDemo(){
 clearInterval(methodTimer);
 methodTimer=null;
 if(methodVisible&&!document.hidden){
  methodTimer=setInterval(()=>showMethodStep((methodIndex+1)%methodSteps.length),3000);
 }
}
methodSteps.forEach((button,index)=>button.onclick=()=>{showMethodStep(index);startMethodDemo()});
showMethodStep(0);
new IntersectionObserver(entries=>{
 methodVisible=entries.some(entry=>entry.isIntersecting);
 if(methodVisible)showMethodStep(0);
 startMethodDemo();
},{threshold:0.2}).observe(document.querySelector('.method-layout'));
document.addEventListener('visibilitychange',startMethodDemo);

const comparisonDemo=document.getElementById('comparison-demo');
const comparisonVideos=[...document.querySelectorAll('.compare-grid video:not(#comparison-demo)')];
const comparisonReplay=document.getElementById('compare-play');
let comparisonGeneration=0;
function compare(index){
 comparisonGeneration++;
 comparisonDemo.pause();comparisonVideos.forEach(v=>v.pause());
 const c=DATA.comparisons[index];
 document.querySelectorAll('[data-compare-case]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.compareCase)===index)));
 comparisonDemo.src=asset(c.demo);comparisonDemo.poster=asset(c.demoPoster);comparisonDemo.load();
 comparisonVideos.forEach((v,i)=>{v.preload='metadata';v.src=asset(c.videos[i]);v.poster=asset(c.posters[i]);v.load()});
 document.querySelector('.compare-note').textContent=c.note;
 comparisonReplay.title='Replay the demo and all four policies together';
 for(const v of [comparisonDemo,...comparisonVideos]){v.muted=true;v.autoplay=true;v.loop=true;v.play().catch(()=>{comparisonReplay.title='Click to play all videos'})}
}
document.querySelectorAll('[data-compare-case]').forEach(b=>b.onclick=()=>compare(Number(b.dataset.compareCase)));
comparisonReplay.onclick=()=>{
 const generation=++comparisonGeneration;
 const videos=[comparisonDemo,...comparisonVideos];
 videos.forEach(v=>{v.pause();v.currentTime=0;v.preload='auto'});
 videos.forEach(v=>v.play().catch(()=>{if(generation===comparisonGeneration)comparisonReplay.title='Use the video play control to resume playback'}));
};
compare(0);

// One recorded trajectory, projected with equal scale. Drag changes the viewing angle.
const rt=DATA.route,canvas=document.getElementById('route-canvas'),ctx=canvas.getContext('2d');let az=-.65,el=.4,drag=null,portionD=1,portionQ=1,playingRoute=false;
const all=[...rt.demo,...rt.execution,...rt.waypoints],lo=[0,1,2].map(k=>Math.min(...all.map(p=>p[k]))-.025),hi=[0,1,2].map(k=>Math.max(...all.map(p=>p[k]))+.025),center=lo.map((v,k)=>(v+hi[k])/2),extent=Math.hypot(...hi.map((v,k)=>v-lo[k]));let cw=500,ch=340;
function pr(p){let[x,y,z]=p.map((v,k)=>v-center[k]),a=Math.cos(az)*x-Math.sin(az)*y,b=Math.sin(az)*x+Math.cos(az)*y,s=Math.min(cw,ch)*.88/extent;return[cw/2+a*s,ch/2+(b*Math.sin(el)-z*Math.cos(el))*s]}
function line(points,color,w=2){if(points.length<2)return;ctx.beginPath();points.forEach((p,i)=>{let[x,y]=pr(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=color;ctx.lineWidth=w;ctx.stroke()}
function routeDraw(){let r=canvas.getBoundingClientRect();cw=r.width||500;ch=r.height||340;let dpi=devicePixelRatio||1;canvas.width=cw*dpi;canvas.height=ch*dpi;ctx.setTransform(dpi,0,0,dpi,0,0);ctx.clearRect(0,0,cw,ch);for(let k=0;k<6;k++){let x=lo[0]+(hi[0]-lo[0])*k/5,y=lo[1]+(hi[1]-lo[1])*k/5;line([[x,lo[1],lo[2]],[x,hi[1],lo[2]]],'#dce4e8',1);line([[lo[0],y,lo[2]],[hi[0],y,lo[2]]],'#dce4e8',1)}line(rt.demo.slice(0,Math.max(1,Math.ceil(rt.demo.length*portionD))),'#8359ac',2);line(rt.execution.slice(0,Math.max(1,Math.ceil(rt.execution.length*portionQ))),'#078f88',3);ctx.font='14px Arial';rt.waypoints.forEach((p,i)=>{let[x,y]=pr(p);ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.strokeStyle='#c39448';ctx.lineWidth=1.5;ctx.stroke();ctx.fillStyle='#94661f';ctx.fillText('W'+(i+1),x+12,y+[-6,-18,19][i])})}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;az+=(e.clientX-drag[0])*.009;el=Math.max(-1.2,Math.min(1.2,el+(e.clientY-drag[1])*.009));drag=[e.clientX,e.clientY];routeDraw()};canvas.onpointerup=canvas.onpointercancel=()=>drag=null;document.querySelector('.route-reset').onclick=()=>{az=-.65;el=.4;routeDraw()};new ResizeObserver(routeDraw).observe(canvas);
document.querySelector('[data-pair="bench-route-3d"]').addEventListener('pairtime',e=>{let d=e.detail;playingRoute=true;portionD=d.demoProgress;portionQ=d.executionProgress;routeDraw()});routeDraw();
})();
