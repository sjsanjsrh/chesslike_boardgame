(function(){
  try{
    function boot(){
      try{ window.__overlayBootCounter = (window.__overlayBootCounter||0) + 1; }catch(e){}
      try{ window.__overlayBootTs = Date.now(); }catch(e){}

  // Ensure root container
      var rootId='injected_overlay-root';
      var root=document.getElementById(rootId);
      if(!root){
        root=document.createElement('div');
        root.id=rootId;
        document.body.appendChild(root);
      }

  // Log overlay installer + setter
      if(!window.injected_overlayOverlaySet){
        var id='injected_overlay-log';
        var el=document.getElementById(id);
        if(!el){
          el=document.createElement('div');
          el.id=id; el.setAttribute('data-injected_overlay','1');
          el.innerHTML='<div>log ready</div>';
          root.appendChild(el);
        }
        var buf={html:'',append:false};
        window.injected_overlayOverlaySet=function(html,a){
          try{ html=String(html||''); if(!html){return true;} buf.html=html; buf.append=(a===true); return true; }catch(e){ return false; }
        };
        window.setInterval(function(){
          try{
            var el=document.getElementById(id); if(!el) return;
            if(buf.html){
              if(buf.append){ el.insertAdjacentHTML('beforeend', buf.html); }
              else { el.innerHTML = buf.html; }
              try{
                var children=el.children; var excess=children.length-80;
                for(var i=0;i<excess;i++){ el.removeChild(children[0]); }
              }catch(_){ }
              buf.html=''; buf.append=false;
            }
          }catch(e){}
        },160);
      }

      // Config panel + state
      if(!window.injected_overlayCfg){ window.injected_overlayCfg = {}; }
      if(!window.injected_overlayCfgGet){ window.injected_overlayCfgGet = function(){ try { return Object.assign({}, window.injected_overlayCfg); } catch(e){ return null; } }; }
      if(!window.injected_overlayCfgSet){ window.injected_overlayCfgSet = function(patch){ try { patch=patch||{}; Object.assign(window.injected_overlayCfg, patch); try{ if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge(); }catch(e){} try{ if(window.injected_overlayUpdateColorBadge) window.injected_overlayUpdateColorBadge(); }catch(e){} return true; } catch(e){ return false; } }; }
      if(!window.injected_overlayInitAuto){ window.injected_overlayInitAuto = function(){ try { if(window.injected_overlayCfgSet) window.injected_overlayCfgSet({ autoDetect: false, thinkOnce: false }); if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge(); return true; } catch(e){ return false; } }; }
      if(!window.injected_overlayCfgClearThinkOnce){ window.injected_overlayCfgClearThinkOnce = function(){ try { window.injected_overlayCfg.thinkOnce=false; return true; } catch(e){ return false; } }; }

      function ensureConfigPanel(defaultTime){
        var cfg = window.injected_overlayCfg || {};
        var dt = (typeof defaultTime==='number' && isFinite(defaultTime) && defaultTime>0)
          ? defaultTime
          : 5.0;
        // defaults
        var cores = (navigator && navigator.hardwareConcurrency) ? navigator.hardwareConcurrency : 4;
        function optimalWorkers(n){
          if (n >= 16) return Math.min(8, Math.floor(n/2));
          if (n >= 8) return Math.min(6, n-2);
          if (n >= 4) return Math.max(1, n-1);
          return Math.max(1, n);
        }
        if (!cfg._initialized || !(typeof cfg.thinkTime==='number' && isFinite(cfg.thinkTime) && cfg.thinkTime>0)) { cfg.thinkTime = dt; cfg._initialized = true; }
        if (!cfg.mode) cfg.mode = 'seq';
        if (typeof cfg.workers !== 'number') cfg.workers = optimalWorkers(cores);
        if (typeof cfg.showVectors !== 'boolean') cfg.showVectors = true;
        if (typeof cfg.thinkOnce !== 'boolean') cfg.thinkOnce = false;
        if (typeof cfg.autoDetect !== 'boolean') cfg.autoDetect = false;
        if (typeof cfg.halt !== 'boolean') cfg.halt = false;
        if (cfg.myColor !== 'w' && cfg.myColor !== 'b') cfg.myColor = 'w';
        if (typeof cfg.collapsed !== 'boolean') cfg.collapsed = true;
        window.injected_overlayCfg = cfg;
        try{ if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge(); }catch(e){}
        try{ if(window.injected_overlayUpdateColorBadge) window.injected_overlayUpdateColorBadge(); }catch(e){}

        var id='injected_overlay-config';
        var el=document.getElementById(id);
        if(!el){
          el=document.createElement('div');
          el.id=id;
          el.innerHTML = ''+
            '<div style="display:flex;align-items:center;gap:8px;">'
          +   '<button id="cfg-toggle" title="펼치기/접기" style="background:transparent;border:none;color:#fff;cursor:pointer;font-size:14px;line-height:1;padding:2px 4px;">▸</button>'
          +   '<b style="flex:1 1 auto;">AI Settings</b>'
          +   '<div style="margin-left:auto;display:flex;gap:6px;align-items:center;">'
          +     '<button id="cfg-halt" style="background:#F44336;color:#fff;border:none;padding:4px 8px;border-radius:4px;cursor:pointer;">Halt</button>'
          +     '<button id="cfg-think-now" style="background:#2196F3;color:#fff;border:none;padding:4px 8px;border-radius:4px;cursor:pointer;">Run AI</button>'
          +   '</div>'
          + '</div>'
          + '<div id="cfg-body" style="margin-top:6px;">'
          +   '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;align-items:center;">'
          +     '<label>Time (s)</label>'
          +     '<input id="cfg-time" type="number" min="0.2" step="0.1" style="width:100%;padding:2px 4px;">'
          +     '<label>Mode</label>'
          +     '<div>'
          +       '<label style="margin-right:6px;"><input type="radio" name="cfg-mode" value="seq"> SEQ</label>'
          +       '<label><input type="radio" name="cfg-mode" value="mp"> MP</label>'
          +     '</div>'
          +     '<label id="cfg-workers-label">Workers</label>'
          +     '<input id="cfg-workers" type="number" min="1" max="64" step="1" style="width:100%;padding:2px 4px;">'
          +     '<label>Options</label>'
          +     '<div>'
          +       '<label><input id="cfg-vectors" type="checkbox" checked> Vectors</label>'
          +     '</div>'
          +   '</div>'
          + '</div>';
          // Insert settings panel below the badge bar if it exists, otherwise at top
          var __bar = document.getElementById('injected_overlay-badges');
          if(__bar && __bar.parentNode === root){
            if(__bar.nextSibling){ root.insertBefore(el, __bar.nextSibling); }
            else { root.appendChild(el); }
          } else {
            if(root.firstChild){ root.insertBefore(el, root.firstChild); } else { root.appendChild(el); }
          }
        }
        function applyToUi(){
          try{
            var c=window.injected_overlayCfg;
            var time=document.getElementById('cfg-time');
            var workers=document.getElementById('cfg-workers');
            var workersLabel=document.getElementById('cfg-workers-label');
            var vectors=document.getElementById('cfg-vectors');
            var seq=document.querySelector('input[name="cfg-mode"][value="seq"]');
            var mp=document.querySelector('input[name="cfg-mode"][value="mp"]');
            if (time) {
              var tv = (typeof c.thinkTime === 'number') ? c.thinkTime : Number(c.thinkTime||0);
              if (!isFinite(tv) || tv<=0) tv = dt;
              time.value = String(tv.toFixed ? tv.toFixed(1) : tv);
            }
            if (workers) workers.value = String(c.workers);
            if (vectors) vectors.checked = !!c.showVectors;
            if (seq) seq.checked = (c.mode === 'seq');
            if (mp) mp.checked = (c.mode === 'mp');
            var showW = (c.mode === 'mp');
            if (workers) workers.style.display = showW ? '' : 'none';
            if (workersLabel) workersLabel.style.display = showW ? '' : 'none';
            var body=document.getElementById('cfg-body');
            var tg=document.getElementById('cfg-toggle');
            if (body) body.style.display = c.collapsed ? 'none' : 'block';
            if (tg) tg.textContent = c.collapsed ? '▸' : '▾';
          }catch(e){}
        }
        function clamp(v, lo, hi){ v=Number(v||0); if(isNaN(v)) v=0; return Math.max(lo, Math.min(hi, v)); }
        try{ applyToUi(); }catch(e){}
        setTimeout(function(){
          try{
            var time=document.getElementById('cfg-time');
            var workers=document.getElementById('cfg-workers');
            var vectors=document.getElementById('cfg-vectors');
            var seq=document.querySelector('input[name="cfg-mode"][value="seq"]');
            var mp=document.querySelector('input[name="cfg-mode"][value="mp"]');
            var run=document.getElementById('cfg-think-now');
            var halt=document.getElementById('cfg-halt');
            var toggle=document.getElementById('cfg-toggle');
            if (time) time.addEventListener('change', function(){ var v = clamp(time.value, 0.2, 999.0); window.injected_overlayCfgSet({ thinkTime: v }); applyToUi(); });
            if (workers) workers.addEventListener('change', function(){ var v = clamp(workers.value, 1, 64); window.injected_overlayCfgSet({ workers: Math.round(v) }); applyToUi(); });
            if (vectors) vectors.addEventListener('change', function(){ var on = !!vectors.checked; window.injected_overlayCfgSet({ showVectors: on }); applyToUi(); try{ if (!on) { if (window.injected_overlayVectorClear) window.injected_overlayVectorClear(); if (window.injected_overlayVectorLegendClear) window.injected_overlayVectorLegendClear(); } else { if (window.injected_overlayVectorSet) window.injected_overlayVectorSet(window.injected_overlayLastArrows || []); } }catch(e){} });
            if (seq) seq.addEventListener('change', function(){ if(seq.checked) window.injected_overlayCfgSet({ mode: 'seq' }); applyToUi(); });
            if (mp) mp.addEventListener('change', function(){ if(mp.checked) window.injected_overlayCfgSet({ mode: 'mp' }); applyToUi(); });
            if (run) run.addEventListener('click', function(){ window.injected_overlayCfgSet({ thinkOnce: true, autoDetect: true }); applyToUi(); try{ if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge(); }catch(e){} });
            if (halt) halt.addEventListener('click', function(){ try{ window.injected_overlayCfgSet({ halt: true, autoDetect: false, thinkOnce: false }); if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge(); }catch(e){} });
            if (toggle) toggle.addEventListener('click', function(){ try{ window.injected_overlayCfgSet({ collapsed: !window.injected_overlayCfg.collapsed }); }catch(e){} applyToUi(); });
          }catch(e){}
          applyToUi();
        },0);
      }

      // Expose installer
      if(!window.injected_overlayInstall){
        window.injected_overlayInstall = function(defaultTime){ try { ensureConfigPanel(defaultTime); return true; } catch(e){ return false; } };
      }

  // Keep overlay clickability and stacking
      window.injected_overlayClickabilityKeepAlive = window.injected_overlayClickabilityKeepAlive || (function(){
        var tid = null;
        function tick(){
          try{
            var r = document.getElementById('injected_overlay-root');
            if (r){ r.style.pointerEvents='auto'; r.style.zIndex='2147483647'; }
            var cfg = document.getElementById('injected_overlay-config');
            if (cfg){
              cfg.style.pointerEvents='auto';
              cfg.style.zIndex='3';
              cfg.style.position = cfg.style.position || 'relative';
            }
            var log = document.getElementById('injected_overlay-log');
            if (log){ log.style.pointerEvents='none'; log.style.zIndex='2'; }
            var badgeBar = document.getElementById('injected_overlay-badges');
            if (badgeBar){ badgeBar.style.pointerEvents='auto'; badgeBar.style.zIndex='6'; }
            var autoBadge = document.getElementById('injected_overlay-auto-badge');
            if (autoBadge){ autoBadge.style.pointerEvents='auto'; autoBadge.style.zIndex='5'; }
            var colorBadge = document.getElementById('injected_overlay-color-badge');
            if (colorBadge){ colorBadge.style.pointerEvents='auto'; colorBadge.style.zIndex='5'; }
            var legend = document.getElementById('injected_overlay-vector-legend');
            if (legend){ legend.style.pointerEvents='none'; legend.style.zIndex='1'; }
          }catch(e){}
        }
        if (!tid){ tick(); tid = window.setInterval(tick, 500); }
        return function(){ tick(); };
      })();

  // Force toggle capture to always work
      (function(){
        try{
          var bound = false;
          if (window.__overlayToggleBound) bound = true;
          if (!bound){
            document.addEventListener('click', function(ev){
              try{
                var t = ev.target;
                if (!t) return;
                if (t.id === 'cfg-toggle'){
                  ev.preventDefault(); ev.stopPropagation();
                  if (ev.stopImmediatePropagation) ev.stopImmediatePropagation();
                  var body = document.getElementById('cfg-body');
                  var c = window.injected_overlayCfg || (window.injected_overlayCfg = {});
                  var collapsed = true;
                  if (body){ collapsed = (body.style.display === 'none'); }
                  else if (typeof c.collapsed === 'boolean'){ collapsed = c.collapsed; }
                  var next = !collapsed;
                  if (body) body.style.display = next ? 'none' : 'block';
                  c.collapsed = next;
                  // 보정 유지
                  if (window.injected_overlayClickabilityKeepAlive) window.injected_overlayClickabilityKeepAlive();
                }
              }catch(e){}
            }, true);
            window.__overlayToggleBound = true;
          }
        }catch(e){}
      })();

  // Auto-install is handled by overlay_bootstrap.js
    }
    // Defer boot until body exists
    if (document && document.body) {
      boot();
    } else {
      document.addEventListener('DOMContentLoaded', function(){ try{ boot(); }catch(e){} }, { once: true });
      var __ovl_wait_i = 0; var __ovl_wait_tid = setInterval(function(){
        try{
          if (document && document.body){ clearInterval(__ovl_wait_tid); boot(); }
          else if(++__ovl_wait_i > 200){ clearInterval(__ovl_wait_tid); }
        }catch(e){ clearInterval(__ovl_wait_tid); }
      }, 50);
    }
  }catch(e){}
})();
