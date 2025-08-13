(function(){
  try{
    if (window.injected_overlayInstallBadges) return;

    function ensureRoot(){
      var rootId='injected_overlay-root';
      var root=document.getElementById(rootId);
      if(!root){
        root=document.createElement('div');
        root.id=rootId;
        document.body.appendChild(root);
      }
      return root;
    }

    function ensureBar(root){
      var barId='injected_overlay-badges';
      var bar=document.getElementById(barId);
      if(!bar){
        bar=document.createElement('div');
        bar.id=barId;
        if(root.firstChild){ root.insertBefore(bar, root.firstChild); } else { root.appendChild(bar); }
      }
      return bar;
    }

    function installAutoBadge(root, bar){
      var id='injected_overlay-auto-badge';
      var el=document.getElementById(id);
      if(!el){
        el=document.createElement('div');
        el.id=id;
        el.className='overlay-badge auto-badge';
        el.textContent='Auto: —';
        (bar||root).appendChild(el);
        try{
          el.addEventListener('click', function(ev){
            try{
              ev.preventDefault(); ev.stopPropagation(); if(ev.stopImmediatePropagation) ev.stopImmediatePropagation();
              var c = window.injected_overlayCfg || (window.injected_overlayCfg = {});
              var next = !c.autoDetect;
              if(window.injected_overlayCfgSet) window.injected_overlayCfgSet({ autoDetect: next });
              if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge();
            }catch(e){}
          });
        }catch(e){}
      }
      window.injected_overlayUpdateAutoBadge = window.injected_overlayUpdateAutoBadge || function(){
        try{
          var c = window.injected_overlayCfg || {};
          var on = !!c.autoDetect;
          var el = document.getElementById(id);
          if(!el) return;
          el.textContent = on ? 'Auto: ON' : 'Auto: OFF';
          el.classList.toggle('on', on);
          el.classList.toggle('off', !on);
        }catch(e){}
      };
      try{ if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge(); }catch(e){}
    }

    function installColorBadge(root, bar){
      var id='injected_overlay-color-badge';
      var el=document.getElementById(id);
      if(!el){
        el=document.createElement('div');
        el.id=id;
        el.className='overlay-badge color-badge';
        el.textContent='Color: —';
        (bar||root).appendChild(el);
        try{
          el.addEventListener('click', function(ev){
            try{
              ev.preventDefault(); ev.stopPropagation(); if(ev.stopImmediatePropagation) ev.stopImmediatePropagation();
              var c = window.injected_overlayCfg || (window.injected_overlayCfg = {});
              var cur = (c.myColor === 'b') ? 'b' : 'w';
              var next = (cur === 'w') ? 'b' : 'w';
              if(window.injected_overlayCfgSet) window.injected_overlayCfgSet({ myColor: next });
              if(window.injected_overlayUpdateColorBadge) window.injected_overlayUpdateColorBadge();
            }catch(e){}
          });
        }catch(e){}
      }
      window.injected_overlayUpdateColorBadge = window.injected_overlayUpdateColorBadge || function(){
        try{
          var c = window.injected_overlayCfg || {};
          var v = (c.myColor === 'b') ? 'B' : 'W';
          var el = document.getElementById(id);
          if(!el) return;
          el.textContent = 'Color: ' + v;
          el.classList.toggle('black', v === 'B');
          el.classList.toggle('white', v !== 'B');
        }catch(e){}
      };
      try{ if(window.injected_overlayUpdateColorBadge) window.injected_overlayUpdateColorBadge(); }catch(e){}
    }

    window.injected_overlayInstallBadges = function(){
      try{
        var root = ensureRoot();
        var bar = ensureBar(root);
        installAutoBadge(root, bar);
        installColorBadge(root, bar);
        if (window.injected_overlayClickabilityKeepAlive) window.injected_overlayClickabilityKeepAlive();
        return true;
      }catch(e){ return false; }
    };

  }catch(e){}
})();
