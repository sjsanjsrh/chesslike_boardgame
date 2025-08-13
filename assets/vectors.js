(function(){
  try{
    // Keep last arrows for toggle restore
    window.injected_overlayLastArrows = window.injected_overlayLastArrows || [];

  function ensureRoot(){
      var rootId='injected_overlay-root';
      var root=document.getElementById(rootId);
      if(!root){
        root=document.createElement('div');
        root.id=rootId;
        var rs=root.style;
        rs.position='fixed'; rs.top='8px'; rs.right='8px'; rs.zIndex='2147483647';
        rs.display='flex'; rs.flexDirection='column'; rs.alignItems='stretch'; rs.gap='8px';
    // Keep overlay root clickable so the settings toggle works during AI.
    rs.pointerEvents='auto';
        document.body.appendChild(root);
      }
      return root;
    }
    function ensureSvg(boardRect){
      var id='injected_overlay-vector-svg';
      var svg=document.getElementById(id);
      if(!svg){
        svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
        svg.id=id;
        var s=svg.style; s.position='fixed'; s.pointerEvents='none'; s.zIndex='2147483646';
        document.body.appendChild(svg);
      }
      svg.setAttribute('width', String(boardRect.width));
      svg.setAttribute('height', String(boardRect.height));
      var s2=svg.style; s2.left=boardRect.left+'px'; s2.top=boardRect.top+'px';
      return svg;
    }
    function clearSvg(){ var svg=document.getElementById('injected_overlay-vector-svg'); if(svg){ while(svg.firstChild) svg.removeChild(svg.firstChild); } }
    function ensureLegend(){
      var root=ensureRoot();
      var id='injected_overlay-vector-legend';
      var el=document.getElementById(id);
      if(!el){
        el=document.createElement('div'); el.id=id;
        var s=el.style; s.position='relative'; s.maxWidth='40vw'; s.minWidth='200px';
        s.background='rgba(0,0,0,0.70)'; s.color='#fff'; s.padding='8px 10px'; s.font='12px/1.4 monospace';
        s.borderRadius='6px'; s.border='1px solid rgba(255,255,255,0.25)'; s.pointerEvents='none';
        root.appendChild(el);
      }
      return el;
    }
    function clearLegend(){ var el=document.getElementById('injected_overlay-vector-legend'); if(el){ el.textContent=''; } }

    function isFlipped(){
      try{
        var el1 = document.querySelector('wc-chess-board');
        if (el1 && el1.classList && el1.classList.contains('flipped')) return true;
        // var el2 = document.getElementById('board-play-computer');
        // if (el2 && el2.classList && el2.classList.contains('flipped')) return true;
      }catch(e){}
      return false;
    }
    function xyCenter(xy, boardRect){
      var f=Number(xy[0]); var rk=Number(xy[1]);
      if(!(f>=1&&f<=8&&rk>=1&&rk<=8)) return null;
      var flip = isFlipped();
      var col = flip ? (8 - f) : (f - 1);
      var rowTop = flip ? (rk - 1) : (8 - rk);
      var cx=(col+0.5)*(boardRect.width/8); var cy=(rowTop+0.5)*(boardRect.height/8);
      return {x:cx, y:cy};
    }

    if(!window.injected_overlayVectorClear){ window.injected_overlayVectorClear = function(){ try{ clearSvg(); }catch(e){} }; }
    if(!window.injected_overlayVectorLegendClear){ window.injected_overlayVectorLegendClear = function(){ try{ clearLegend(); }catch(e){} }; }

    if(!window.injected_overlayVectorSet){
      window.injected_overlayVectorSet = function(arrows){
        try{
          arrows = Array.isArray(arrows) ? arrows : [];
          try{ window.injected_overlayLastArrows = arrows.slice(); }catch(e){}
          try{
            if (window.injected_overlayCfg && window.injected_overlayCfg.showVectors === false){ clearSvg(); clearLegend(); return true; }
          }catch(e){}
          // var host=document.getElementById('board-play-computer');
          var host=document.querySelector('wc-chess-board');
          if(!host){ clearSvg(); clearLegend(); return true; }
          var r=host.getBoundingClientRect();
          var svg=ensureSvg(r); clearSvg(); var ns=svg.namespaceURI;

          arrows.forEach(function(a){
            var p1=xyCenter(a.fromXY, r); var p2=xyCenter(a.toXY, r);
            if(!p1||!p2) return;
            var line=document.createElementNS(ns,'line');
            line.setAttribute('x1', String(p1.x)); line.setAttribute('y1', String(p1.y));
            line.setAttribute('x2', String(p2.x)); line.setAttribute('y2', String(p2.y));
            line.setAttribute('stroke', a.color||'#ff0'); line.setAttribute('stroke-width', String(a.width||3));
            line.setAttribute('stroke-linecap','round'); line.setAttribute('opacity','0.65'); svg.appendChild(line);
            var radius=(a.width ? (a.width*3) : 8); var circle=document.createElementNS(ns,'circle');
            circle.setAttribute('cx', String(p2.x)); circle.setAttribute('cy', String(p2.y)); circle.setAttribute('r', String(radius));
            circle.setAttribute('fill', a.color||'#ff0'); circle.setAttribute('fill-opacity','0.22'); circle.setAttribute('stroke', a.color||'#ff0');
            circle.setAttribute('stroke-opacity','0.9'); circle.setAttribute('stroke-width','1.5'); svg.appendChild(circle);
          });

          var legend=ensureLegend(); var html='';
          for(var i=0;i<arrows.length;i++){
            var a=arrows[i]; var color=a.color||'#ff0'; var label=a.label||'';
            html += '<div style="display:flex;align-items:center;margin:2px 0;">'
                 + '<span style="display:inline-block;width:10px;height:10px;background:'+color+';border:1px solid rgba(255,255,255,0.8);margin-right:6px;"></span>'
                 + String(i+1)+'. '+ label + '</div>';
          }
          legend.innerHTML=html; return true;
        }catch(e){ return false; }
      };
    }

    if(!window.injected_overlayVectorShowLast){ window.injected_overlayVectorShowLast = function(){ try{ return window.injected_overlayVectorSet(window.injected_overlayLastArrows || []); }catch(e){ return false; } }; }
  }catch(e){}
})();
