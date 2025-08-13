(function(){
  try {
    var CSS_ID = 'injected_overlay-style';
    var CSS_TEXT = (typeof window !== 'undefined' && window.__overlayCssText) ? String(window.__overlayCssText) : '';
    var __ens_tid = null;
    var __ens_pending = false;
    var __last_ens = 0;

    function ensureCss(){
      try {
        var s = document.getElementById(CSS_ID);
        if (!s) {
          s = document.createElement('style');
          s.id = CSS_ID;
          s.textContent = CSS_TEXT;
          (document.head || document.documentElement).appendChild(s);
        } else if (s.textContent !== CSS_TEXT) {
          s.textContent = CSS_TEXT;
        }
  } catch (e) {}
    }

    function ensureOverlay(){
      try {
        var hasRoot = !!document.getElementById('injected_overlay-root');
        var hasConfig = !!document.getElementById('injected_overlay-config');
        if (!hasConfig && window.injected_overlayInstall) {
          if (window.getThinkTimeFromPython) {
            window.getThinkTimeFromPython().then(function(val) {
              window.injected_overlayInstall(val);
            });
          } else {
            window.injected_overlayInstall();
          }
        }
        var hasBadges = !!document.getElementById('injected_overlay-badges');
        if (!hasBadges && window.injected_overlayInstallBadges) {
          window.injected_overlayInstallBadges();
        }
        if (window.injected_overlayClickabilityKeepAlive) {
          try { window.injected_overlayClickabilityKeepAlive(); } catch(e){}
        }
      } catch (e) {}
    }

    function ensureAll(){
      ensureCss();
      ensureOverlay();
      __last_ens = Date.now();
      __ens_pending = false;
    }

    function scheduleEnsure(delay){
      if (delay == null) delay = 80;
      if (__ens_pending) return;
      __ens_pending = true;
      if (__ens_tid) { try{ clearTimeout(__ens_tid); }catch(e){} }
      __ens_tid = setTimeout(function(){ __ens_tid = null; ensureAll(); }, delay);
    }

  // initial (deferred one tick)
    scheduleEnsure(0);

  // keepalive timers
    if (!window.__overlayBootstrapCssKeepAlive) {
      window.__overlayBootstrapCssKeepAlive = setInterval(ensureCss, 600);
    }
    if (!window.__overlayBootstrapUiKeepAlive) {
      window.__overlayBootstrapUiKeepAlive = setInterval(function(){
        try {
          var hasRoot = !!document.getElementById('injected_overlay-root');
          if (!hasRoot) scheduleEnsure(100);
    } catch (e) {}
      }, 900);
    }

  // DOM hooks
    try {
      var mo = new MutationObserver(function(){ scheduleEnsure(120); });
      mo.observe(document.documentElement, { childList: true, subtree: true });
      window.__overlayBootstrapObserver = mo;
  } catch (e) {}

  // SPA hooks
    try {
      var _pushState = history.pushState;
      history.pushState = function(){ try{ _pushState.apply(this, arguments); }catch(e){} scheduleEnsure(80); };
      window.addEventListener('popstate', function(){ scheduleEnsure(80); });
  } catch (e) {}

  // DOMContentLoaded hook
    try {
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function(){ scheduleEnsure(0); }, { once: true });
      }
  } catch (e) {}

  // expose manual trigger
    window.injected_overlayBootstrapEnsure = function(){ scheduleEnsure(0); };
  } catch (e) {}

  if (window.getThinkTimeFromPython) {
    window.getThinkTimeFromPython().then(function(val) {
      if (window.injected_overlayInstall) window.injected_overlayInstall(val);
    });
  } else {
    if (window.injected_overlayInstall) window.injected_overlayInstall();
  }
})();
