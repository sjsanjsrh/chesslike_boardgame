from __future__ import annotations

import time
import re
from pathlib import Path
import threading
import queue
from typing import List, Optional, Tuple

from alphabeta import fen_to_board, GameState, rc_to_algebraic, iterative_deepening_search
from simple_multiprocess import get_multiprocess_move

# Playwright (sync API)
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


THINK_TIME = 5.0
FILES = 'abcdefgh'
VECTOR_COLORS = ["#E91E63", "#B56E3F", "#009688", "#FF9800", "#9C27B0"]
PIECE_CODES = {
    'wk': 'K', 'wq': 'Q', 'wr': 'R', 'wb': 'B', 'wn': 'N', 'wp': 'P',
    'bk': 'k', 'bq': 'q', 'br': 'r', 'bb': 'b', 'bn': 'n', 'bp': 'p',
}

BASE_DIR = Path(__file__).resolve().parent

# Note: Playwright on Windows requires ProactorEventLoop for subprocess; avoid forcing Selector policy here.

# PyPy on Windows: Proactor WriteFile may choke on bytearray. Convert to bytes in IocpProactor.send.
def _patch_pypy_proactor_bytearray():
    try:
        import sys as _sys
        if getattr(_sys, 'implementation', None) and getattr(_sys.implementation, 'name', '') == 'pypy' and _sys.platform.startswith('win'):
            import asyncio as _asyncio
            try:
                from asyncio import windows_events as _we
                _orig_send = _we.IocpProactor.send
                def _patched_send(self, conn, data):
                    try:
                        if isinstance(data, bytearray):
                            data = bytes(data)
                    except Exception:
                        pass
                    return _orig_send(self, conn, data)
                _we.IocpProactor.send = _patched_send  # type: ignore
            except Exception:
                pass
    except Exception:
        pass

_patch_pypy_proactor_bytearray()


def _compress_fen_row(cells: List[Optional[str]]) -> str:
    out = []
    empty = 0
    for ch in cells:
        if ch is None:
            empty += 1
        else:
            if empty:
                out.append(str(empty))
                empty = 0
            out.append(ch)
    if empty:
        out.append(str(empty))
    return ''.join(out)


def parse_chesscom_html_to_fen(html: str) -> str:
    board: List[List[Optional[str]]] = [[None for _ in range(8)] for _ in range(8)]
    for m in re.finditer(r'<div[^>]*\bclass="([^"]*)"[^>]*>', html, flags=re.IGNORECASE):
        classes = m.group(1).split()
        if not ("piece" in classes or "promotion-piece" in classes):
            continue
        pc_code = None
        square_token = None
        for cls in classes:
            if cls in PIECE_CODES:
                pc_code = cls
            elif cls.startswith('square-') and len(cls) >= 9:
                square_token = cls
        if not pc_code or not square_token:
            continue
        piece_char = PIECE_CODES[pc_code]
        xy = square_token.split('-', 1)[1]
        if len(xy) != 2 or not xy.isdigit():
            continue
        file_idx = int(xy[0])
        rank_idx = int(xy[1])
        if not (1 <= file_idx <= 8 and 1 <= rank_idx <= 8):
            continue
        row = 8 - rank_idx
        col = file_idx - 1
        board[row][col] = piece_char
    fen_rows: List[str] = [_compress_fen_row(board[r]) for r in range(8)]
    return '/'.join(fen_rows)

def parse_chesscom_html_to_piece_markers(html: str) -> list:
    """Extract current recognized piece positions as UI markers.
    Returns list of dicts: { xy: 'fr', label: 'P', color: '#...' }
    """
    markers: list = []
    for m in re.finditer(r'<div[^>]*\bclass="([^"]*)"[^>]*>', html, flags=re.IGNORECASE):
        classes = m.group(1).split()
        if not ("piece" in classes or "promotion-piece" in classes):
            continue
        pc_code = None
        square_token = None
        for cls in classes:
            if cls in PIECE_CODES:
                pc_code = cls
            elif cls.startswith('square-') and len(cls) >= 9:
                square_token = cls
        if not pc_code or not square_token:
            continue
        piece_char = PIECE_CODES.get(pc_code)
        xy = square_token.split('-', 1)[1]
        if not piece_char or len(xy) != 2 or not xy.isdigit():
            continue
        color = '#4CAF50' if piece_char.isupper() else '#F44336'
        markers.append({ 'xy': xy, 'label': piece_char, 'color': color })
    return markers

# --- Conservative FEN helpers for move plausibility ---
def _fen_to_grid(fen_rows: str) -> list:
    rows = fen_rows.split('/')
    grid = []
    for r in rows:
        row = []
        for ch in r:
            if ch.isdigit():
                row.extend(['.'] * int(ch))
            else:
                row.append(ch)
        # guard
        row = (row + ['.'] * 8)[:8]
        grid.append(row)
    # guard
    while len(grid) < 8:
        grid.append(['.'] * 8)
    return grid[:8]

def _is_white_piece(ch: str) -> bool:
    return len(ch) == 1 and ch.isupper()

def _is_black_piece(ch: str) -> bool:
    return len(ch) == 1 and ch.islower()

def _plausible_opponent_move(f_before: str, f_after: str, my_color: str) -> bool:
    try:
        g1 = _fen_to_grid(f_before)
        g2 = _fen_to_grid(f_after)
        opp_is_white = (str(my_color).lower().startswith('b'))
        is_opp = (_is_white_piece if opp_is_white else _is_black_piece)
        is_me = (_is_black_piece if opp_is_white else _is_white_piece)

        changed = 0
        opp_from = 0
        opp_to = 0
        me_added = 0
        me_removed = 0
        for r in range(8):
            for c in range(8):
                a = g1[r][c]
                b = g2[r][c]
                if a == b:
                    continue
                changed += 1
                if is_opp(a) and (b == '.' or is_me(b) or is_opp(b)):
                    opp_from += 1
                if is_opp(b) and (a == '.' or is_me(a)):
                    opp_to += 1
                if is_me(b) and not is_me(a):
                    me_added += 1
                if is_me(a) and not is_me(b):
                    me_removed += 1

        # Conservative acceptance rules:
        # - Opponent must have at least one from and at least one to square
        # - My pieces must not increase
        # - My pieces can be captured at most one
        # - Limit total changed squares to a small number to avoid noisy changes
        if opp_from >= 1 and opp_to >= 1 and me_added == 0 and me_removed in (0, 1) and changed <= 6:
            return True
        return False
    except Exception:
        # On failure, be conservative (reject)
        return False

def load_state_from_chesscom_html(html: str, turn: str = 'w') -> GameState:
    fen = parse_chesscom_html_to_fen(html)
    gs = GameState()
    gs.board = fen_to_board(fen)
    gs.turn = 'w' if turn not in ('w', 'b') else turn
    return gs

def _board_flipped(html: str) -> bool:
    board_class = None
    m = re.search(r'<wc-chess-board[^>]*\bclass="([^"]*)"', html, flags=re.IGNORECASE)
    if m:
        board_class = m.group(1)
    if board_class:
        cls_tokens = board_class.split()
        flipped = ('flipped' in cls_tokens)
        return flipped
    return False


def _sq_to_xy(square: str) -> str:
    file_ch = square[0].lower()
    rank = int(square[1])
    file_idx = ord(file_ch) - ord('a') + 1
    return f"{file_idx}{rank}"


def _rc_to_ui_algebraic(rc: Tuple[int, int]) -> str:
    r, c = rc
    return f"{FILES[c]}{8 - r}"


# ========== Asset injection & overlay/vectors helpers ==========
def _read_text(path: Path) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def _register_init_assets(context) -> None:
    """Register init scripts at the context level so they apply to all future pages/documents."""
    overlay_js = _read_text(BASE_DIR / 'assets' / 'overlay.js')
    vectors_js = _read_text(BASE_DIR / 'assets' / 'vectors.js')
    badges_js = _read_text(BASE_DIR / 'assets' / 'badges.js')
    overlay_css = _read_text(BASE_DIR / 'assets' / 'overlay.css')
    overlay_bootstrap = _read_text(BASE_DIR / 'assets' / 'overlay_bootstrap.js')

    # CSS text must be available before other scripts
    context.add_init_script(script=f"window.__overlayCssText = {overlay_css!r};")
    # Core scripts
    context.add_init_script(script=overlay_js)
    context.add_init_script(script=vectors_js)
    context.add_init_script(script=badges_js)
    context.add_init_script(script=overlay_bootstrap)
    # Fallback: ensure style present early
    context.add_init_script(script=f"(function(){{ try{{ if(!document.getElementById('injected_overlay-style')){{ var s=document.createElement('style'); s.id='injected_overlay-style'; s.textContent={overlay_css!r}; (document.head||document.documentElement).appendChild(s); }} }}catch(e){{}} }})();")


def _overlay_set(page, html: str, append: bool = False) -> None:
    page.evaluate(
        "(arg)=>{ if(window.injected_overlayOverlaySet){ return window.injected_overlayOverlaySet(arg.h, arg.a===true); } }",
        {"h": html, "a": append},
    )


def _ensure_config_ui(page, default_time: float = THINK_TIME) -> None:
    # Assets should already be present via context init scripts; just install UI
    page.evaluate("(t)=>{ if(window.injected_overlayInstall){ window.injected_overlayInstall(t); } }", float(default_time))
    page.evaluate("()=>{ if(window.injected_overlayInstallBadges){ window.injected_overlayInstallBadges(); } }")


def _overlay_init_auto(page) -> None:
    try:
        page.evaluate("()=>{ if(window.injected_overlayInitAuto){ window.injected_overlayInitAuto(); } if(window.injected_overlayInstallBadges){ window.injected_overlayInstallBadges(); } }")
    except Exception:
        pass


def _overlay_is_halt(page) -> bool:
    try:
        return bool(page.evaluate("()=> (window.injected_overlayCfg && window.injected_overlayCfg.halt) || false"))
    except Exception:
        return False


def _overlay_set_halt(page) -> None:
    """오버레이 Halt를 강제로 켭니다."""
    try:
        page.evaluate("()=>{ if(window.injected_overlayCfgSet){ window.injected_overlayCfgSet({ halt: true, autoDetect: false, thinkOnce: false }); if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge(); } }")
    except Exception:
        pass

def _force_halt_reset(page) -> None:
    """Force-stop UI and reset overlay state immediately while keeping Halt ON."""
    try:
        page.evaluate(
            "()=>{ if(window.injected_overlayCfgSet){ window.injected_overlayCfgSet({ halt: true, autoDetect: false, thinkOnce: false }); if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge(); } }"
        )
    except Exception:
        pass
    try:
        _vector_clear(page)
    except Exception:
        pass
    try:
        _overlay_set(page, "<div><b>AI</b> <span style='opacity:.85'>halted</span></div>")
    except Exception:
        pass

def _overlay_clear_halt_keep_cfg(page) -> None:
    """Turn off halt without changing other config flags, update badges if available."""
    try:
        page.evaluate(
            "()=>{ if(window.injected_overlayCfgSet){ window.injected_overlayCfgSet({ halt: false }); if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge(); } }"
        )
    except Exception:
        pass

def _is_game_over_modal_open(page) -> bool:
    """game-over-modal-content 가 DOM에 나타났는지 확인."""
    try:
        loc = page.locator('.game-over-modal-content')
        return bool(loc) and loc.first.is_visible()
    except Exception:
        return False


def _vector_set(page, arrows: list) -> None:
    page.evaluate("(arr)=>{ if(window.injected_overlayVectorSet){ window.injected_overlayVectorSet(arr); } }", arrows)


def _vector_clear(page) -> None:
    page.evaluate("()=>{ if(window.injected_overlayVectorClear){ window.injected_overlayVectorClear(); } if(window.injected_overlayVectorLegendClear){ window.injected_overlayVectorLegendClear(); } }")


def _squares_set(page, squares: list) -> None:
    page.evaluate("(arr)=>{ if(window.injected_overlaySquaresSet){ window.injected_overlaySquaresSet(arr); } }", squares)

def _squares_clear(page) -> None:
    page.evaluate("()=>{ if(window.injected_overlaySquaresClear){ window.injected_overlaySquaresClear(); } }")

def _pl_click_move_simple(page, from_sq: str, to_sq: str) -> None:
    """빠른 클릭: 원/목표만 클릭하고 검증은 생략."""
    from_xy = _sq_to_xy(from_sq)
    to_xy = _sq_to_xy(to_sq)
    board = page.locator('wc-chess-board')
    try:
        board.locator(f'.square-{from_xy}').first.click(force=True, timeout=800)
    except Exception:
        try:
            board.locator(f'.piece.square-{from_xy}').first.click(force=True, timeout=800)
        except Exception:
            pass
    # 좌표 클릭
    bbox = board.bounding_box()
    if not bbox:
        return
    bx, by, bw, bh = bbox.values()
    try:
        tf = int(to_xy[0]); tr = int(to_xy[1])
    except Exception:
        return
    flipped = False
    try:
        flipped = _board_flipped(page.content())
    except Exception:
        pass
    if flipped:
        tx = bx + ((8 - tf + 0.5) * (bw / 8))
        ty = by + ((tr - 1 + 0.5) * (bh / 8))
    else:
        tx = bx + ((tf - 1 + 0.5) * (bw / 8))
        ty = by + ((8 - tr + 0.5) * (bh / 8))
    try:
        page.mouse.click(tx, ty)
    except Exception:
        pass

def _pl_click_move(page, from_sq: str, to_sq: str, my_color: str, timeout_ms: int = 6000) -> None:
    """Click a move by selecting a piece and then clicking the destination square directly.
    After clicks, verify the move applied by checking DOM changes (origin piece removed or destination occupied).
    """
    from_xy = _sq_to_xy(from_sq)
    to_xy = _sq_to_xy(to_sq)

    board = page.locator('wc-chess-board')

    for attempt in range(3):
        if _overlay_is_halt(page):
            _force_halt_reset(page)
            raise Exception("halted")
        # 1) Select the piece
        piece_locators = [
            board.locator(f'.piece.square-{from_xy}').first,
            board.locator(f'.square-{from_xy}').first
        ]
        last_exc = None
        clicked_piece = False
        for pl in piece_locators:
            try:
                pl.wait_for(state='attached', timeout=500)
                pl.click(force=True, timeout=500)
                clicked_piece = True
                break
            except Exception as e:
                last_exc = e
                time.sleep(0.1)
                continue

        # 2) Wait for the piece to be selected (highlight element added)
        hl = board.locator(f'.highlight.square-{from_xy}')
        highlighted = False
        try:
            hl.wait_for(state='attached', timeout=500)
            highlighted = True
        except Exception:
            time.sleep(0.1)

    if not clicked_piece:
        raise last_exc or Exception(f"Piece .square-{from_xy} not clickable")
    if not highlighted:
        raise Exception(f"Piece .square-{from_xy} not highlighted")

    # Baseline before applying the move (for robust verification)
    try:
        html_before = _get_current_html(page)
        fen_before = parse_chesscom_html_to_fen(html_before)
        opp_is_white = (str(my_color).lower().startswith('b'))
        me_is_white = not opp_is_white
        cnt_opp_before = sum(1 for ch in fen_before if (ch.isupper() if opp_is_white else ch.islower()))
        cnt_me_before = sum(1 for ch in fen_before if (ch.isupper() if me_is_white else ch.islower()))
    except Exception:
        fen_before = ''
        cnt_opp_before = -1
        cnt_me_before = -1

    # 3) Select Move Target
    if _overlay_is_halt(page):
        _force_halt_reset(page)
        raise Exception("halted")
    bbox = board.bounding_box()
    if not bbox:
        raise Exception("Could not get board bounding box")
    board_x, board_y, board_w, board_h = bbox.values()
    try:
        to_file = int(to_xy[0])
        to_rank = int(to_xy[1])
    except Exception:
        raise Exception(f"Invalid to_xy: {to_xy}")
    if _board_flipped(page.content()):
        to_x = board_x + ((8 - to_file + 0.5) * (board_w / 8))
        to_y = board_y + ((to_rank - 1 + 0.5) * (board_h / 8))
    else:
        to_x = board_x + ((to_file - 1 + 0.5) * (board_w / 8))
        to_y = board_y + ((8 - to_rank + 0.5) * (board_h / 8))
    page.mouse.click(to_x, to_y)

    # 4) Verify applied robustly
    def _origin_empty() -> bool:
        try:
            return board.locator(f'.piece.square-{from_xy}').count() == 0
        except Exception:
            return False

    def _dest_has_piece() -> bool:
        try:
            return board.locator(f'.piece.square-{to_xy}').count() > 0
        except Exception:
            return False

    def _stable_fen_pair() -> tuple[str, str]:
        try:
            f1 = parse_chesscom_html_to_fen(_get_current_html(page))
            time.sleep(0.1)
            f2 = parse_chesscom_html_to_fen(_get_current_html(page))
            return f1, f2
        except Exception:
            return ('', '')

    deadline = time.time() + max(0.5, timeout_ms/1000.0)
    last_seen_fen = fen_before
    while time.time() < deadline:
        if _overlay_is_halt(page):
            _force_halt_reset(page)
            raise Exception("halted")
        # Fast DOM condition: destination has piece and origin is empty
        if _dest_has_piece() and _origin_empty():
            f1, f2 = _stable_fen_pair()
            if f1 and f1 == f2:
                try:
                    opp_is_white = (str(my_color).lower().startswith('b'))
                    me_is_white = not opp_is_white
                    cnt_opp_after = sum(1 for ch in f2 if (ch.isupper() if opp_is_white else ch.islower()))
                    cnt_me_after = sum(1 for ch in f2 if (ch.isupper() if me_is_white else ch.islower()))
                    # Our move: opponent piece-count must be same or -1 (if we captured),
                    # and our piece-count should remain the same.
                    if (cnt_opp_before < 0 or cnt_opp_after in (cnt_opp_before, cnt_opp_before - 1)) and \
                       (cnt_me_before < 0 or cnt_me_after == cnt_me_before):
                        return
                except Exception:
                    return

        # FEN-based condition: changed and stable with valid piece-count delta
        try:
            fen_now = parse_chesscom_html_to_fen(_get_current_html(page))
        except Exception:
            fen_now = ''
        if fen_now and fen_now != last_seen_fen:
            time.sleep(0.1)
            try:
                fen_conf = parse_chesscom_html_to_fen(_get_current_html(page))
            except Exception:
                fen_conf = ''
            if fen_conf and fen_conf == fen_now:
                try:
                    opp_is_white = (str(my_color).lower().startswith('b'))
                    me_is_white = not opp_is_white
                    cnt_opp_now = sum(1 for ch in fen_now if (ch.isupper() if opp_is_white else ch.islower()))
                    cnt_me_now = sum(1 for ch in fen_now if (ch.isupper() if me_is_white else ch.islower()))
                    # Our move verification: opp count same or -1; my count unchanged
                    if (cnt_opp_before < 0 or cnt_opp_now in (cnt_opp_before, cnt_opp_before - 1)) and \
                       (cnt_me_before < 0 or cnt_me_now == cnt_me_before):
                        return
                except Exception:
                    return
            last_seen_fen = fen_now

        time.sleep(0.06)
    raise PWTimeout(f"Move not applied for {from_sq}->{to_sq} within {timeout_ms}ms")


def _build_arrows_from_top_moves(top_moves: list) -> list:
    arrows = []
    for idx, item in enumerate(top_moves or []):
        fr = item.get('from'); to = item.get('to')
        if not fr or not to:
            continue
        try:
            fr_alg = rc_to_algebraic(tuple(fr))
            to_alg = rc_to_algebraic(tuple(to))
            fr_xy = _sq_to_xy(fr_alg)
            to_xy = _sq_to_xy(to_alg)
        except Exception:
            continue
        col = VECTOR_COLORS[idx % len(VECTOR_COLORS)]
        w = 4 if idx == 0 else 3 if idx in (1,2) else 2
        try:
            val = item.get('val')
            val_txt = f" {float(val):.2f}" if val is not None else ''
        except Exception:
            val_txt = ''
        label = f"{fr_alg}->{to_alg}{val_txt}"
        arrows.append({'fromXY': fr_xy, 'toXY': to_xy, 'color': col, 'width': w, 'label': label})
    return arrows


# ================= Background Analysis Worker =================
class AnalysisWorker:
    class Cancelled(Exception):
        pass

    def __init__(self):
        self._in_q: "queue.Queue[dict]" = queue.Queue()
        self._out_q: "queue.Queue[dict]" = queue.Queue()
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._job_guard = threading.Lock()
        self._cancel_current = threading.Event()

    @property
    def out_queue(self):
        return self._out_q

    def start(self):
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name='AnalysisWorker', daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        try:
            self._in_q.put_nowait({'type': 'stop'})
        except Exception:
            pass
        if self._thr:
            try:
                self._thr.join(timeout=1.5)
            except Exception:
                pass

    def submit(self, html: str, my_color: str, think_time: float, mode: str, workers: int, want_click: bool):
        """Submit a new analysis job, cancelling any previous."""
        with self._job_guard:
            self._cancel_current.set()  # signal cancel to current job
            try:
                self._in_q.put_nowait({
                    'type': 'job', 'html': html, 'my_color': my_color,
                    'think_time': float(think_time), 'mode': str(mode or 'seq'),
                    'workers': int(max(1, workers or 1)), 'want_click': bool(want_click)
                })
            except Exception:
                pass

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._in_q.get(timeout=0.2)
            except Exception:
                continue
            if not item or item.get('type') == 'stop':
                break
            if item.get('type') != 'job':
                continue
            # Prepare job
            html = item.get('html') or ''
            my_color = 'b' if str(item.get('my_color') or '').lower().startswith('b') else 'w'
            think_time = float(item.get('think_time') or THINK_TIME)
            mode = str(item.get('mode') or 'seq')
            workers = int(item.get('workers') or 4)
            want_click = bool(item.get('want_click') or False)
            # Reset cancel flag for this job
            self._cancel_current.clear()

            gs = load_state_from_chesscom_html(html, turn=my_color)
            start_ts = time.time()
            last_best = {'move': None, 'val': None}

            def _progress(evt: dict):
                if self._cancel_current.is_set():
                    raise AnalysisWorker.Cancelled()
                # massage values to AI perspective for display only
                e = dict(evt or {})
                try:
                    bv = e.get('best_val')
                    if isinstance(bv, (int, float)) and my_color == 'b':
                        e['best_val'] = -float(bv)
                except Exception:
                    pass
                # keep last seen best
                try:
                    if e.get('best_move'):
                        last_best['move'] = e.get('best_move')
                        last_best['val'] = e.get('best_val')
                except Exception:
                    pass
                # map top_moves values for label perspective
                try:
                    tm = e.get('top_moves')
                    if isinstance(tm, list):
                        mm = []
                        for it in tm:
                            try:
                                v = it.get('val')
                                if isinstance(v, (int, float)) and my_color == 'b':
                                    v = -float(v)
                                mm.append({'from': it.get('from'), 'to': it.get('to'), 'val': v})
                            except Exception:
                                mm.append(it)
                        e['top_moves'] = mm
                except Exception:
                    pass
                try:
                    self._out_q.put_nowait({'type': 'progress', 'evt': e, 'cfg': {
                        'mode': mode, 'think_time': think_time, 'workers': workers, 'my_color': my_color
                    }, 'last_best': dict(last_best)})
                except Exception:
                    pass

            # run engine
            try:
                if (mode or '').lower().startswith('mp'):
                    def _mp_progress(evt: dict):
                        _progress(evt)
                    move, val, nodes, depth = get_multiprocess_move(gs, think_time, max(1, int(workers)), progress=_mp_progress, top_k=5)
                else:
                    val, move, nodes, depth = iterative_deepening_search(gs, max_time=think_time, start_depth=4, progress=_progress, top_k=5)
            except AnalysisWorker.Cancelled:
                # cancelled: emit a notice and continue
                try:
                    self._out_q.put_nowait({'type': 'cancelled'})
                except Exception:
                    pass
                continue
            except Exception as e:
                try:
                    self._out_q.put_nowait({'type': 'error', 'error': str(e)})
                except Exception:
                    pass
                continue

            elapsed = max(0.0, time.time() - start_ts)
            try:
                self._out_q.put_nowait({'type': 'result', 'move': move, 'val': val, 'nodes': nodes, 'depth': depth, 'elapsed': elapsed, 'cfg': {'my_color': my_color, 'want_click': want_click}})
            except Exception:
                pass


def make_ai_move(page, html: str, my_color: str, think_time: float = THINK_TIME, mode: str = 'seq', workers: int = 4, show_vectors: bool = True) -> bool:
    gs = load_state_from_chesscom_html(html, turn=my_color)
    start_ts = time.time()
    # 콘솔 진행 로그
    class _Cancelled(Exception):
        pass

    # cache last seen best to show on UI even if current event lacks best_move
    _last_best = { 'move': None, 'val': None }

    def _progress(evt: dict):
        try:
            if _overlay_is_halt(page):
                _force_halt_reset(page)
                raise _Cancelled()
            depth = evt.get('depth')
            elapsed = float(evt.get('elapsed') or 0.0)
            nodes_total = int(evt.get('nodes_total') or 0)
            nps = (nodes_total/elapsed) if elapsed > 0 else 0.0
            best_move = evt.get('best_move')
            best_val = evt.get('best_val')
            # Display eval from AI perspective (white+: advantage for AI)
            try:
                if isinstance(best_val, (int, float)) and my_color == 'b':
                    best_val = -float(best_val)
                # Avoid showing bound-like huge numbers (e.g., 190000+)
                if isinstance(best_val, (int, float)) and abs(best_val) >= 190000:
                    # fallback to last cached value if any
                    lv = _last_best.get('val')
                    if isinstance(lv, (int, float)):
                        best_val = lv
            except Exception:
                pass
            remaining = evt.get('remaining')
            try:
                remaining = float(remaining) if remaining is not None else None
            except Exception:
                remaining = None
            used = max(0.0, min(think_time, (think_time - remaining) if remaining is not None else elapsed))
            pct = (used/think_time*100.0) if think_time > 0 else 0.0

            cfg_label = f"mode={mode} | time={think_time:.1f}s" + (f" | workers={workers}" if (mode or '').lower().startswith('mp') else '')
            parts = [f"depth={depth}", f"elapsed={elapsed:.2f}s", f"nodes={nodes_total}", f"nps={nps:.0f}"]
            lines = [f"<div><b>AI</b> <span style='opacity:.85'>{cfg_label}</span></div>", f"<div>{' | '.join(parts)}</div>"]
            if best_move:
                # update cache
                try:
                    _last_best['move'] = best_move
                    _last_best['val'] = best_val
                except Exception:
                    pass
                fr_ui = _rc_to_ui_algebraic(best_move[0])
                to_ui = _rc_to_ui_algebraic(best_move[1])
                lines.append(f"<div>pv={fr_ui}->{to_ui}</div>")
                if best_val is not None:
                    lines.append(f"<div>eval={best_val}</div>")
            else:
                # fallback display using cached best
                lb = _last_best.get('move')
                lb_val = _last_best.get('val')
                try:
                    if lb:
                        fr_ui = _rc_to_ui_algebraic(lb[0])
                        to_ui = _rc_to_ui_algebraic(lb[1])
                        lines.append(f"<div>pv={fr_ui}->{to_ui}</div>")
                        if isinstance(lb_val, (int, float)):
                            lines.append(f"<div>eval={lb_val}</div>")
                except Exception:
                    pass
            if think_time and think_time > 0:
                bar = ("<div style='margin-top:6px;height:8px;background:#333;border-radius:4px;overflow:hidden;'>"
                       f"<div style='height:100%;width:{pct:.1f}%;background:#4CAF50;'></div></div>")
                lines.append(bar)
            _overlay_set(page, ''.join(lines))

            if show_vectors:
                tm = evt.get('top_moves')
                if isinstance(tm, list) and len(tm) > 0:
                    # Map values to AI perspective for labels
                    mapped = []
                    for item in tm:
                        try:
                            v = item.get('val')
                            if isinstance(v, (int, float)) and my_color == 'b':
                                v = -float(v)
                            mapped.append({ 'from': item.get('from'), 'to': item.get('to'), 'val': v })
                        except Exception:
                            mapped.append(item)
                    _vector_set(page, _build_arrows_from_top_moves(mapped))
                else:
                    # fallback: draw cached best
                    lb = _last_best.get('move')
                    lb_val = _last_best.get('val')
                    if lb:
                        try:
                            v = lb_val
                            if isinstance(v, (int, float)) and my_color == 'b':
                                v = -float(v)
                            mapped = [{ 'from': lb[0], 'to': lb[1], 'val': v }]
                            _vector_set(page, _build_arrows_from_top_moves(mapped))
                        except Exception:
                            pass
        except _Cancelled:
            raise
        except Exception:
            pass

    # Early halt check before any heavy work
    if _overlay_is_halt(page):
        _force_halt_reset(page)
        return False

    if (mode or '').lower().startswith('mp'):
        try:
            def _mp_progress(evt: dict):
                try:
                    _progress(evt)
                except _Cancelled:
                    raise
                except Exception:
                    pass
            move, val, nodes, depth = get_multiprocess_move(gs, think_time, max(1, int(workers or 1)), progress=_mp_progress, top_k=5)
        except _Cancelled:
            print("[AI] 취소됨(Halt)")
            _force_halt_reset(page)
            try:
                if show_vectors:
                    _vector_clear(page)
            except Exception:
                pass
            return False
        except Exception:
            try:
                val, move, nodes, depth = iterative_deepening_search(gs, max_time=think_time, start_depth=4, progress=_progress, top_k=5)
            except _Cancelled:
                print("[AI] 취소됨(Halt)")
                _force_halt_reset(page)
                try:
                    if show_vectors:
                        _vector_clear(page)
                except Exception:
                    pass
                return False
    else:
        try:
            val, move, nodes, depth = iterative_deepening_search(gs, max_time=think_time, start_depth=4, progress=_progress, top_k=5)
        except _Cancelled:
            print("[AI] 취소됨(Halt)")
            _force_halt_reset(page)
            try:
                if show_vectors:
                    _vector_clear(page)
            except Exception:
                pass
            return False

    elapsed = max(0.0, time.time() - start_ts)
    if not move:
        print(f"[AI] done | depth={depth} elapsed={elapsed:.2f}s | no legal move")
        try:
            nps_done = (nodes/elapsed) if elapsed > 0 else 0.0
            final_lines = [
                "<div><b>AI</b> <span style='opacity:.85'>done</span></div>",
                f"<div>depth={depth} | elapsed={elapsed:.2f}s | nodes={nodes} | nps={nps_done:.0f}</div>",
                '<div>no legal move (checkmate/stalemate)</div>'
            ]
            _overlay_set(page, ''.join(final_lines))
            if show_vectors:
                _vector_clear(page)
        except Exception:
            pass
        return False

    if isinstance(move, (list, tuple)):
        if len(move) >= 2:
            fr, to = move[0], move[1]
        else:
            raise ValueError(f"Unexpected move tuple length: {len(move)}")
    else:
        raise ValueError("Move must be a tuple or list")

    eng_from, eng_to = rc_to_algebraic(fr), rc_to_algebraic(to)
    from_sq, to_sq = _rc_to_ui_algebraic(fr), _rc_to_ui_algebraic(to)
    # display eval from AI perspective
    disp_val = -val if my_color == 'b' and isinstance(val, (int, float)) else val
    print(f"AI Move: engine {eng_from}->{eng_to} | ui {from_sq}->{to_sq} (val={disp_val}, nodes={nodes}, depth={depth})")
    try:
        nps_done = (nodes/elapsed) if elapsed > 0 else 0.0
        final_lines = [
            "<div><b>AI</b> <span style='opacity:.85'>done</span></div>",
            f"<div>depth={depth} | elapsed={elapsed:.2f}s | nodes={nodes} | nps={nps_done:.0f}</div>",
            f"<div>pv={from_sq}->{to_sq}</div>",
            f"<div>eval={disp_val}</div>",
        ]
        _overlay_set(page, ''.join(final_lines))
    except Exception:
        pass
    try:
        # If halted before clicking, abort
        if _overlay_is_halt(page):
            _force_halt_reset(page)
            raise _Cancelled()
        _pl_click_move(page, from_sq, to_sq, my_color, timeout_ms=3500)
    except PWTimeout as e:
        # Not using hint-based clicks anymore; this reflects a generic move click timeout
        print(f"[ui] move click timeout: {e}")
        return False
    except _Cancelled:
        print("[ui] 취소됨(Halt)")
        _force_halt_reset(page)
        try:
            if show_vectors:
                _vector_clear(page)
        except Exception:
            pass
        return False
    except Exception as e:
        print(f"[ui] move click failed: {e}")
        return False
    return True


def _get_current_html(page) -> str:
    return page.content()


def _navigate_with_fallback(page, url: str, alt_url: Optional[str] = None) -> None:
    """Navigate to url with tolerant behavior: use 'commit' to avoid hard timeouts,
    wait for domcontentloaded separately, and optionally try an alternate URL.
    """
    alt = alt_url or url.replace('/ko/', '/')
    tried_alt = False
    for attempt in range(2):
        try:
            page.goto(url if not tried_alt else alt, wait_until='commit', timeout=15000)
        except Exception:
            # swallow and try to proceed to load state
            pass
        # Try to reach DOMContentLoaded; if it fails, maybe try alt next
        try:
            page.wait_for_load_state('domcontentloaded', timeout=20000)
            return
        except Exception:
            if tried_alt:
                break
            tried_alt = True
            continue
    # Final attempt to at least get something loaded
    try:
        page.wait_for_load_state('load', timeout=10000)
    except Exception:
        pass


def _wait_for_board_change(page, last_fen: str, my_color: str, timeout_s: float = 120.0, poll_s: float = 0.5) -> Tuple[str, str]:
    """Wait for a stable board change. Expect an opponent move:
    - Opponent piece-count should remain the same (or increase only in promotions, which we'll treat as same for count),
    - My piece-count should be either same (quiet) or -1 (if my piece was captured).
    The new FEN must be different and stable across two samples.
    """
    def _cnt_side(f: str, is_white: bool) -> int:
        return sum(1 for ch in f if (ch.isupper() if is_white else ch.islower()))

    opp_is_white = (str(my_color).lower().startswith('b'))
    me_is_white = not opp_is_white
    last_cnt_opp = _cnt_side(last_fen, opp_is_white)
    last_cnt_me = _cnt_side(last_fen, me_is_white)
    deadline = time.time() + timeout_s
    last_stable = None  # track last stable pair for fallback
    stable_repeats = 0
    while time.time() < deadline:
        if _overlay_is_halt(page):
            _force_halt_reset(page)
            raise TimeoutError("halted")
        html1 = _get_current_html(page)
        fen1 = parse_chesscom_html_to_fen(html1)
        if fen1 != last_fen:
            # debounce/stability check
            time.sleep(min(0.2, max(0.1, poll_s/2)))
            html2 = _get_current_html(page)
            fen2 = parse_chesscom_html_to_fen(html2)
            if fen2 == fen1:
                # track stability
                if last_stable == fen2:
                    stable_repeats += 1
                else:
                    last_stable = fen2
                    stable_repeats = 1
                try:
                    cnt2_opp = _cnt_side(fen2, opp_is_white)
                    cnt2_me = _cnt_side(fen2, me_is_white)
                    ok_counts = (cnt2_opp == last_cnt_opp) and (cnt2_me in (last_cnt_me, last_cnt_me - 1))
                    # Very conservative check: ensure the board diff looks like an opponent move
                    ok_plausible = _plausible_opponent_move(last_fen, fen2, my_color)
                    if ok_counts and ok_plausible:
                        # micro grace: absorb rapid consecutive updates (animations/instant reply)
                        micro_deadline = time.time() + min(0.25, max(0.1, poll_s))
                        latest_html, latest_fen = html2, fen2
                        while time.time() < micro_deadline:
                            html3 = _get_current_html(page)
                            fen3 = parse_chesscom_html_to_fen(html3)
                            if fen3 != latest_fen:
                                # restart confirmation for the new change
                                time.sleep(min(0.2, max(0.08, poll_s/2)))
                                html4 = _get_current_html(page)
                                fen4 = parse_chesscom_html_to_fen(html4)
                                if fen4 == fen3:
                                    latest_html, latest_fen = html4, fen4
                                    continue
                            time.sleep(min(0.12, max(0.06, poll_s/3)))
                        return latest_html, latest_fen
                    # strong stability fallback: if counts rule fails but we've seen >=3 repeats, accept
                    if stable_repeats >= 3 and ok_plausible:
                        return html2, fen2
                except Exception:
                    # If counting fails, still accept stable change
                    return html2, fen2
        time.sleep(poll_s)
    raise TimeoutError("board change wait timeout")


def _get_cfg(page) -> dict:
    try:
        return page.evaluate("()=> (window.injected_overlayCfgGet && window.injected_overlayCfgGet()) || null") or {}
    except Exception:
        return {}


def _clear_think_once(page) -> None:
    try:
        page.evaluate("()=>{ if(window.injected_overlayCfgClearThinkOnce){ window.injected_overlayCfgClearThinkOnce(); } }")
    except Exception:
        pass


def _is_page_alive(page) -> bool:
    try:
        # Use Playwright's is_closed() which remains reliable across reloads
        return not page.is_closed()
    except Exception:
        # On transient errors (e.g., during reload), assume alive to avoid premature exit
        return True


def _get_overlay_boot_state(page) -> Tuple[Optional[int], Optional[int]]:
    try:
        return tuple(page.evaluate("()=>[window.__overlayBootTs||null, window.__overlayBootCounter||null]"))  # type: ignore
    except Exception:
        return (None, None)


def auto_play_loop(page, my_color: str, think_time: float = THINK_TIME, mode: str = 'seq', workers: int = 4, max_plies: Optional[int] = None):
    """메인 루프: DOM을 주기적으로 읽어 오버레이/UI를 갱신하고,
    분석은 백그라운드 태스크(AnalysisWorker)가 진행. 내 수 반영 감지는 수행하지 않음."""
    html = _get_current_html(page)
    last_fen = parse_chesscom_html_to_fen(html)
    empty_fen = '8/8/8/8/8/8/8/8'
    seen_non_empty = (last_fen != empty_fen)
    print("[auto] 시작 FEN:", last_fen)
    print("[auto] 대기모드 진입")
    # 새로고침 감지용 부팅 상태 스냅샷과 초기화
    boot_ts, boot_cnt = _get_overlay_boot_state(page)
    boot_mark = (boot_ts, boot_cnt)
    # 초기 상태: Auto Detect OFF (배지/설정 일괄 초기화)
    _overlay_init_auto(page)
    reload_switch_done = False
    poll_s_idle = 0.3
    game_over_halt_done = False
    # start background analysis worker
    worker = AnalysisWorker()
    worker.start()
    out_q = worker.out_queue
    active_job = False
    active_want_click = False
    while True:
        # 브라우저/페이지 종료 감지 시 루프 종료 -> 프로그램 자동 종료
        if not _is_page_alive(page):
            print("[auto] 페이지가 닫혔습니다. 종료합니다.")
            try:
                worker.stop()
            except Exception:
                pass
            return

        # 게임 종료 모달 감지 시 자동 Halt
        try:
            if not game_over_halt_done and _is_game_over_modal_open(page):
                game_over_halt_done = True
                print("[auto] 게임 종료 감지됨: Halt")
                _overlay_set_halt(page)
                try:
                    _vector_clear(page)
                except Exception:
                    pass
        except Exception:
            pass

        # 오버레이 재부팅(새로고침) 감지 시, 동일 init 경로로 OFF 초기화
        try:
            cur_ts, cur_cnt = _get_overlay_boot_state(page)
            if not reload_switch_done and (cur_ts, cur_cnt) != boot_mark and (cur_ts or cur_cnt):
                reload_switch_done = True
                print("[auto] 새로고침 감지됨: 이후 FEN 자동 감지 비활성화(대기 모드 유지)")
                _overlay_init_auto(page)
        except Exception:
            pass
        cfg = _get_cfg(page) or {}
        cur_time = float(cfg.get('thinkTime') or think_time or THINK_TIME)
        mode = str(cfg.get('mode') or 'seq')
        workers = int(cfg.get('workers') or 4)
        show_vectors = bool(cfg.get('showVectors') if cfg.get('showVectors') is not None else True)
        show_pieces = bool(cfg.get('showPieces') if cfg.get('showPieces') is not None else False)
        think_once = bool(cfg.get('thinkOnce') or False)
        auto_detect_on = bool(cfg.get('autoDetect') or False)
        # 루프마다 myColor 최신값 사용
        my_color = 'b' if str(cfg.get('myColor') or '').lower().startswith('b') else 'w'

        # Immediate halt handling:
        # - If user requested run (thinkOnce) or enabled auto-detect, clear halt and proceed
        # - Otherwise, keep halted and idle
        if _overlay_is_halt(page):
            if think_once or auto_detect_on:
                _overlay_clear_halt_keep_cfg(page)
            else:
                _force_halt_reset(page)
                time.sleep(poll_s_idle)
                continue

        if think_once and not active_job:
            _clear_think_once(page)
            if _overlay_is_halt(page):
                _overlay_clear_halt_keep_cfg(page)
            html_now = _get_current_html(page)
            # showPieces 갱신은 AI 입력 시점에만 수행
            try:
                if show_pieces:
                    _squares_set(page, parse_chesscom_html_to_piece_markers(html_now))
            except Exception:
                pass
            worker.submit(html_now, my_color, cur_time, mode, workers, want_click=True)
            active_job = True
            active_want_click = True
            # 진행은 out_queue 이벤트로 처리
            # 계속 루프 진행

        try:
            html_now = _get_current_html(page)
            fen_now = parse_chesscom_html_to_fen(html_now)
            if fen_now != last_fen:
                # 초기 빈판->비어있지 않음: 베이스라인만 갱신
                if not seen_non_empty and fen_now != empty_fen:
                    last_fen = fen_now
                    seen_non_empty = True
                    continue
                # 디바운스 후 안정화 확인
                time.sleep(0.12)
                html_conf = _get_current_html(page)
                fen_conf = parse_chesscom_html_to_fen(html_conf)
                if fen_conf != fen_now:
                    # 일시적 변화(드래그 등) 무시
                    continue
                # 상대/내 변화 판별
                is_opp = _plausible_opponent_move(last_fen, fen_conf, my_color)
                # 내 변화는 색 반전하여 판정
                my_color_rev = 'b' if str(my_color).lower().startswith('w') else 'w'
                is_me = _plausible_opponent_move(last_fen, fen_conf, my_color_rev)
                # 베이스라인은 안정화된 판으로 갱신
                last_fen = fen_conf
                seen_non_empty = (last_fen != empty_fen)
                if is_opp:
                    print("[auto] 상대 수 감지. FEN:", last_fen)
                    if auto_detect_on:
                        if _overlay_is_halt(page):
                            _overlay_clear_halt_keep_cfg(page)
                        if not active_job:
                            # showPieces 갱신은 AI 입력 시점에만 수행
                            try:
                                if show_pieces:
                                    _squares_set(page, parse_chesscom_html_to_piece_markers(html_conf))
                            except Exception:
                                pass
                            worker.submit(html_conf, my_color, cur_time, mode, workers, want_click=True)
                            active_job = True
                            active_want_click = True
                else:
                    # 내 수 또는 기타 변화: 작업 제출하지 않음
                    pass
        except Exception:
            pass

        # Consume analysis events and update UI / click if done
        try:
            while True:
                ev = out_q.get_nowait()
                et = ev.get('type')
                if et == 'progress':
                    e = ev.get('evt') or {}
                    cfg_view = ev.get('cfg') or {}
                    # Build overlay lines similar to previous implementation
                    try:
                        depth = e.get('depth'); elapsed = float(e.get('elapsed') or 0.0)
                        nodes_total = int(e.get('nodes_total') or 0)
                        nps = (nodes_total/elapsed) if elapsed>0 else 0.0
                        best_move = e.get('best_move'); best_val = e.get('best_val')
                        tm = e.get('top_moves')
                        think_t = float(cfg_view.get('think_time') or THINK_TIME)
                        mode_v = str(cfg_view.get('mode') or 'seq')
                        workers_v = int(cfg_view.get('workers') or 4)
                        myc = str(cfg_view.get('my_color') or my_color)
                        used = min(think_t, elapsed); pct = (used/think_t*100.0) if think_t>0 else 0.0
                        cfg_label = f"mode={mode_v} | time={think_t:.1f}s" + (f" | workers={workers_v}" if mode_v.lower().startswith('mp') else '')
                        parts = [f"depth={depth}", f"elapsed={elapsed:.2f}s", f"nodes={nodes_total}", f"nps={nps:.0f}"]
                        lines = [f"<div><b>AI</b> <span style='opacity:.85'>{cfg_label}</span></div>", f"<div>{' | '.join(parts)}</div>"]
                        if best_move:
                            fr_ui = _rc_to_ui_algebraic(best_move[0]); to_ui = _rc_to_ui_algebraic(best_move[1])
                            lines.append(f"<div>pv={fr_ui}->{to_ui}</div>")
                            if best_val is not None:
                                lines.append(f"<div>eval={best_val}</div>")
                        if think_t and think_t>0:
                            bar = ("<div style='margin-top:6px;height:8px;background:#333;border-radius:4px;overflow:hidden;'>"
                                   f"<div style='height:100%;width:{pct:.1f}%;background:#4CAF50;'></div></div>")
                            lines.append(bar)
                        _overlay_set(page, ''.join(lines))
                        if show_vectors:
                            _vector_set(page, _build_arrows_from_top_moves(tm or []))
                    except Exception:
                        pass
                elif et == 'result':
                    active_job = False
                    res_move = ev.get('move'); val = ev.get('val'); depth = ev.get('depth'); nodes = ev.get('nodes'); elapsed = float(ev.get('elapsed') or 0.0)
                    cfg_res = ev.get('cfg') or {}
                    want_click = bool(cfg_res.get('want_click'))
                    # Final overlay
                    try:
                        nps_done = (nodes/elapsed) if elapsed>0 else 0.0
                        if res_move:
                            fr_ui = _rc_to_ui_algebraic(res_move[0]); to_ui = _rc_to_ui_algebraic(res_move[1])
                            disp_val = -val if (cfg_res.get('my_color') == 'b' and isinstance(val, (int, float))) else val
                            final_lines = [
                                "<div><b>AI</b> <span style='opacity:.85'>done</span></div>",
                                f"<div>depth={depth} | elapsed={elapsed:.2f}s | nodes={nodes} | nps={nps_done:.0f}</div>",
                                f"<div>pv={fr_ui}->{to_ui}</div>",
                                f"<div>eval={disp_val}</div>",
                            ]
                            _overlay_set(page, ''.join(final_lines))
                    except Exception:
                        pass
                    # Optional click without waiting for reflection
                    if res_move and want_click and not _overlay_is_halt(page):
                        try:
                            fr_ui = _rc_to_ui_algebraic(res_move[0]); to_ui = _rc_to_ui_algebraic(res_move[1])
                            _pl_click_move_simple(page, fr_ui, to_ui)
                        except Exception:
                            pass
                elif et == 'cancelled':
                    active_job = False
                elif et == 'error':
                    active_job = False
        except Exception:
            pass
        time.sleep(poll_s_idle)


def main():
    URL = "https://www.chess.com/ko/play/computer"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1000, "height": 720})
        try:
            # Register init assets at context-level before creating any pages
            _register_init_assets(context)
            page = context.new_page()
            # Robust navigation with fallback to non-locale URL
            _navigate_with_fallback(page, URL, alt_url="https://www.chess.com/play/computer")
            # Page loaded; optional manual ensure hook
            try:
                page.evaluate("()=>{ if(window.injected_overlayBootstrapEnsure){ window.injected_overlayBootstrapEnsure(); } }")
            except Exception:
                pass
            # Try to wait for the board element, but don't crash if missing
            try:
                page.wait_for_selector('wc-chess-board', timeout=30000)
            except Exception:
                pass


            # JS에서 THINK_TIME을 요청할 수 있도록 expose_function 등록
            async def get_think_time():
                return THINK_TIME
            page.expose_function("getThinkTimeFromPython", get_think_time)

            # 오버레이/UI 설치 및 초기화(대기 모드 False)
            _ensure_config_ui(page, default_time=THINK_TIME)
            _overlay_init_auto(page)

            # 초기 상태 출력
            html = page.content()
            fen = parse_chesscom_html_to_fen(html)
            my_color = 'w'
            print('FEN(piece placement):', fen)
            # showPieces 갱신은 AI 입력 시에만 수행

            # 대기/자동응답 루프 (브라우저 닫히면 자동 종료)
            auto_play_loop(page, my_color, think_time=THINK_TIME, mode='seq', workers=4, max_plies=None)
        finally:
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()
