from __future__ import annotations

import time
import re
from pathlib import Path
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


def _inject_assets(page, ensure_current: bool = False) -> None:
    # Deprecated: init scripts are now registered at context-level via _register_init_assets
    return None


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


def _overlay_clear_halt_and_auto_off(page) -> None:
    try:
        page.evaluate("()=>{ if(window.injected_overlayCfgSet){ window.injected_overlayCfgSet({ halt: false, autoDetect: false, thinkOnce: false }); if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge(); } }")
    except Exception:
        pass

def _overlay_set_halt(page) -> None:
    """오버레이 Halt를 강제로 켭니다."""
    try:
        page.evaluate("()=>{ if(window.injected_overlayCfgSet){ window.injected_overlayCfgSet({ halt: true, autoDetect: false, thinkOnce: false }); if(window.injected_overlayUpdateAutoBadge) window.injected_overlayUpdateAutoBadge(); } }")
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


def _pl_click_move(page, from_sq: str, to_sq: str, timeout_ms: int = 6000) -> None:
    """Click a move by selecting a piece and then clicking the destination square directly.
    After clicks, verify the move applied by checking DOM changes (origin piece removed or destination occupied).
    """
    from_xy = _sq_to_xy(from_sq)
    to_xy = _sq_to_xy(to_sq)

    board = page.locator('wc-chess-board')

    for attempt in range(3):
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

    # 3) Select Move Target
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

    # 4) Verify applied: origin piece removed or destination occupied
    def _dest_or_origin_changed() -> bool:
        return page.locator(f'wc-chess-board >>> .piece.square-{to_xy}').count()

    deadline = time.time() + max(0.5, timeout_ms/1000.0)
    while time.time() < deadline:
        if _dest_or_origin_changed():
            return
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
            _overlay_clear_halt_and_auto_off(page)
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
                _overlay_clear_halt_and_auto_off(page)
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
            _overlay_clear_halt_and_auto_off(page)
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
            raise _Cancelled()
        _pl_click_move(page, from_sq, to_sq, timeout_ms=3500)
    except PWTimeout as e:
        # Not using hint-based clicks anymore; this reflects a generic move click timeout
        print(f"[ui] move click timeout: {e}")
        return False
    except _Cancelled:
        print("[ui] 취소됨(Halt)")
        _overlay_clear_halt_and_auto_off(page)
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


# _safe_goto removed (unused)


def _wait_for_board_change(page, last_fen: str, timeout_s: float = 120.0, poll_s: float = 0.5) -> Tuple[str, str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        html = _get_current_html(page)
        fen = parse_chesscom_html_to_fen(html)
        if fen != last_fen:
            return html, fen
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
    """무한 대기 루프: 상대 수(FEN 변화) 감지 시 자동 한 수 응답, Run AI 클릭 시 즉시 한 수 실행."""
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
    while True:
        # 브라우저/페이지 종료 감지 시 루프 종료 -> 프로그램 자동 종료
        if not _is_page_alive(page):
            print("[auto] 페이지가 닫혔습니다. 종료합니다.")
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
        think_once = bool(cfg.get('thinkOnce') or False)
        auto_detect_on = bool(cfg.get('autoDetect') or False)
        # 루프마다 myColor 최신값 사용
        my_color = 'b' if str(cfg.get('myColor') or '').lower().startswith('b') else 'w'

        if think_once:
            _clear_think_once(page)
            html = _get_current_html(page)
            ok = make_ai_move(page, html, my_color, think_time=cur_time, mode=mode, workers=workers, show_vectors=show_vectors)
            if ok:
                try:
                    html, last_fen = _wait_for_board_change(page, last_fen, timeout_s=20.0, poll_s=0.2)
                    print("[auto] 내 수 반영됨. FEN:", last_fen)
                    seen_non_empty = (last_fen != empty_fen)
                except TimeoutError:
                    print("[auto][경고] 내 수 반영 감지 실패(타임아웃). 계속 대기모드 유지.")
                    last_fen = parse_chesscom_html_to_fen(_get_current_html(page))
                    seen_non_empty = (last_fen != empty_fen)
            else:
                # 문제로 인해 대기 전환: Auto Detect OFF 초기화
                _overlay_init_auto(page)
            continue

        try:
            html_now = _get_current_html(page)
            fen_now = parse_chesscom_html_to_fen(html_now)
            if fen_now != last_fen:
                if not seen_non_empty and fen_now != empty_fen:
                    last_fen = fen_now
                    seen_non_empty = True
                    continue
                # 자동 감지 비활성화 상태면 FEN만 갱신하고 넘어감
                if not auto_detect_on:
                    last_fen = fen_now
                    continue
                print("[auto] 상대 수 감지. FEN:", fen_now)
                last_fen = fen_now
                time.sleep(0.1)
                ok = make_ai_move(page, html_now, my_color, think_time=cur_time, mode=mode, workers=workers, show_vectors=show_vectors)
                if ok:
                    try:
                        _, last_fen = _wait_for_board_change(page, last_fen, timeout_s=20.0, poll_s=0.2)
                        print("[auto] 응답 수 반영됨. FEN:", last_fen)
                        seen_non_empty = (last_fen != empty_fen)
                    except TimeoutError:
                        print("[auto][경고] 응답 수 반영 감지 실패(타임아웃). 계속 대기.")
                        last_fen = parse_chesscom_html_to_fen(_get_current_html(page))
                        seen_non_empty = (last_fen != empty_fen)
                else:
                    # 문제로 인해 대기 전환: Auto Detect OFF 초기화
                    _overlay_init_auto(page)
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
            page.goto(URL, wait_until='domcontentloaded')
            # Page loaded; optional manual ensure hook
            try:
                page.evaluate("()=>{ if(window.injected_overlayBootstrapEnsure){ window.injected_overlayBootstrapEnsure(); } }")
            except Exception:
                pass
            try:
                page.wait_for_selector('wc-chess-board', timeout=15000)
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
