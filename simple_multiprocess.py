from concurrent.futures import ProcessPoolExecutor, wait
import time
import math
from typing import List, Tuple, Optional
from alphabeta import GameState


def _evaluate_child_position(args):
    """Worker: evaluate a single root move in a separate process using fixed depth.

    Args:
        args: tuple(state_copy, move, target_depth)

    Returns:
        (move, val, nodes, depth_done)
    """
    state_copy, move, target_depth = args

    # Apply the root move on the copied state
    move_stack = []
    state_copy.apply_move(move[0], move[1], move[2], move_stack)

    # Fixed-depth alpha-beta from the child position
    # Import locally to avoid potential pickling issues
    from alphabeta import alpha_beta as _ab, MAX_PLY as _MAX_PLY
    tt = {}
    killers = [[None, None] for _ in range(_MAX_PLY)]
    hist = {}
    maximizing = (state_copy.turn == 'w')
    val, _bm, nodes = _ab(
        state_copy, max(0, target_depth), -math.inf, math.inf,
        maximizing, tt, move_stack, 0, 0, killers, hist
    )

    return move, val, nodes, target_depth

def _root_move_key(m):
    """간단한 루트 무브 정렬 키: 캡처/프로모션 우선, 그 외는 그대로."""
    meta = m[2] or {}
    is_cap = 1 if meta.get('capture') else 0
    is_prom = 1 if meta.get('promotion') else 0
    # 더 높은 점수 먼저 오도록 음수로
    return (-is_cap, -is_prom)

def get_multiprocess_move(state: GameState, max_time: float = 2.0, max_workers: Optional[int] = None,
                          progress=None, top_k: int = 5):
    """
    병렬 루트 라운드 탐색(부분완료 허용):
    - 깊이 d에서 루트 무브들을 동일 고정 깊이로 병렬 평가(child_depth=d-1)
    - 남은 시간 내 완료된 결과만 모아 사용(부분완료 시에도 선택)
    - 전체 라운드 완주 시 d를 1 증가
    """
    # If already inside a child process (e.g., AnalysisWorker), avoid creating
    # another process pool. Fall back to sequential IDS to keep progress flowing.
    try:
        import multiprocessing as _mp
        if getattr(_mp.current_process(), 'name', '') != 'MainProcess':
            from alphabeta import iterative_deepening_search as _ids
            val, move, nodes, depth = _ids(state, max_time=max_time, start_depth=4, progress=progress, top_k=top_k)
            return move, val, nodes, depth
    except Exception:
        pass
    moves = state.generate_all_moves()
    if not moves:
        return None, 0, 0, 0

    import multiprocessing
    if max_workers is None:
        cpu = multiprocessing.cpu_count() or 2
        max_workers = max(1, min(cpu - 1, 4))

    side = state.turn
    start_ts = time.time()
    deadline = start_ts + max_time

    # 설정: 시작 깊이 및 최대 라운드 상한(안전장치)
    if max_time <= 0.4:
        start_depth = 3
    elif max_time <= 1.0:
        start_depth = 4
    else:
        start_depth = 5
    max_depth_cap = 64

    last_full_results: Optional[List[Tuple[tuple, float, int, int]]] = None
    last_full_depth = 0
    last_partial_results: Optional[List[Tuple[tuple, float, int, int]]] = None
    last_partial_depth = 0
    nodes_total = 0
    last_top_moves: Optional[List[dict]] = None

    # helpers
    def _emit(payload: dict):
        if callable(progress):
            try:
                progress(payload)
            except Exception:
                pass

    def _pick_best(results, depth_used):
        best_move = None
        best_val = None
        best_signed = -math.inf
        for move, val, _nodes, _d in results:
            signed = val if side == 'w' else -val
            if best_val is None or signed > best_signed:
                best_val = val
                best_signed = signed
                best_move = move
        return best_move, best_val, nodes_total, depth_used

    def _top_from_results(results, k):
        sorted_list = sorted(
            results,
            key=lambda t: (t[1] if side == 'w' else -t[1]),
            reverse=True
        )[:k]
        return [
            {'from': m[0], 'to': m[1], 'val': v}
            for (m, v, _n, _dd) in sorted_list
        ]

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            depth = start_depth
            while depth <= max_depth_cap and time.time() < deadline:
                # child 고정 깊이는 루트에서 한 수 둔 다음이므로 depth-1
                child_depth = max(0, depth - 1)
                # 루트 무브 간단 정렬(캡처/프로모션 우선)
                ordered = sorted(moves, key=_root_move_key)
                # depth start
                now = time.time()
                _emit({
                    'event': 'depth_start',
                    'depth': depth,
                    'elapsed': now - start_ts,
                    'remaining': max(0.0, deadline - now),
                    'nodes_total': nodes_total,
                    'best_move': None,
                    'best_val': None,
                })

                futures = [executor.submit(_evaluate_child_position, (state, m, child_depth)) for m in ordered]

                round_results: List[Tuple[tuple, float, int, int]] = []
                best_so_far: Optional[Tuple[tuple, float]] = None

                # poll futures periodically for timely updates
                pending = set(futures)
                while pending and time.time() < deadline:
                    slice_deadline = min(0.2, max(0.0, deadline - time.time()))
                    if slice_deadline <= 0:
                        break
                    done, pending = wait(pending, timeout=slice_deadline)
                    # collect finished
                    for fut in done:
                        try:
                            move, val, nodes, d_done = fut.result()
                            nodes_total += int(nodes or 0)
                            round_results.append((move, val, nodes, d_done))
                            # update best
                            if best_so_far is None:
                                best_so_far = (move, val)
                            else:
                                cur = val if side == 'w' else -val
                                best = best_so_far[1] if side == 'w' else -best_so_far[1]
                                if cur > best:
                                    best_so_far = (move, val)
                        except Exception:
                            pass

                    # emit progress update every slice
                    now = time.time()
                    top_moves = _top_from_results(round_results, top_k)
                    _emit({
                        'event': 'update',
                        'depth': depth,
                        'elapsed': now - start_ts,
                        'remaining': max(0.0, deadline - now),
                        'nodes_total': nodes_total,
                        'nps': (nodes_total / max(1e-6, (now - start_ts))),
                        'best_move': best_so_far[0] if best_so_far else None,
                        'best_val': best_so_far[1] if best_so_far else None,
                        'top_moves': top_moves,
                    })

                # cancel remaining if any
                for fut in pending:
                    try:
                        fut.cancel()
                    except Exception:
                        pass

                if round_results and len(round_results) == len(ordered):
                    # 라운드 완주
                    last_full_results = round_results
                    last_full_depth = depth
                    now = time.time()
                    top_moves = _top_from_results(last_full_results, top_k)
                    last_top_moves = top_moves
                    # report depth_complete with best snapshot
                    bm, bv, _nn, _dd = _pick_best(last_full_results, last_full_depth)
                    _emit({
                        'event': 'depth_complete',
                        'depth': depth,
                        'elapsed': now - start_ts,
                        'remaining': max(0.0, deadline - now),
                        'nodes_total': nodes_total,
                        'best_move': bm,
                        'best_val': bv,
                        'top_moves': top_moves,
                    })
                    depth += 1
                elif round_results:
                    # 부분완주 → 기록 후 종료
                    last_partial_results = round_results
                    last_partial_depth = depth
                    now = time.time()
                    top_moves = _top_from_results(round_results, top_k)
                    last_top_moves = top_moves
                    bm, bv, _nn, _dd = _pick_best(round_results, last_partial_depth)
                    _emit({
                        'event': 'depth_complete',
                        'depth': depth,
                        'elapsed': now - start_ts,
                        'remaining': max(0.0, deadline - now),
                        'nodes_total': nodes_total,
                        'best_move': bm,
                        'best_val': bv,
                        'top_moves': top_moves,
                    })
                    break
                else:
                    # nothing harvested -> stop
                    now = time.time()
                    _emit({
                        'event': 'depth_complete',
                        'depth': depth,
                        'elapsed': now - start_ts,
                        'remaining': max(0.0, deadline - now),
                        'nodes_total': nodes_total,
                        'best_move': None,
                        'best_val': None,
                        'top_moves': [],
                    })
                    break

    except Exception:
        # 예외 시 아래 폴백 경로
        pass

    if last_full_results:
        best_move, best_val, nodes_out, depth_used = _pick_best(last_full_results, last_full_depth)
    elif last_partial_results:
        best_move, best_val, nodes_out, depth_used = _pick_best(last_partial_results, last_partial_depth)
    else:
        best_move, best_val, nodes_out, depth_used = None, None, nodes_total, 0

    # Emit a final event for UI consistency (ensure chosen best is present in top_moves)
    final_top = list(last_top_moves) if last_top_moves else []
    if best_move is not None:
        bm_from, bm_to = best_move[0], best_move[1]
        present = any((tm.get('from') == bm_from and tm.get('to') == bm_to) for tm in final_top)
        if not present:
            try:
                final_top.insert(0, {'from': bm_from, 'to': bm_to, 'val': float(best_val if best_val is not None else 0.0)})
            except Exception:
                final_top.insert(0, {'from': bm_from, 'to': bm_to, 'val': 0.0})
    if isinstance(top_k, int) and top_k > 0:
        final_top = final_top[:top_k]
    now = time.time()
    _emit({
        'event': 'final',
        'depth': depth_used,
        'elapsed': now - start_ts,
        'remaining': max(0.0, deadline - now),
        'nodes_total': nodes_total,
        'best_move': best_move,
        'best_val': best_val,
        'top_moves': final_top,
    })

    if best_move is not None:
        return best_move, best_val, nodes_out, depth_used

    # 백업: 순차 iterative deepening (안전한 폴백)
    try:
        from alphabeta import iterative_deepening_search
        val, move, nodes, depth = iterative_deepening_search(state, max_time=max_time, start_depth=4, top_k=top_k)
        return move, val, nodes, depth
    except Exception:
        return None, 0, 0, 0
