from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError, wait, FIRST_COMPLETED, ALL_COMPLETED
import time
import math
from alphabeta import GameState, PIECE_VALUES


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

def get_multiprocess_move(state: GameState, max_time: float = 2.0, max_workers: int = None,
                          progress=None, top_k: int = 5):
    """
    병렬 루트 라운드 탐색(부분완료 허용):
    - 깊이 d에서 루트 무브들을 동일 고정 깊이로 병렬 평가(child_depth=d-1)
    - 남은 시간 내 완료된 결과만 모아 사용(부분완료 시에도 선택)
    - 전체 라운드 완주 시 d를 1 증가
    """
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

    last_full_results = None  # List of tuples (move, val, nodes, depth)
    last_full_depth = 0
    last_partial_results = None
    last_partial_depth = 0
    nodes_total = 0
    last_top_moves = None

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            depth = start_depth
            while depth <= max_depth_cap and time.time() < deadline:
                # child 고정 깊이는 루트에서 한 수 둔 다음이므로 depth-1
                child_depth = max(0, depth - 1)
                # 루트 무브 간단 정렬(캡처/프로모션 우선)
                ordered = sorted(moves, key=_root_move_key)
                # 라운드 제출
                # depth 시작 알림
                if callable(progress):
                    now = time.time()
                    try:
                        progress({
                            'event': 'depth_start',
                            'depth': depth,
                            'elapsed': now - start_ts,
                            'remaining': max(0.0, deadline - now),
                            'nodes_total': nodes_total,
                            'best_move': None,
                            'best_val': None,
                        })
                    except Exception:
                        pass

                futures = [executor.submit(_evaluate_child_position, (state, m, child_depth)) for m in ordered]

                # 시간 내 as_completed로 수집(부분완료 허용)
                timeout = max(0.0, deadline - time.time())
                round_results = []
                best_so_far = None  # (move, val)
                if timeout > 0:
                    try:
                        for fut in as_completed(futures, timeout=timeout):
                            try:
                                move, val, nodes, d_done = fut.result()
                                nodes_total += int(nodes or 0)
                                round_results.append((move, val, nodes, d_done))
                                # 진행 업데이트 이벤트
                                if callable(progress):
                                    now = time.time()
                                    # 부호 보정 기준에 따라 현재 최선 추정 유지
                                    if best_so_far is None:
                                        best_so_far = (move, val)
                                    else:
                                        # 백/흑 시그널 비교
                                        cur = val if side == 'w' else -val
                                        best = best_so_far[1] if side == 'w' else -best_so_far[1]
                                        if cur > best:
                                            best_so_far = (move, val)
                                    # 상위 K 후보 계산 (부분완료 기준)
                                    topk = top_k
                                    sorted_partial = sorted(
                                        round_results,
                                        key=lambda t: (t[1] if side == 'w' else -t[1]),
                                        reverse=True
                                    )[:topk]
                                    top_moves = [
                                        {'from': m[0], 'to': m[1], 'val': v}
                                        for (m, v, _n, _dd) in sorted_partial
                                    ]
                                    try:
                                        progress({
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
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except TimeoutError:
                        pass
                # 남은 작업 취소
                for fut in futures:
                    if not fut.done():
                        fut.cancel()

                if round_results and len(round_results) == len(ordered):
                    # 라운드 완주
                    last_full_results = round_results
                    last_full_depth = depth
                    if callable(progress):
                        now = time.time()
                        # 완주 시 상위 후보 계산
                        topk = top_k
                        sorted_full = sorted(
                            last_full_results,
                            key=lambda t: (t[1] if side == 'w' else -t[1]),
                            reverse=True
                        )[:topk]
                        top_moves = [
                            {'from': m[0], 'to': m[1], 'val': v}
                            for (m, v, _n, _dd) in last_full_results
                            if (m, v) in [(sf[0], sf[1]) for sf in sorted_full]
                        ]
                        last_top_moves = top_moves
                        try:
                            progress({
                                'event': 'depth_complete',
                                'depth': depth,
                                'elapsed': now - start_ts,
                                'remaining': max(0.0, deadline - now),
                                'nodes_total': nodes_total,
                                'top_moves': top_moves,
                            })
                        except Exception:
                            pass
                    depth += 1
                elif round_results:
                    # 부분완주 → 기록 후 종료
                    last_partial_results = round_results
                    last_partial_depth = depth
                    if callable(progress):
                        now = time.time()
                        topk = top_k
                        sorted_partial = sorted(
                            round_results,
                            key=lambda t: (t[1] if side == 'w' else -t[1]),
                            reverse=True
                        )[:topk]
                        top_moves = [
                            {'from': m[0], 'to': m[1], 'val': v}
                            for (m, v, _n, _dd) in round_results
                            if (m, v) in [(sp[0], sp[1]) for sp in sorted_partial]
                        ]
                        last_top_moves = top_moves
                        try:
                            progress({
                                'event': 'depth_complete',
                                'depth': depth,
                                'elapsed': now - start_ts,
                                'remaining': max(0.0, deadline - now),
                                'nodes_total': nodes_total,
                                'top_moves': top_moves,
                            })
                        except Exception:
                            pass
                    break
                else:
                    # 수확 없음 → 종료
                    if callable(progress):
                        now = time.time()
                        try:
                            progress({
                                'event': 'depth_complete',
                                'depth': depth,
                                'elapsed': now - start_ts,
                                'remaining': max(0.0, deadline - now),
                                'nodes_total': nodes_total,
                            })
                        except Exception:
                            pass
                    break

    except Exception:
        # 예외 시 아래 폴백 경로
        pass

    def _pick_best(results, depth_used):
        # 마지막 완주 깊이에서 최선 수 선택
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

    if last_full_results:
        best_move, best_val, nodes_out, depth_used = _pick_best(last_full_results, last_full_depth)
    elif last_partial_results:
        best_move, best_val, nodes_out, depth_used = _pick_best(last_partial_results, last_partial_depth)
    else:
        best_move, best_val, nodes_out, depth_used = None, None, nodes_total, 0

    # Emit a final event for UI consistency (ensure chosen best is present in top_moves)
    if callable(progress):
        try:
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
            progress({
                'event': 'final',
                'depth': depth_used,
                'elapsed': now - start_ts,
                'remaining': max(0.0, deadline - now),
                'nodes_total': nodes_total,
                'best_move': best_move,
                'best_val': best_val,
                'top_moves': final_top,
            })
        except Exception:
            pass

    if best_move is not None:
        return best_move, best_val, nodes_out, depth_used

    # 백업: 순차 iterative deepening (안전한 폴백)
    try:
        from alphabeta import iterative_deepening_search
        val, move, nodes, depth = iterative_deepening_search(state, max_time=max_time, start_depth=4, top_k=top_k)
        return move, val, nodes, depth
    except Exception:
        return None, 0, 0, 0
