import argparse
import time
from typing import Tuple, Optional

from alphabeta import GameState, fen_to_board, rc_to_algebraic, iterative_deepening_search
from simple_multiprocess import get_multiprocess_move

Move = Tuple[Tuple[int, int], Tuple[int, int], dict]


def seq_engine(gs: GameState, think_time: float):
    t0 = time.perf_counter()
    val, move, nodes, depth = iterative_deepening_search(gs, max_time=think_time, start_depth=4)
    elapsed = time.perf_counter() - t0
    return move, val, nodes, depth, elapsed


def mp_engine(gs: GameState, think_time: float, workers: int):
    t0 = time.perf_counter()
    move, val, nodes, depth = get_multiprocess_move(gs, max_time=think_time, max_workers=workers)
    elapsed = time.perf_counter() - t0
    return move, val, nodes, depth, elapsed


def move_to_str(m: Optional[Move]) -> str:
    if not m:
        return "-"
    fr, to = m[0], m[1]
    return f"{rc_to_algebraic(fr)}->{rc_to_algebraic(to)}"


def play_one_game(think_time: float, max_moves: int, workers: int, start_fen: str,
                  seq_color: str = 'w'):
    gs = GameState()
    gs.board = fen_to_board(start_fen)
    gs.turn = 'w'
    move_stack = []

    # 엔진 배정
    engines = {
        'w': ('SEQ' if seq_color == 'w' else 'MP'),
        'b': ('SEQ' if seq_color == 'b' else 'MP'),
    }

    plies = 0
    history = []
    forced_result = None
    forced_reason = None

    while plies < max_moves:
        side = gs.turn
        which = engines[side]
        if which == 'SEQ':
            move, val, nodes, depth, elapsed = seq_engine(gs, think_time)
        else:
            move, val, nodes, depth, elapsed = mp_engine(gs, think_time, workers)

        # 조기 승부판정: 킹 부재 신호값
        if abs(val) >= 100000:
            forced_result = '1-0' if val > 0 else '0-1'
            forced_reason = 'no-king'
            # 진행 라인 마무리 개행
            print()
            break

        if not move:
            # 게임 종료: 무브 없음
            break

        fr, to, meta = move[0], move[1], move[2]
        gs.apply_move(fr, to, meta, move_stack)
        history.append((side, which, move_to_str(move), val, nodes, depth, elapsed))

        # 진행 상황 출력 (덮어쓰기 갱신)
        status = (f"[{len(history)}/{max_moves}] "
                  f"{side.upper()}[{which}] {move_to_str(move)}  "
                  f"val={val:.2f} d={depth} t={elapsed:.2f}s n={nodes}")
        # 캐리지 리턴으로 같은 줄 덮어쓰기, 충분히 길게 공백 패딩
        print("\r" + status.ljust(120), end='', flush=True)

        # 종료 조건 체크
        if gs.is_checkmate():
            break
        if gs.is_stalemate():
            break

        plies += 1

    # 진행 출력 줄 마무리 개행
    print()

    # 결과 판정
    result = '1/2-1/2'
    reason = 'max-moves' if plies >= max_moves else 'no-move'
    if forced_result is not None:
        result = forced_result
        reason = forced_reason
    else:
        if gs.is_checkmate():
            # 현재 턴이 체크메이트를 당한 것인지 알기 위해 직전 사이드가 승자
            winner_side = 'b' if gs.turn == 'w' else 'w'
            if winner_side == 'w':
                result = '1-0'
            else:
                result = '0-1'
            reason = 'checkmate'
        elif gs.is_stalemate():
            result = '1/2-1/2'
            reason = 'stalemate'

    # 집계 통계
    stats = {
        'plies': len(history),
        'moves': len(history) // 2,
        'w_engine': engines['w'],
        'b_engine': engines['b'],
        'result': result,
        'reason': reason,
        'w_time': round(sum(h[6] for h in history if h[0] == 'w'), 3),
        'b_time': round(sum(h[6] for h in history if h[0] == 'b'), 3),
        'w_nodes': sum(h[4] for h in history if h[0] == 'w'),
        'b_nodes': sum(h[4] for h in history if h[0] == 'b'),
        'w_nps': 0,
        'b_nps': 0,
    }
    stats['w_nps'] = int(stats['w_nodes'] / stats['w_time']) if stats['w_time'] > 0 else 0
    stats['b_nps'] = int(stats['b_nodes'] / stats['b_time']) if stats['b_time'] > 0 else 0

    return stats, history


def main():
    ap = argparse.ArgumentParser(description='Sequential vs Multiprocessing engine arena')
    ap.add_argument('--games', type=int, default=2, help='Number of games (colors alternate)')
    ap.add_argument('--time', type=float, default=1.0, help='Think time per move (seconds)')
    ap.add_argument('--max-moves', type=int, default=120, help='Max half-moves (plies) before draw')
    ap.add_argument('--workers', type=int, default=4, help='Workers for multiprocessing engine')
    ap.add_argument('--fen', type=str, default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR',
                    help='Start position FEN (piece placement field only)')
    args = ap.parse_args()

    total = {'SEQ': 0, 'MP': 0, 'draw': 0}

    for gi in range(args.games):
        seq_color = 'w' if gi % 2 == 0 else 'b'
        print(f"\n=== Game {gi+1}/{args.games} | SEQ as {seq_color.upper()} ===")
        stats, history = play_one_game(args.time, args.max_moves, args.workers, args.fen, seq_color)

        # 결과 출력
        print(f"Result: {stats['result']} ({stats['reason']}) | W:{stats['w_engine']} vs B:{stats['b_engine']}")
        print(f"Moves: {stats['moves']} | W time {stats['w_time']}s N={stats['w_nodes']} NPS={stats['w_nps']} | "
              f"B time {stats['b_time']}s N={stats['b_nodes']} NPS={stats['b_nps']}")
        # 수순 로그: 앞 5개, 뒤 5개만 출력
        if len(history) <= 10:
            rng = enumerate(history, start=1)
            for idx, h in rng:
                side, eng, mstr, val, nodes, depth, elapsed = h
                print(f"{idx:02d}. {side.upper()}[{eng}] {mstr}  val={val:.2f} d={depth} t={elapsed:.2f}s n={nodes}")
        else:
            first = history[:5]
            last = history[-5:]
            for idx, h in enumerate(first, start=1):
                side, eng, mstr, val, nodes, depth, elapsed = h
                print(f"{idx:02d}. {side.upper()}[{eng}] {mstr}  val={val:.2f} d={depth} t={elapsed:.2f}s n={nodes}")
            print("...")
            start_idx = len(history) - len(last) + 1
            for off, h in enumerate(last):
                idx = start_idx + off
                side, eng, mstr, val, nodes, depth, elapsed = h
                print(f"{idx:02d}. {side.upper()}[{eng}] {mstr}  val={val:.2f} d={depth} t={elapsed:.2f}s n={nodes}")

        # 승패 집계
        if stats['result'] == '1-0':
            total['SEQ' if stats['w_engine'] == 'SEQ' else 'MP'] += 1
        elif stats['result'] == '0-1':
            total['SEQ' if stats['b_engine'] == 'SEQ' else 'MP'] += 1
        else:
            total['draw'] += 1

    print("\n=== Summary ===")
    print(f"SEQ wins: {total['SEQ']} | MP wins: {total['MP']} | Draws: {total['draw']}")


if __name__ == '__main__':
    # Windows 멀티프로세싱 호환을 위해 가드 필요
    main()
