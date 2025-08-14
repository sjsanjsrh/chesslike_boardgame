from typing import List, Tuple, Dict, Optional, Callable
import math, time
import multiprocessing

# Updated values based on modern chess theory (Kaufman 2021 system scaled by 10)
# Standard ratios: P=1, N=3.2, B=3.3, R=5.3, Q=9.4
PIECE_VALUES = {'K':200,'Q':94,'R':53,'B':33,'N':32,'P':10}

# ---------------------
# Piece classes (standard pieces only)
# ---------------------
class Piece:
    def __init__(self, side: str, name: str):
        self.side = side  # 'w' or 'b'
        self.name = name  # 'K','Q','R','N','B','P'

    def __repr__(self):
        return f"{self.side}{self.name}"

    def get_moves(self, r: int, c: int, state) -> List[Tuple[int,int, dict]]:
        return []

def on_board(r,c):
    return 0 <= r < 8 and 0 <= c < 8

def generate_sliding_moves(r,c,state,deltas):
    moves=[]
    for dr,dc in deltas:
        nr,nc=r+dr,c+dc
        while on_board(nr,nc):
            t = state.board[nr][nc]
            if t is None:
                moves.append((nr,nc,{}))
            else:
                if t.side != state.turn:
                    moves.append((nr,nc,{'capture': True}))
                break
            nr+=dr; nc+=dc
    return moves

class Rook(Piece):
    def __init__(self, side): super().__init__(side,'R')
    def get_moves(self,r,c,state):
        return generate_sliding_moves(r,c,state,[(1,0),(-1,0),(0,1),(0,-1)])

class Bishop(Piece):
    def __init__(self, side): super().__init__(side,'B')
    def get_moves(self,r,c,state):
        return generate_sliding_moves(r,c,state,[(1,1),(1,-1),(-1,1),(-1,-1)])

class Queen(Piece):
    def __init__(self, side): super().__init__(side,'Q')
    def get_moves(self,r,c,state):
        return generate_sliding_moves(r,c,state,[(1,0),(-1,0),(0,1),(0,-1),
                                                 (1,1),(1,-1),(-1,1),(-1,-1)])

class Knight(Piece):
    def __init__(self, side): super().__init__(side,'N')
    def get_moves(self,r,c,state):
        moves=[]
        for dr,dc in [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]:
            nr,nc=r+dr,c+dc
            if on_board(nr,nc):
                t=state.board[nr][nc]
                if t is None or t.side!=self.side:
                    moves.append((nr,nc,{'capture': t is not None}))
        return moves

class King(Piece):
    def __init__(self, side): super().__init__(side,'K')
    def get_moves(self,r,c,state):
        moves=[]
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                nr,nc=r+dr,c+dc
                if on_board(nr,nc):
                    t=state.board[nr][nc]
                    if t is None or t.side!=self.side:
                        moves.append((nr,nc,{'capture': t is not None}))
        
        # Add castling moves
        castling_moves = self.get_castling_moves(r, c, state)
        moves.extend(castling_moves)
        
        return moves
    
    def get_castling_moves(self, r, c, state):
        moves = []
        
        # 킹이 초기 위치 4번 컬럼에 있는지만 체크 (랭크는 상관없이)
        if c != 4:
            return moves
        
        # Check if king has moved (simplified check based on position and castle_check)
        if not state.castle_check[self.side][0] and not state.castle_check[self.side][1]:
            return moves
            
        # Check kingside castling (short castling)
        if state.castle_check[self.side][1]:  # kingside castle allowed
            if self.can_castle_kingside(r, c, state):
                moves.append((r, c+2, {'castle': 'kingside'}))
        
        # Check queenside castling (long castling)
        if state.castle_check[self.side][0]:  # queenside castle allowed
            if self.can_castle_queenside(r, c, state):
                moves.append((r, c-2, {'castle': 'queenside'}))
        
        return moves
    
    def can_castle_kingside(self, r, c, state):
        # 킹의 현재 위치를 기준으로 상대적 위치 체크
        # Check if squares between king and rook are empty (킹 우측 1칸, 2칸)
        if c + 2 >= 8:  # 보드 범위 체크
            return False
        square1 = state.board[r][c+1]
        square2 = state.board[r][c+2]
        if square1 is not None or square2 is not None:
            return False
        # Check if rook is in correct position (킹 우측 3칸)
        if c + 3 >= 8:
            return False
        rook = state.board[r][c+3]
        if rook is None or rook.name != 'R' or rook.side != self.side:
            return False
        # 공격 중인지 확인(현재/통과/도착 칸)
        enemy = 'b' if self.side == 'w' else 'w'
        if (state.is_under_attack(r, c, enemy) or
            state.is_under_attack(r, c+1, enemy) or
            state.is_under_attack(r, c+2, enemy)):
            return False
        return True

    def can_castle_queenside(self, r, c, state):
        # 킹의 현재 위치를 기준으로 상대적 위치 체크
        if c - 3 < 0:
            return False
        square1 = state.board[r][c-1]
        square2 = state.board[r][c-2]
        square3 = state.board[r][c-3]
        if (square1 is not None or square2 is not None or square3 is not None):
            return False
        if c - 4 < 0:
            return False
        rook = state.board[r][c-4]
        if rook is None or rook.name != 'R' or rook.side != self.side:
            return False
        enemy = 'b' if self.side == 'w' else 'w'
        if (state.is_under_attack(r, c, enemy) or
            state.is_under_attack(r, c-1, enemy) or
            state.is_under_attack(r, c-2, enemy)):
            return False
        return True

class Pawn(Piece):
    def __init__(self, side): super().__init__(side,'P')
    def get_moves(self,r,c,state):
        moves=[]
        # White moves up the board (towards row decreasing), Black moves down (row increasing)
        dir = -1 if self.side == 'w' else 1
        nr, nc = r + dir, c

        # Promotion row (white at row 0, black at row 7)
        promotion_row = 0 if self.side == 'w' else 7

        # Forward moves
        if on_board(nr,nc) and state.board[nr][nc] is None:
            if nr == promotion_row:
                moves.append((nr,nc,{'promotion': True}))
            else:
                moves.append((nr,nc,{}))
            # Double push from starting rank
            start_row = 6 if self.side == 'w' else 1
            nr2 = r + 2*dir
            if r == start_row and on_board(nr2,nc) and state.board[nr2][nc] is None:
                moves.append((nr2,nc,{'double_pawn_push': True}))

        # Diagonal captures
        for dc in (-1, 1):
            nr, nc = r + dir, c + dc
            if on_board(nr,nc):
                t = state.board[nr][nc]
                if t is not None and t.side != self.side:
                    if nr == promotion_row:
                        moves.append((nr,nc, {'capture': True, 'promotion': True}))
                    else:
                        moves.append((nr,nc, {'capture': True}))

        # En passant captures
        en_passant_moves = self.get_en_passant_moves(r, c, state)
        moves.extend(en_passant_moves)

        return moves

    def get_en_passant_moves(self, r, c, state):
        moves = []
        
        if state.en_passant_target is None:
            return moves
        
        target_r, target_c = state.en_passant_target
        dir = -1 if self.side == 'w' else 1
        
        # Check if we can capture en passant
        # The target square should be one rank ahead of us and one file to the side
        if r + dir == target_r and abs(c - target_c) == 1:
            # The enemy pawn should be on the same rank as us, at the target column
            enemy_pawn = state.board[r][target_c]
            if (enemy_pawn is not None and enemy_pawn.name == 'P' and 
                enemy_pawn.side != self.side):
                moves.append((target_r, target_c, {'en_passant': True}))
        
        return moves

    def get_promotion_piece(self):
        # Pawn은 오직 Queen으로만 진급
        return Queen(self.side)

# ---------------------
# Game State
# ---------------------
# Castling rights bitmask
CR_WK, CR_WQ, CR_BK, CR_BQ = 1, 2, 4, 8

def parse_castling_rights(s: str) -> int:
    if not s or s == '-':
        return 0
    m = 0
    if 'K' in s: m |= CR_WK
    if 'Q' in s: m |= CR_WQ
    if 'k' in s: m |= CR_BK
    if 'q' in s: m |= CR_BQ
    return m

def emit_castling_rights(m: int) -> str:
    out = []
    if m & CR_WK: out.append('K')
    if m & CR_WQ: out.append('Q')
    if m & CR_BK: out.append('k')
    if m & CR_BQ: out.append('q')
    return ''.join(out) or '-'

class GameState:
    def __init__(self):
        self.board = [[None]*8 for _ in range(8)]
        self.turn = 'w'
        self.captured = {'w': [], 'b': []}
        self.move_count = 0
        self.castle_check={'w': [True,True], 'b': [True,True]}
        self.en_passant_target = None  # (row, col) of en passant target square
        
        # King position tracking for check detection
        self.w_king_pos = (7, 4)  # Initial position: e1
        self.b_king_pos = (0, 4)  # Initial position: e8
        
        # Castling rights tracking
        self.w_castling_rights = [True, True]  # [queenside, kingside]
        self.b_castling_rights = [True, True]  # [queenside, kingside]
        
        # Move counters for fifty-move rule and game notation
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        # KQkq 권리 통합 비트마스크 (기본: 모두 가능)
        self.castling_rights: int = (CR_WK | CR_WQ | CR_BK | CR_BQ)
        
        # Initialize the board with starting position
        self._initialize_board()
    
    def _initialize_board(self):
        """Set up the initial chess board position"""
        # b pieces (top of board, row 0-1)
        self.board[0] = [
            Rook('b'), Knight('b'), Bishop('b'), Queen('b'),
            King('b'), Bishop('b'), Knight('b'), Rook('b')
        ]
        self.board[1] = [Pawn('b') for _ in range(8)]
        
        # Empty squares (rows 2-5)
        for row in range(2, 6):
            self.board[row] = [None for _ in range(8)]
        
        # w pieces (bottom of board, row 6-7)
        self.board[6] = [Pawn('w') for _ in range(8)]
        self.board[7] = [
            Rook('w'), Knight('w'), Bishop('w'), Queen('w'),
            King('w'), Bishop('w'), Knight('w'), Rook('w')
        ]

    def apply_move(self, from_rc: Tuple[int,int], to_rc: Tuple[int,int], metadata: dict, move_stack: list):
        fr,fc = from_rc
        tr,tc = to_rc
        moving = self.board[fr][fc]
        captured = self.board[tr][tc]
        # Build undo record (use tuples for squares; include defaults expected by undo)
        undo_rec = {
            'fr': (fr, fc),
            'to': (tr, tc),
            'meta': (metadata.copy() if metadata else {}),
            'captured': captured,
            'promoted_from': None,
            'prev_castle_rights': { 'w': self.castle_check['w'][:], 'b': self.castle_check['b'][:] },
            'prev_en_passant_target': self.en_passant_target,
        }
        # Handle castling
        if metadata.get('castle'):
            castle_type = metadata['castle']
            if castle_type == 'kingside':
                # Move rook for kingside castling
                rook = self.board[fr][fc+3]
                self.board[fr][fc+1] = rook  # Rook moves to f1/f8
                self.board[fr][fc+3] = None
                undo_rec['castle_info'] = {'type': 'kingside', 'rook_from': (fr, fc+3), 'rook_to': (fr, fc+1)}
            elif castle_type == 'queenside':
                # Move rook for queenside castling
                rook = self.board[fr][fc-4]
                self.board[fr][fc-1] = rook  # Rook moves to d1/d8
                self.board[fr][fc-4] = None
                undo_rec['castle_info'] = {'type': 'queenside', 'rook_from': (fr, fc-4), 'rook_to': (fr, fc-1)}
        
        # Handle en passant
        if metadata.get('en_passant'):
            # Capture the pawn that was passed
            enemy_pawn_row = fr  # The enemy pawn is on the same rank as the moving pawn
            enemy_pawn_col = tc
            enemy_pawn = self.board[enemy_pawn_row][enemy_pawn_col]
            if enemy_pawn:
                self.captured[enemy_pawn.side].append(enemy_pawn)
                self.board[enemy_pawn_row][enemy_pawn_col] = None
                undo_rec['en_passant_captured'] = enemy_pawn
        
        # Reset en passant target (will be set again if needed)
        self.en_passant_target = None
        
        # Set en passant target for double pawn pushes
        if moving.name == 'P' and metadata.get('double_pawn_push'):
            # Set en passant target square behind the pawn
            ep_target_row = (fr + tr) // 2  # Middle square
            ep_target_col = fc
            self.en_passant_target = (ep_target_row, ep_target_col)
        
        # Update castle rights when king or rook moves
        if moving.name == 'K':
            # King moved - lose all castling rights
            self.castle_check[moving.side] = [False, False]
        elif moving.name == 'R':
            # Rook moved - lose castling rights for that side
            if fr == 0 or fr == 7:  # Starting rank for rook
                if fc == 0:  # Queenside rook
                    self.castle_check[moving.side][0] = False
                elif fc == 7:  # Kingside rook
                    self.castle_check[moving.side][1] = False
        
        # Capture rook affects castling rights
        if captured is not None:
            if captured.name == 'R':
                if tr == 0 or tr == 7:  # Starting rank
                    if tc == 0:  # Queenside rook captured
                        self.castle_check[captured.side][0] = False
                    elif tc == 7:  # Kingside rook captured
                        self.castle_check[captured.side][1] = False
            self.captured[captured.side].append(captured)
        
        self.board[tr][tc] = moving
        self.board[fr][fc] = None
        
        # Update king positions
        if moving.name == 'K':  # If a king moved
            if moving.side == 'w':
                undo_rec['prev_w_king_pos'] = self.w_king_pos
                self.w_king_pos = (tr, tc)
            else:
                undo_rec['prev_b_king_pos'] = self.b_king_pos
                self.b_king_pos = (tr, tc)
        
        # Handle pawn promotion
        if moving.name == 'P':
            promotion_row = 7 if moving.side == 'w' else 0
            if (tr == promotion_row) and metadata.get('promotion'):
                undo_rec['promoted_from'] = moving
                self.board[tr][tc] = moving.get_promotion_piece()
            else:
                # 기존 코드와 호환 (혹시 promotion meta가 없는 경우)
                if (moving.side == 'w' and tr == 0) or (moving.side == 'b' and tr == 7):
                    undo_rec['promoted_from'] = moving
                    self.board[tr][tc] = Queen(moving.side)

        move_stack.append(undo_rec)

        # Toggle turn and increment move count
        self.turn = 'b' if self.turn == 'w' else 'w'
        self.move_count += 1

    def undo_move(self, move_stack: list):
        if not move_stack:
            return
        u = move_stack.pop()
        # Restore castle rights
        if 'prev_castle_rights' in u:
            self.castle_check = u['prev_castle_rights']
        
        # Restore en passant target
        if 'prev_en_passant_target' in u:
            self.en_passant_target = u['prev_en_passant_target']
        
        # Restore king positions
        if 'prev_w_king_pos' in u:
            self.w_king_pos = u['prev_w_king_pos']
        if 'prev_b_king_pos' in u:
            self.b_king_pos = u['prev_b_king_pos']
        
        # Handle castling undo
        if u.get('castle_info'):
            castle_info = u['castle_info']
            # Move rook back to original position
            rook_from = castle_info['rook_from']
            rook_to = castle_info['rook_to']
            rook = self.board[rook_to[0]][rook_to[1]]
            self.board[rook_from[0]][rook_from[1]] = rook
            self.board[rook_to[0]][rook_to[1]] = None
        
        # Handle en passant undo
        if u.get('en_passant_captured'):
            # Restore the captured pawn
            enemy_pawn = u['en_passant_captured']
            # The enemy pawn was on the same rank as the from square
            enemy_pawn_row = u['fr'][0]
            enemy_pawn_col = u['to'][1]
            self.board[enemy_pawn_row][enemy_pawn_col] = enemy_pawn
            self.captured[enemy_pawn.side].pop()
        
        if u['promoted_from'] is not None:
            self.board[u['to'][0]][u['to'][1]] = u['promoted_from']
        moving = self.board[u['to'][0]][u['to'][1]]
        self.board[u['fr'][0]][u['fr'][1]] = moving
        self.board[u['to'][0]][u['to'][1]] = u['captured']
        if u['captured'] is not None:
            self.captured[u['captured'].side].pop()
        self.turn = 'b' if self.turn == 'w' else 'w'
        self.move_count -= 1


    def generate_all_moves(self, skip_castling: bool = False):
        moves=[]
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p is None:
                    continue
                
                if p.side != self.turn:
                    continue
                    
                if p.name == 'K' and skip_castling:
                    # Get king moves without castling
                    for dr in (-1,0,1):
                        for dc in (-1,0,1):
                            if dr==0 and dc==0: continue
                            nr,nc=r+dr,c+dc
                            if on_board(nr,nc):
                                t=self.board[nr][nc]
                                if t is None or (t.side != p.side):
                                    moves.append(((r,c),(nr,nc),{'capture': t is not None},p))
                else:
                    for nr,nc,meta in p.get_moves(r,c,self):
                        moves.append(((r,c),(nr,nc),meta,p))
        return moves

    def zobrist_hash(self) -> str:
        """Simple-but-safe hash for transposition table.
        Include side-to-move, piece placement, castling rights and en-passant target.
        (Full Zobrist not needed here; string key is sufficient for Python dict.)
        """
        rows = []
        for r in range(8):
            row = []
            for c in range(8):
                p = self.board[r][c]
                if p is None:
                    row.append('.')
                else:
                    row.append(f"{p.side}{p.name}")
            rows.append(''.join(row))

        # Encode castling rights from current rights tracker
        cr = []
        # order: KQkq (w kingside, w queenside, b kingside, b queenside)
        try:
            if self.castle_check['w'][1]: cr.append('K')
            if self.castle_check['w'][0]: cr.append('Q')
            if self.castle_check['b'][1]: cr.append('k')
            if self.castle_check['b'][0]: cr.append('q')
        except Exception:
            # Fallback: no rights known
            pass
        cr_s = ''.join(cr) or '-'

        # Encode en passant target square, if any
        ep = self.en_passant_target
        ep_s = f"{ep[0]},{ep[1]}" if isinstance(ep, tuple) else '-'

        return f"{self.turn}|{'|'.join(rows)}|CR:{cr_s}|EP:{ep_s}"
    
    def is_in_check(self, side: str = None, skip_castling: bool = False) -> bool:
        """Check if the given side is in check"""
        if side is None:
            side = self.turn
        
        # Find the king
        king_pos = None
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.name == 'K' and piece.side == side:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        
        if not king_pos:
            return False  # No king found
        
        # Check if any enemy piece can attack the king
        enemy_side = 'b' if side == 'w' else 'w'
        orig_turn = self.turn
        self.turn = enemy_side
        
        # Generate moves but skip castling to avoid recursion
        enemy_moves = self.generate_all_moves(skip_castling=True)
        self.turn = orig_turn
        
        for move in enemy_moves:
            if move[1] == king_pos:  # Enemy can capture king
                return True
        
        return False

    def gives_check(self, from_sq, to_sq, meta):
        move_stack = []
        self.apply_move(from_sq, to_sq, meta, move_stack)
        in_check = self.is_in_check(self.turn)  # 이동 후 턴은 상대방
        self.undo_move(move_stack)
        return in_check

    def is_under_attack(self, row: int, col: int, attacking_side: str) -> bool:
        """Check if a square is under attack by a given side"""
        orig_turn = self.turn
        self.turn = attacking_side
        
        # Generate moves for the attacking side
        moves = self.generate_all_moves(skip_castling=True)
        self.turn = orig_turn
        
        # Check if any move targets the given square
        for move in moves:
            if move[1] == (row, col):
                return True
        
        return False
    
    def is_checkmate(self) -> bool:
        """Check if current player is in checkmate"""
        if not self.is_in_check():
            return False  # Not in check, can't be checkmate
        
        # Try all possible moves to see if any gets out of check
        moves = self.generate_all_moves()
        move_stack = []
        
        for move in moves:
            # Try the move
            self.apply_move(move[0], move[1], move[2], move_stack)
            
            # Check if still in check
            enemy_side = 'b' if self.turn == 'w' else 'w'
            still_in_check = self.is_in_check(enemy_side)
            
            # Undo the move
            self.undo_move(move_stack)
            
            if not still_in_check:
                return False  # Found a move that gets out of check
        
        return True  # No move gets out of check = checkmate
    
    def is_stalemate(self) -> bool:
        """Check if current player is in stalemate"""
        if self.is_in_check():
            return False  # In check, can't be stalemate
        
        # Check if there are any legal moves
        moves = self.generate_all_moves()
        move_stack = []
        
        for move in moves:
            # Try the move
            self.apply_move(move[0], move[1], move[2], move_stack)
            
            # Check if this puts us in check (illegal move)
            in_check = self.is_in_check(move[3].side)
            
            # Undo the move
            self.undo_move(move_stack)
            
            if not in_check:
                return False  # Found a legal move
        
        return True  # No legal moves = stalemate
    def apply_null_move(self, move_stack):
        """
        Null move: 아무 수를 두지 않고 턴을 넘김.
        현재 상태를 move_stack에 저장해 두었다가 undo_null_move로 복원 가능.
        """
        # 현재 상태 저장
        state_snapshot = {
            "turn": self.turn,
            "halfmove_clock": self.halfmove_clock,
            "fullmove_number": self.fullmove_number,
            "zobrist": self.zobrist_hash(),  # 복구 시 참고
            # 필요 시 캐슬 권리, 앙파상 가능성 등 추가
            "castle_rights": self.castle_rights.copy() if hasattr(self, "castle_rights") else None,
            "en_passant": self.en_passant if hasattr(self, "en_passant") else None
        }
        move_stack.append(("null", state_snapshot))

        # 턴 변경
        self.turn = 'b' if self.turn == 'w' else 'w'

        # 풀무브 넘버는 백에서 흑으로 넘어갈 때만 증가
        if self.turn == 'w':
            self.fullmove_number += 1

        # Null move에서는 halfmove_clock 증가 (잡거나 폰 이동이 아니므로)
        self.halfmove_clock += 1

    def undo_null_move(self, move_stack):
        """
        Null move 취소: apply_null_move로 넘긴 턴을 원상복구
        """
        if not move_stack:
            raise ValueError("undo_null_move 호출 시 move_stack이 비어있음")

        move_type, state_snapshot = move_stack.pop()
        if move_type != "null":
            raise ValueError("마지막 수가 null move가 아님")

        # 스냅샷 복원
        self.turn = state_snapshot["turn"]
        self.halfmove_clock = state_snapshot["halfmove_clock"]
        self.fullmove_number = state_snapshot["fullmove_number"]
        if "castle_rights" in state_snapshot and state_snapshot["castle_rights"] is not None:
            self.castle_rights = state_snapshot["castle_rights"].copy()
        if "en_passant" in state_snapshot:
            self.en_passant = state_snapshot["en_passant"]
    
    def is_endgame(self) -> bool:
        """
        엔드게임 여부를 판정.
        기준:
        1. 양쪽 퀸이 모두 없거나
        2. 한쪽만 퀸이 있지만, 다른 기물(룩, 비숍, 나이트, 폰)의 총 가치가 작음
        """
        # 기물 가치 기준 (퀸 제외)
        PIECE_VALUES_SIMPLE = {
            'P': 1,  # Pawn
            'N': 3,  # Knight
            'B': 3,  # Bishop
            'R': 5   # Rook
        }

        white_queens = 0
        black_queens = 0
        white_minor_major_score = 0
        black_minor_major_score = 0

        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece is None:
                    continue

                name = piece.name.upper()
                if name == 'Q':
                    if piece.side == 'w':
                        white_queens += 1
                    else:
                        black_queens += 1
                elif name in PIECE_VALUES_SIMPLE:
                    if piece.side == 'w':
                        white_minor_major_score += PIECE_VALUES_SIMPLE[name]
                    else:
                        black_minor_major_score += PIECE_VALUES_SIMPLE[name]

        # 양쪽 퀸 없음
        if white_queens == 0 and black_queens == 0:
            return True

        # 한쪽만 퀸 있고, 상대방 다른 기물 점수가 매우 낮음 (ex: 7점 이하)
        if white_queens == 0 and black_minor_major_score <= 7:
            return True
        if black_queens == 0 and white_minor_major_score <= 7:
            return True

        return False
# ---------------------
# Improved evaluation
# ---------------------
def evaluate_position(state: GameState, prev_score: Optional[float] = None) -> float:
    values = PIECE_VALUES
    score = 0
    exist = 0
    # Material
    for r in range(8):
        for c in range(8):
            p = state.board[r][c]
            if p:
                val = values[p.name]
                score += val if p.side == 'w' else -val
                if p.name == 'P':
                    # Reward advancement: white towards row 0, black towards row 7
                    adv = (6 - r) if p.side == 'w' else (r - 1)
                    score += (adv * 2) if p.side == 'w' else -(adv * 2)

    # Mobility & coop
    mobility_weights = {'K': 0.3, 'Q': 0.05, 'R': 4,'B': 5, 'N': 5, 'P': 12}
    mobility_score = 0
    coop_score = 0
    for side in ('w', 'b'):
        orig_turn = state.turn
        state.turn = side
        moves = state.generate_all_moves()
        # 각 기물별 이동 수를 가중합
        for m in moves:
            piece = m[3]  # (from,to,meta,piece) 에서 piece
            mobility_score += mobility_weights[piece.name] * (1 if side == 'w' else -1)
            target = state.board[m[1][0]][m[1][1]]
            if target is None:
                continue
            if target.side != side and target.name != 'K':
                coop_score += (values[target.name]/values[piece.name] if side == 'w' else -values[target.name]/values[piece.name])
        state.turn = orig_turn
    # 가중치
    score += mobility_score * 0.10023
    score += coop_score * 0.200125

    return score

# ---------------------
# Alpha-Beta with MVV-LVA, Killer Moves and History Heuristic + transposition
# ---------------------
MAX_PLY = 128
MATE_VALUE = 400000  # Use below arena's no-king threshold to avoid false signals

# Transposition Table bound types
TT_EXACT = 'EXACT'
TT_LOWER = 'LOWER'  # value is a lower bound (fail-high)
TT_UPPER = 'UPPER'  # value is an upper bound (fail-low)

# Futility pruning parameters (conservative)
FUT_MAX_GAIN = PIECE_VALUES.get('Q', 90)  # assume at most queen swing
FUT_WINDOW = 150  # only apply when window is reasonably narrow (score units)

NULL_MOVE_ENABLE=False

# - OVERFLOW_PREVENTION: predict next-depth cost and stop starting a new depth when likely to overrun
# - ROOT_EARLY_ABORT: experimental root-level early abort within a depth based on per-child time estimates
OVERFLOW_PREVENTION: bool = True
ROOT_EARLY_ABORT: bool = False

def move_key_tuple(m):
    # create a hashable key for a move (from,to)
    fr, to = m[0], m[1]
    return (fr[0],fr[1], to[0], to[1])

def alpha_beta(state: GameState, depth: int, alpha: float, beta: float, maximizing: bool,
               trans_table: Dict[str, Tuple[int, float, Optional[Tuple[Tuple[int,int],Tuple[int,int]]]]],
               move_stack: list, nodes=0, ply: int = 0,
               killer_moves: Optional[List[List[Optional[Tuple[Tuple[int,int],Tuple[int,int]]]]]] = None,
               history: Optional[Dict[Tuple[int,int,int,int], float]] = None,
               prev_score: Optional[float] = None,
               pv_move: Optional[Tuple[Tuple[int,int],Tuple[int,int]]] = None,
               deadline: Optional[float] = None,
               budget: Optional[Dict[str, int]] = None):

    # --- 회계/노드 카운트 ---
    if budget is not None:
        budget['used'] = budget.get('used', 0) + 1
    nodes += 1

    # --- TT probe ---
    key = state.zobrist_hash()
    tt = trans_table.get(key)
    alpha_orig, beta_orig = alpha, beta

    if tt is not None and tt[0] >= depth:
        _, tt_val, tt_bound, tt_bm = tt
        if tt_bound == TT_EXACT:
            return tt_val, tt_bm, nodes
        elif tt_bound == TT_LOWER:
            alpha = max(alpha, tt_val)
        elif tt_bound == TT_UPPER:
            beta = min(beta, tt_val)
        if alpha >= beta:
            return tt_val, tt_bm, nodes

    # --- 리프 ---
    if depth == 0:
        score = evaluate_position(state, prev_score)
        trans_table[key] = (depth, score, TT_EXACT, None)
        return score, None, nodes

    # --- 체크 상태 확인 ---
    node_in_check = state.is_in_check(state.turn)

    # --- Null Move Pruning ---
    NULL_REDUCTION = 2
    if (NULL_MOVE_ENABLE
        and depth >= NULL_REDUCTION + 1 
        and not node_in_check 
        and not pv_move
        and not state.is_endgame()):
        try:
            state.apply_null_move(move_stack)
            if not state.is_in_check(state.turn):  # null move 후 즉시 체크 확인
                null_score, _, null_nodes = alpha_beta(
                    state,
                    depth - 1 - NULL_REDUCTION,
                    -beta,
                    -beta + 1,
                    not maximizing,
                    trans_table,
                    [],
                    0,
                    ply + 1,
                    killer_moves,
                    history,
                    pv_move=None,
                    deadline=deadline,
                    budget=budget
                )
                nodes += null_nodes
                null_score = -null_score
                state.undo_null_move(move_stack)

                if null_score >= beta:
                    return beta, None, nodes  # 안전하게 beta 반환
            else:
                state.undo_null_move(move_stack)
        except SearchTimeout:
            state.undo_null_move(move_stack)
            raise

    # --- 수 생성 ---
    moves = state.generate_all_moves()

    # --- 체크메이트/스테일메이트 ---
    if not moves:
        in_check = state.is_in_check(state.turn)
        if in_check:
            val = -MATE_VALUE + ply if state.turn == 'w' else MATE_VALUE - ply
        else:
            val = 0
        trans_table[key] = (depth, val, TT_EXACT, None)
        return val, None, nodes

    # --- 무브 오더링 스코어러 ---
    values = PIECE_VALUES
    def score_move(m):
        meta = m[2]
        score = 0.0
        if pv_move and (m[0], m[1]) == pv_move:
            score += 1_000_000
        elif tt and len(tt) >= 4 and tt[3] and (m[0], m[1]) == (tt[3][0], tt[3][1]):
            score += 900_000
        elif ('capture' in meta) or ('en_passant' in meta and meta['en_passant']):
            if meta.get('en_passant'):
                score += 100
            else:
                tr, tc = m[1]
                victim = state.board[tr][tc]
                if victim:
                    score += values.get(victim.name, 0) * 100
                    if len(m) > 3:
                        score -= values.get(m[3].name, 0)
        elif killer_moves is not None and ply < len(killer_moves):
            mk = (m[0], m[1])
            if killer_moves[ply][0] and mk == killer_moves[ply][0]:
                score += 8000
            elif killer_moves[ply][1] and mk == killer_moves[ply][1]:
                score += 7000
        if history is not None:
            hk = move_key_tuple(m)
            score += history.get(hk, 0.0)
        if meta.get('castle'):
            score += 5000
        return -score

    moves.sort(key=score_move)

    # --- 탐색 루프 ---
    best_move = None
    moves_searched = 0

    allow_futility = (depth <= 2 and not node_in_check and (beta - alpha) < FUT_WINDOW)
    static_eval = evaluate_position(state) if allow_futility else None

    if maximizing:
        value = -200000
        for i, m in enumerate(moves):
            meta = m[2]
            if allow_futility and ('capture' not in meta) and (not meta.get('promotion', False)):
                if not state.gives_check(m[0], m[1], meta):
                    if static_eval is not None and static_eval + FUT_MAX_GAIN <= alpha:
                        continue
            fr_r, fr_c = m[0]
            if state.board[fr_r][fr_c] is None:
                continue
            moving_side = state.turn
            state.apply_move(m[0], m[1], meta, move_stack)
            if state.is_in_check(moving_side):
                state.undo_move(move_stack)
                continue
            moves_searched += 1
            do_full = True
            is_quiet = ('capture' not in meta) and (not meta.get('promotion', False))
            is_pv = (pv_move is not None and i == 0)
            if (depth >= 3 and i >= 3 and is_quiet and not node_in_check and not is_pv):
                red = int(0.75 + math.log(i + 1) * math.log(depth) / 2.25)
                if red > depth - 1:
                    red = depth - 1
                reduced_depth = max(1, depth - 1 - red)
                try:
                    score, _, lmr_nodes = alpha_beta(
                        state, reduced_depth, alpha, beta, False,
                        trans_table, [], 0, ply + 1,
                        killer_moves, history,
                        pv_move=None, deadline=deadline, budget=budget
                    )
                except SearchTimeout:
                    state.undo_move(move_stack)
                    raise
                nodes += lmr_nodes
                if score > alpha:
                    do_full = True
                else:
                    do_full = False
            if do_full:
                try:
                    score, _, search_nodes = alpha_beta(
                        state, depth - 1, alpha, beta, False,
                        trans_table, [], 0, ply + 1,
                        killer_moves, history,
                        pv_move=(pv_move if i == 0 else None),
                        deadline=deadline, budget=budget
                    )
                except SearchTimeout:
                    state.undo_move(move_stack)
                    raise
                nodes += search_nodes
            state.undo_move(move_stack)
            if score > value:
                value = score
                best_move = (m[0], m[1], meta)
            if value > alpha:
                alpha = value
            if alpha >= beta:
                if is_quiet and killer_moves is not None and ply < len(killer_moves):
                    mk = (m[0], m[1])
                    if killer_moves[ply][0] != mk:
                        killer_moves[ply][1] = killer_moves[ply][0]
                        killer_moves[ply][0] = mk
                if history is not None and is_quiet:
                    hk = move_key_tuple(m)
                    history[hk] = history.get(hk, 0.0) + (depth * depth)
                break
    else:
        value = 200000
        for i, m in enumerate(moves):
            meta = m[2]
            if allow_futility and ('capture' not in meta) and (not meta.get('promotion', False)):
                if not state.gives_check(m[0], m[1], meta):
                    if static_eval is not None and static_eval - FUT_MAX_GAIN >= beta:
                        continue
            fr_r, fr_c = m[0]
            if state.board[fr_r][fr_c] is None:
                continue
            moving_side = state.turn
            state.apply_move(m[0], m[1], meta, move_stack)
            if state.is_in_check(moving_side):
                state.undo_move(move_stack)
                continue
            moves_searched += 1
            do_full = True
            is_quiet = ('capture' not in meta) and (not meta.get('promotion', False))
            is_pv = (pv_move is not None and i == 0)
            if (depth >= 3 and i >= 3 and is_quiet and not node_in_check and not is_pv):
                red = int(0.75 + math.log(i + 1) * math.log(depth) / 2.25)
                if red > depth - 1:
                    red = depth - 1
                reduced_depth = max(1, depth - 1 - red)
                try:
                    score, _, lmr_nodes = alpha_beta(
                        state, reduced_depth, alpha, beta, True,
                        trans_table, [], 0, ply + 1,
                        killer_moves, history,
                        pv_move=None, deadline=deadline, budget=budget
                    )
                except SearchTimeout:
                    state.undo_move(move_stack)
                    raise
                nodes += lmr_nodes
                if score < beta:
                    do_full = True
                else:
                    do_full = False
            if do_full:
                try:
                    score, _, search_nodes = alpha_beta(
                        state, depth - 1, alpha, beta, True,
                        trans_table, [], 0, ply + 1,
                        killer_moves, history,
                        pv_move=(pv_move if i == 0 else None),
                        deadline=deadline, budget=budget
                    )
                except SearchTimeout:
                    state.undo_move(move_stack)
                    raise
                nodes += search_nodes
            state.undo_move(move_stack)
            if score < value:
                value = score
                best_move = (m[0], m[1], meta)
            if value < beta:
                beta = value
            if alpha >= beta:
                if is_quiet and killer_moves is not None and ply < len(killer_moves):
                    mk = (m[0], m[1])
                    if killer_moves[ply][0] != mk:
                        killer_moves[ply][1] = killer_moves[ply][0]
                        killer_moves[ply][0] = mk
                if history is not None and is_quiet:
                    hk = move_key_tuple(m)
                    history[hk] = history.get(hk, 0.0) + (depth * depth)
                break

    # --- 모든 수가 잘려 실제 하위 탐색이 없었을 때 ---
    if moves_searched == 0:
        if state.turn == 'w':
            return -500000, None, nodes
        else:
            return 500000, None, nodes

    # --- TT 저장 ---
    value_final = alpha if maximizing else beta
    if value_final <= alpha_orig:
        bound = TT_UPPER
    elif value_final >= beta_orig:
        bound = TT_LOWER
    else:
        bound = TT_EXACT
    trans_table[key] = (depth, value_final, bound, (best_move[0], best_move[1]) if best_move else None)
    return value_final, best_move, nodes

# Iterative Deepening with enhanced features and Aspiration Windows
def iterative_deepening_search(state: GameState, max_time: float = 2.0, start_depth: int = 4,
                               progress: Optional[Callable[[dict], None]] = None,
                               top_k: int = 5):
    """Iterative deepening alpha-beta with lightweight progress events.
    Returns (best_val, best_move, nodes_total, last_completed_depth).
    """
    trans_table: Dict[str, Tuple[int, float, Optional[Tuple[Tuple[int,int],Tuple[int,int]]]]] = {}
    move_stack: list = []
    maximizing = (state.turn == 'w')

    # Heuristics tables
    killer_moves = [[None, None] for _ in range(MAX_PLY)]
    history: Dict[Tuple[int,int,int,int], float] = {}

    best_move: Optional[Tuple[Tuple[int,int],Tuple[int,int],dict]] = None
    best_val: float = (-200000 if maximizing else 200000)
    nodes_total = 0
    start_time = time.time()
    last_completed_depth = start_depth - 1
    pv_move: Optional[Tuple[Tuple[int,int],Tuple[int,int]]] = None
    last_top_moves: Optional[list] = None

    # Timing/model history for depth scheduling
    depth_time_hist: List[float] = []  # seconds per completed depth
    nodes_hist: List[int] = []         # nodes added per completed depth

    def _compute_top_moves(k: int = 5):
        moves = state.generate_all_moves()
        scored = []
        for m in moves:
            fr, to, meta = m[0], m[1], m[2]
            fr_r, fr_c = fr
            if state.board[fr_r][fr_c] is None:
                continue
            state.apply_move(fr, to, meta, move_stack)
            try:
                val = evaluate_position(state)
            finally:
                state.undo_move(move_stack)
            scored.append((val, fr, to))
        reverse = True if maximizing else False
        scored.sort(key=lambda x: x[0], reverse=reverse)
        return [{'from': fr, 'to': to, 'val': float(v)} for (v, fr, to) in scored[:k]]

    # Time deadline
    deadline = time.perf_counter() + max(0.0, max_time)
    allowed_deadline = deadline + (0.25 * max(0.0, max_time))

    depth = start_depth
    # Aspiration window parameters (score units: pawn ≈ 10)
    ASP_INIT = 5.0
    ASP_MAX = 200.0
    asp_window = ASP_INIT
    while True:
        # Time check + predictive scheduling before starting next depth
        elapsed = time.time() - start_time
        remaining = max(0.0, (deadline - time.perf_counter()))
        allowed_remaining = max(0.0, (allowed_deadline - time.perf_counter()))
        if allowed_remaining <= 0:
            break
        # Estimate next depth cost with simple growth model; optionally stop if likely to exceed budget
        if OVERFLOW_PREVENTION and depth_time_hist:
            if len(depth_time_hist) >= 2:
                growth = depth_time_hist[-1] / max(1e-6, depth_time_hist[-2])
            elif len(nodes_hist) >= 2 and nodes_hist[-2] > 0:
                growth = nodes_hist[-1] / max(1e-6, nodes_hist[-2])
            else:
                growth = 2.0
            growth = min(4.0, max(1.2, float(growth)))
            predicted_time_for_next = depth_time_hist[-1] * growth
            if predicted_time_for_next > allowed_remaining * 0.9:
                break

        # Depth start event
        if callable(progress):
            try:
                progress({
                    'event': 'depth_start',
                    'depth': depth,
                    'elapsed': elapsed,
                    'remaining': max(0.0, (deadline - time.perf_counter())),
                    'nodes_total': nodes_total,
                    'best_move': best_move,
                    'best_val': best_val,
                    'top_moves': _compute_top_moves(top_k),
                })
            except Exception:
                pass

        depth_start_ts = time.perf_counter()

        # Root-level loop with per-move updates (MP-like)
        root_moves = state.generate_all_moves()

        # Simple ordering: captures/promotions first
        def _root_key(m):
            meta = m[2] or {}
            is_cap = 1 if meta.get('capture') else 0
            is_prom = 1 if meta.get('promotion') else 0
            is_castle = 1 if meta.get('castle') else 0
            # PV move from previous iteration to the very front
            is_pv = 0
            if pv_move is not None and (m[0], m[1]) == pv_move:
                is_pv = 1
            return (-is_pv, -is_cap, -is_prom, -is_castle)

        ordered = sorted(root_moves, key=_root_key)

        root_results = []  # list of (move, val, nodes)
        local_best_move = None
        local_best_val = (-200000 if maximizing else 200000)
        # Root aspiration window centered at previous best
        if last_completed_depth >= start_depth and best_move is not None:
            root_alpha = float(best_val) - asp_window
            root_beta = float(best_val) + asp_window
        else:
            root_alpha, root_beta = -200000.0, 200000.0

        try:
            for m in ordered:
                # Time check before each child (experimental root early-abort)
                if ROOT_EARLY_ABORT:
                    i = len(root_results)
                    if i > 0:
                        elapsed_depth = time.perf_counter() - depth_start_ts
                        avg_per_child = elapsed_depth / max(1, i)
                        remaining_children = max(0, len(ordered) - i)
                        est_remaining = avg_per_child * remaining_children
                        if (time.perf_counter() + est_remaining) > allowed_deadline:
                            raise SearchTimeout()

                fr, to, meta = m[0], m[1], m[2]

                # Skip stale move if from-square empty
                fr_r, fr_c = fr
                if state.board[fr_r][fr_c] is None:
                    continue

                moving_side = state.turn
                state.apply_move(fr, to, meta, move_stack)
                # legality guard: if own king in check, skip
                if state.is_in_check(moving_side):
                    state.undo_move(move_stack)
                    continue

                # Search child position with depth-1
                child_maximizing = (state.turn == 'w')
                try:
                    # Use aspiration at the root; if fail-low/high, expand and re-search
                    use_alpha = root_alpha
                    use_beta = root_beta
                    val, _bm, child_nodes = alpha_beta(
                        state, max(0, depth - 1), use_alpha, use_beta,
                        child_maximizing, trans_table, move_stack,
                        0, 1, killer_moves, history, pv_move=pv_move,
                        deadline=deadline, budget=None
                    )
                    while (val <= use_alpha or val >= use_beta) and asp_window < ASP_MAX:
                        asp_window = min(ASP_MAX, asp_window * 2.0)
                        center = float(best_val) if last_completed_depth >= start_depth else 0.0
                        use_alpha = center - asp_window
                        use_beta = center + asp_window
                        v2, _bm2, n2 = alpha_beta(
                            state, max(0, depth - 1), use_alpha, use_beta,
                            child_maximizing, trans_table, move_stack,
                            0, 1, killer_moves, history, pv_move=pv_move,
                            deadline=deadline, budget=None
                        )
                        val = v2
                        child_nodes += int(n2 or 0)
                except SearchTimeout:
                    state.undo_move(move_stack)
                    raise
                finally:
                    # ensure undo
                    state.undo_move(move_stack)

                nodes_total += int(child_nodes or 0)
                root_results.append((m, val, child_nodes))

                # Update local best
                better = (val > local_best_val) if maximizing else (val < local_best_val)
                if better:
                    local_best_val = val
                    local_best_move = m
                    # Sharpen root window around improving best (still inclusive of previous bounds)
                    if last_completed_depth >= start_depth and best_move is not None:
                        root_alpha = max(root_alpha, float(local_best_val) - asp_window)
                        root_beta = max(root_beta, root_alpha + 1.0)

                # Emit MP-like partial update
                if callable(progress):
                    try:
                        # Build top_k from evaluated root_results so far
                        sorted_partial = sorted(
                            root_results,
                            key=lambda t: (t[1] if maximizing else -t[1]),
                            reverse=True
                        )[:top_k]
                        top_moves = [
                            {'from': mm[0], 'to': mm[1], 'val': vv}
                            for (mm, vv, _nn) in sorted_partial
                        ]
                        now = time.time()
                        progress({
                            'event': 'update',
                            'depth': depth,
                            'elapsed': now - start_time,
                            'remaining': max(0.0, (deadline - time.perf_counter())),
                            'nodes_total': nodes_total,
                            'best_move': local_best_move,
                            'best_val': local_best_val,
                            'top_moves': top_moves,
                        })
                    except Exception:
                        pass

            # Completed full depth
            if local_best_move is not None:
                best_move = local_best_move
                best_val = local_best_val
                last_completed_depth = depth
                pv_move = (best_move[0], best_move[1])
                # Tighten aspiration window again for next depth
                asp_window = max(ASP_INIT, asp_window / 2.0)

            # Record depth timing and nodes for scheduling model
            try:
                depth_time_measured = max(0.0, time.perf_counter() - depth_start_ts)
                depth_time_hist.append(float(depth_time_measured))
                nodes_hist.append(max(0, int(nodes_total)))
            except Exception:
                pass

            # Emit depth_complete with final top list
            if callable(progress):
                try:
                    sorted_full = sorted(
                        root_results,
                        key=lambda t: (t[1] if maximizing else -t[1]),
                        reverse=True
                    )[:top_k]
                    top_moves = [
                        {'from': mm[0], 'to': mm[1], 'val': vv}
                        for (mm, vv, _nn) in sorted_full
                    ]
                    # remember for final event
                    last_top_moves = top_moves
                    progress({
                        'event': 'depth_complete',
                        'depth': depth,
                        'elapsed': time.time() - start_time,
                        'remaining': max(0.0, (deadline - time.perf_counter())),
                        'depth_time': max(0.0, time.perf_counter() - depth_start_ts),
                        'nodes_total': nodes_total,
                        'best_move': best_move,
                        'best_val': best_val,
                        'top_moves': top_moves,
                    })
                except Exception:
                    pass

        except SearchTimeout:
            # Partial results: emit timeout + depth summary with what we have
            if callable(progress):
                try:
                    reason = 'time' if time.perf_counter() >= allowed_deadline else 'node_budget'
                    # Emit a timeout event with current best and partial top_moves
                    sorted_partial = sorted(
                        root_results,
                        key=lambda t: (t[1] if maximizing else -t[1]),
                        reverse=True
                    )[:top_k]
                    top_moves = [
                        {'from': mm[0], 'to': mm[1], 'val': vv}
                        for (mm, vv, _nn) in sorted_partial
                    ]
                    last_top_moves = top_moves
                    progress({
                        'event': 'timeout', 
                        'depth': depth, 
                        'reason': reason, 
                        'best_move': local_best_move or best_move, 
                        'best_val': local_best_val if local_best_move else best_val, 
                        'top_moves': top_moves
                    })
                except Exception:
                    pass
            break

        depth += 1

    # Final progress
    if callable(progress):
        try:
            final_top = list(last_top_moves) if last_top_moves else []
            # Guarantee the chosen best_move is present in final top_moves for UI consistency
            if best_move is not None:
                bm_from, bm_to, _bm_meta = best_move[0], best_move[1], best_move[2]
                found = any((tm.get('from') == bm_from and tm.get('to') == bm_to) for tm in final_top)
                if not found:
                    try:
                        final_top.insert(0, {'from': bm_from, 'to': bm_to, 'val': float(best_val if best_val is not None else 0.0)})
                    except Exception:
                        final_top.insert(0, {'from': bm_from, 'to': bm_to, 'val': 0.0})
            # trim to top_k
            if top_k is not None and isinstance(top_k, int) and top_k > 0:
                final_top = final_top[:top_k]

            progress({
                'event': 'final',
                'depth': last_completed_depth,
                'elapsed': time.time() - start_time,
                'nodes_total': nodes_total,
                'best_move': best_move,
                'best_val': best_val,
                'top_moves': final_top,
            })
        except Exception:
            pass

    return best_val, best_move, nodes_total, last_completed_depth

# Custom exception used for cooperative search timeout
class SearchTimeout(Exception):
    pass

# ---------------------
# IO helpers & FEN loader
# ---------------------
FILES = 'abcdefgh'
def algebraic_to_rc(s: str) -> Tuple[int,int]:
    file = s[0].lower()
    rank = s[1]
    c = FILES.index(file)
    r = 8 - int(rank)  # <-- 수정: 1이 7, 8이 0이 되도록 변환
    return r,c

def rc_to_algebraic(rc: Tuple[int,int]) -> str:
    r,c = rc
    return f"{FILES[c]}{8 - r}"  # <-- 수정: 내부 0이 8, 7이 1이 되도록 변환

def print_board(state: GameState):
    print('  a b c d e f g h')
    for r in range(8):  # row 0 (top) is rank 8
        rank = 8 - r
        rowstr = f"{rank} "
        for c in range(8):
            p = state.board[r][c]
            if p is None:
                rowstr += '. '
            else:
                sym = p.name.upper() if p.side=='w' else p.name.lower()
                rowstr += sym + ' '
        print(rowstr)
    print()
    print(f"Turn: {state.turn} | Captured W: {[p.name for p in state.captured['w']]} | Captured B: {[p.name for p in state.captured['b']]}\n")

def fen_to_board(fen: str):
    piece_map = {
        'K': lambda: King('w'), 'Q': lambda: Queen('w'),
        'R': lambda: Rook('w'), 'B': lambda: Bishop('w'),
        'N': lambda: Knight('w'), 'P': lambda: Pawn('w'),
        'k': lambda: King('b'), 'q': lambda: Queen('b'),
        'r': lambda: Rook('b'), 'b': lambda: Bishop('b'),
        'n': lambda: Knight('b'), 'p': lambda: Pawn('b'),
    }
    rows = fen.split()[0].split('/')  # FEN lists ranks from 8 -> 1; row 0 = rank 8
    board = []
    for row in rows:
        br = []
        for ch in row:
            if ch.isdigit():
                br.extend([None]*int(ch))
            else:
                br.append(piece_map[ch]())
        board.append(br)
    return board

# ---------------------
# CLI
# ---------------------
# === PARALLEL PROCESSING OPTIMIZATIONS ===
# ThreadPoolExecutor functions removed - only ProcessPoolExecutor provides true parallelism due to Python GIL

if __name__ == "__main__":
    gs = GameState()
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    gs.board = fen_to_board(start_fen)

    print("Enhanced Alpha-Beta CLI (activity + cooperation + adaptive depth + killer/history)")
    print_board(gs)

    human_side = input("Choose your side (w/b) [w]: ").strip().lower() or 'w'
    max_time = float(input("Max AI think time (sec) [2.0]: ").strip() or "2.0")

    move_stack = []
    while True:
        moves = gs.generate_all_moves()
        if not moves:
            print("No legal moves. Game over.")
            break

        if gs.turn == human_side:
            user = input("Your move (e2e4): ").strip()
            if user.lower() in ('resign','quit','exit'):
                print("You resigned.")
                break
            if len(user) == 4:
                from_s, to_s = user[:2], user[2:]
            else:
                parts = user.split()
                if len(parts) != 2:
                    print("Invalid input")
                    continue
                from_s, to_s = parts
            try:
                fr = algebraic_to_rc(from_s)
                to = algebraic_to_rc(to_s)
            except Exception as e:
                print("Invalid coord:", e)
                continue
            legal = None
            for m in moves:
                if m[0] == fr and m[1] == to:
                    legal = m; break
            if not legal:
                print("Illegal move.")
                continue
            gs.apply_move(legal[0], legal[1], legal[2], move_stack)
            print_board(gs)
        else:
            print("AI thinking...")
            val, move, nodes, depth = iterative_deepening_search(gs, max_time=max_time, start_depth=4)
            print(f"AI 탐색 완료 깊이: {depth} (탐색 노드 수: {nodes})")
            if not move:
                print("AI found no move.")
                break
            fr, to, meta = move
            print(f"AI: {rc_to_algebraic(fr)} -> {rc_to_algebraic(to)} (eval {val})")
            gs.apply_move(fr, to, meta, move_stack)
            print_board(gs)