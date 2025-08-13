# chess_gui.py - Tkinter GUI for Chess with Alpha-Beta AI
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time
from typing import List, Tuple, Dict, Optional
import math

# Import the chess engine components from alphabeta.py
from alphabeta import (
    GameState, rc_to_algebraic, fen_to_board, on_board, iterative_deepening_search
)
from simple_multiprocess import get_multiprocess_move

class ChessGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chess with Alpha-Beta AI")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        self.root.minsize(800, 600)

        # Game state
        self.game_state = GameState()
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        self.game_state.board = fen_to_board(start_fen)
        self.move_stack = []

        # GUI state
        self.selected_square = None
        self.possible_moves = []
        self.selection_side = None  # side ('w'/'b') of currently selected piece
        self.view_only_selection = False  # if True, selection is preview-only (no move execution)
        self.human_side = 'w'
        self.ai_thinking = False
        self.ai_max_time = 2.0
        self.use_multiprocessing = False  # AI 모드 설정
        # Vector overlay settings
        self.show_vectors = True
        self._vector_colors = ["#FF0000", "#FF7B00", "#EEFF00", "#1EFF00", "#00FFFF"]


        # 자동으로 최적의 워커 수 계산
        import multiprocessing
        self.cpu_count = multiprocessing.cpu_count()
        # CPU 코어 수에 따라 적절한 워커 수 설정
        if self.cpu_count >= 16:
            self.optimal_workers = min(8, self.cpu_count // 2)  # 많은 코어: 절반 사용
        elif self.cpu_count >= 8:
            self.optimal_workers = min(6, self.cpu_count - 2)   # 중간 코어: 2개 여유
        elif self.cpu_count >= 4:
            self.optimal_workers = self.cpu_count - 1           # 적은 코어: 1개 여유
        else:
            self.optimal_workers = self.cpu_count               # 매우 적은 코어: 모두 사용

        self.max_workers = self.optimal_workers  # 멀티프로세싱 워커 수

        # AI analysis tracking
        self.analysis_data = {
            'start_time': 0,
            'current_depth': 0,
            'total_nodes': 0,
            'best_move': None,
            'evaluation': 0,
            'progress': 0,
            'elapsed': 0.0,
            'remaining': None,
            'nps': 0,
            'status': ''
        }

        # Colors
        self.light_color = "#F0D9B5"
        self.dark_color = "#B58863"
        self.selected_color = "#FFFF00"
        self.possible_move_color = "#90EE90"
        self.castle_move_color = "#87CEEB"  # Sky blue for castling moves
        self.en_passant_color = "#FFB6C1"  # Light pink for en passant moves
        self.last_move_color = "#FFB347"

        # Piece symbols (Unicode chess pieces)
        self.piece_symbols = {
            'wK': '♔', 'wQ': '♕', 'wR': '♖', 'wB': '♗', 'wN': '♘', 'wP': '♙',
            'bK': '♚', 'bQ': '♛', 'bR': '♜', 'bB': '♝', 'bN': '♞', 'bP': '♟'
        }

        self.last_move = None  # Store last move for highlighting
        self.top_moves = []  # list of dicts: {'from': (r,c), 'to': (r,c), 'val': float}

        self.setup_ui()
        self.update_board()

        # Start AI if it's AI's turn
        if self.game_state.turn != self.human_side:
            self.root.after(500, self.ai_move)
    
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Left panel for board
        board_frame = tk.Frame(main_frame)
        board_frame.pack(side='left', padx=(0, 20))
        
        # Board canvas
        self.canvas = tk.Canvas(board_frame, width=480, height=480, bg='white')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_square_click)
        
        # Right panel for controls and info
        control_frame = tk.Frame(main_frame, width=300)  # 조금 더 넓게
        control_frame.pack(side='right', fill='y', padx=(20, 0))
        control_frame.pack_propagate(False)
        
        # Create tabbed interface
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill='both', expand=True, pady=(0, 10))
        
        # Tab 1: Game Controls
        game_tab = ttk.Frame(notebook)
        notebook.add(game_tab, text='Game')
        
        # Tab 2: AI Settings
        ai_tab = ttk.Frame(notebook)
        notebook.add(ai_tab, text='AI')
        
        # Tab 3: Analysis
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text='Analysis')
        
        # Tab 4: Game Info
        info_tab = ttk.Frame(notebook)
        notebook.add(info_tab, text='Info')
        
        # === GAME TAB ===
        self.setup_game_tab(game_tab)
        
        # === AI TAB ===
        self.setup_ai_tab(ai_tab)
        
        # === ANALYSIS TAB ===
        self.setup_analysis_tab(analysis_tab)
        
    # === INFO TAB ===
        self.setup_info_tab(info_tab)
        
        self.log_message("Game started. You are playing as White.")
    
    def setup_game_tab(self, parent):
        """Setup the Game tab"""
        # Game controls title
        tk.Label(parent, text="Game Controls", font=('Arial', 14, 'bold')).pack(pady=(10, 15))
        
        # Side selection
        side_frame = tk.Frame(parent)
        side_frame.pack(fill='x', pady=10)
        tk.Label(side_frame, text="Your side:", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.side_var = tk.StringVar(value='w')
        tk.Radiobutton(side_frame, text="White", variable=self.side_var, value='w', 
                      command=self.change_side).pack(anchor='w')
        tk.Radiobutton(side_frame, text="Black", variable=self.side_var, value='b', 
                      command=self.change_side).pack(anchor='w')
        
        # Buttons
        button_frame = tk.Frame(parent)
        button_frame.pack(fill='x', pady=15)
        
        tk.Button(button_frame, text="New Game", command=self.new_game, 
                 bg='lightgreen', width=18).pack(pady=3)
        tk.Button(button_frame, text="Undo Move", command=self.undo_move, 
                 bg='lightblue', width=18).pack(pady=3)
        tk.Button(button_frame, text="AI Hint", command=self.get_hint, 
                 bg='lightyellow', width=18).pack(pady=3)
        
        # Status display
        status_frame = tk.Frame(parent)
        status_frame.pack(fill='both', expand=True, pady=10)
        tk.Label(status_frame, text="Game Status", font=('Arial', 11, 'bold')).pack()
        
        self.status_text = tk.Text(status_frame, height=8, wrap='word', 
                                  font=('Arial', 9), bg='#f0f0f0')
        status_scroll = tk.Scrollbar(status_frame, command=self.status_text.yview)
        self.status_text.config(yscrollcommand=status_scroll.set)
        self.status_text.pack(side='left', fill='both', expand=True)
        status_scroll.pack(side='right', fill='y')
    
    def setup_ai_tab(self, parent):
        """Setup the AI tab"""
        # AI Settings title
        tk.Label(parent, text="AI Settings", font=('Arial', 14, 'bold')).pack(pady=(10, 15))
        
        # AI difficulty
        diff_frame = tk.Frame(parent)
        diff_frame.pack(fill='x', pady=10)
        tk.Label(diff_frame, text="AI Think Time:", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.time_var = tk.StringVar(value="2.0")
        time_spin = tk.Spinbox(diff_frame, from_=0.5, to=1000.0, increment=0.5, 
                              textvariable=self.time_var, width=10,
                              command=self.change_ai_time)
        time_spin.pack(anchor='w')
        tk.Label(diff_frame, text="seconds").pack(anchor='w')
        
        # AI Mode selection
        ai_mode_frame = tk.Frame(parent)
        ai_mode_frame.pack(fill='x', pady=15)
        tk.Label(ai_mode_frame, text="AI Mode:", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.ai_mode_var = tk.StringVar(value='sequential')
        tk.Radiobutton(ai_mode_frame, text="Sequential", 
                      variable=self.ai_mode_var, value='sequential',
                      command=self.change_ai_mode).pack(anchor='w')
        tk.Radiobutton(ai_mode_frame, text="Multiprocessing", 
                      variable=self.ai_mode_var, value='multiprocessing',
                      command=self.change_ai_mode).pack(anchor='w')
        
        # Worker count for multiprocessing
        self.worker_frame = tk.Frame(parent)
        self.worker_frame.pack(fill='x', pady=10)
        tk.Label(self.worker_frame, text="Workers:", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.worker_var = tk.StringVar(value=str(self.max_workers))
        worker_spin = tk.Spinbox(self.worker_frame, from_=1, to=16, increment=1,
                                textvariable=self.worker_var, width=8,
                                command=self.change_worker_count)
        worker_spin.pack(anchor='w')
        tk.Label(self.worker_frame, text=f"(CPU cores: {self.cpu_count}, optimal: {self.optimal_workers})").pack(anchor='w')
        self.worker_frame.pack_forget()  # Initially hidden

    
    def setup_analysis_tab(self, parent):
        """Setup the Analysis tab"""
        # AI Analysis title
        tk.Label(parent, text="AI Analysis", font=('Arial', 14, 'bold')).pack(pady=(10, 15))
    
        # Progress bar for AI thinking
        progress_frame = tk.Frame(parent)
        progress_frame.pack(fill='x', pady=10)
        tk.Label(progress_frame, text="AI Progress:", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=250)
        self.progress_bar.pack(fill='x', pady=5)
        
        # Current analysis info
        analysis_info_frame = tk.Frame(parent)
        analysis_info_frame.pack(fill='x', pady=10)
        tk.Label(analysis_info_frame, text="Current Analysis:", font=('Arial', 10, 'bold')).pack(anchor='w')
        self.analysis_label = tk.Label(analysis_info_frame, text="Waiting...", 
                                     font=('Arial', 9), bg='#e0e0e0', relief='sunken', height=2)
        self.analysis_label.pack(fill='x', pady=5)
        
        # Depth and nodes info
        stats_frame = tk.Frame(parent)
        stats_frame.pack(fill='x', pady=10)
        tk.Label(stats_frame, text="Search Statistics:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        info_grid = tk.Frame(stats_frame)
        info_grid.pack(fill='x', pady=5)
        
        self.depth_label = tk.Label(info_grid, text="Depth: 0", font=('Arial', 9), bg='#f0f0f0', relief='sunken')
        self.depth_label.grid(row=0, column=0, sticky='ew', padx=(0, 2))
        
        self.nodes_label = tk.Label(info_grid, text="Nodes: 0", font=('Arial', 9), bg='#f0f0f0', relief='sunken')
        self.nodes_label.grid(row=0, column=1, sticky='ew', padx=(2, 0))
        
        info_grid.grid_columnconfigure(0, weight=1)
        info_grid.grid_columnconfigure(1, weight=1)

        # Time/NPS and status
        extra_frame = tk.Frame(parent)
        extra_frame.pack(fill='x', pady=6)
        self.time_nps_label = tk.Label(extra_frame, text="Time: 0.00s | NPS: 0", font=('Arial', 9), bg='#f7f7f7', relief='sunken')
        self.time_nps_label.pack(fill='x', pady=(0, 4))
        self.status_label = tk.Label(extra_frame, text="", font=('Arial', 9), bg='#f7f7f7', relief='sunken')
        self.status_label.pack(fill='x')

        # Candidate vectors controls (merged from Vectors tab)
        vectors_frame = tk.Frame(parent)
        vectors_frame.pack(fill='x', pady=6)
        self.show_vectors_var = tk.BooleanVar(value=self.show_vectors)
        tk.Checkbutton(vectors_frame, text="Show candidate arrows", variable=self.show_vectors_var,
                       command=self._toggle_vectors).pack(anchor='w')
        # Legend title
        tk.Label(parent, text="Legend (Top candidates)", font=('Arial', 11, 'bold')).pack(anchor='w', pady=(8, 4))
        # Legend list container
        self.legend_container = tk.Frame(parent)
        self.legend_container.pack(fill='x')
        # initial legend build
        try:
            self._update_vector_legend()
        except Exception:
            pass
    
    def setup_info_tab(self, parent):
        """Setup the Info tab"""
        # Game Info title
        tk.Label(parent, text="Game Information", font=('Arial', 14, 'bold')).pack(pady=(10, 15))
        
        # Captured pieces display
        captured_frame = tk.Frame(parent)
        captured_frame.pack(fill='x', pady=10)
        tk.Label(captured_frame, text="Captured Pieces", font=('Arial', 11, 'bold')).pack(anchor='w')
        
        self.captured_white = tk.Label(captured_frame, text="White: ", 
                                     font=('Arial', 14), bg='white', relief='sunken', height=2)
        self.captured_white.pack(fill='x', pady=2)
        
        self.captured_black = tk.Label(captured_frame, text="Black: ", 
                                     font=('Arial', 14), bg='white', relief='sunken', height=2)
        self.captured_black.pack(fill='x', pady=2)
        
        # Castling rights display
        castle_frame = tk.Frame(parent)
        castle_frame.pack(fill='x', pady=15)
        tk.Label(castle_frame, text="Castling Rights", font=('Arial', 11, 'bold')).pack(anchor='w')
        
        self.castle_rights = tk.Label(castle_frame, text="White: O-O O-O-O | Black: O-O O-O-O", 
                                    font=('Arial', 9), bg='#f0f0f0', relief='sunken', height=2)
        self.castle_rights.pack(fill='x', pady=5)
        
        # En passant target display
        ep_frame = tk.Frame(parent)
        ep_frame.pack(fill='x', pady=10)
        tk.Label(ep_frame, text="En Passant Target", font=('Arial', 11, 'bold')).pack(anchor='w')
        
        self.en_passant_label = tk.Label(ep_frame, text="None", 
                                       font=('Arial', 9), bg='#f0f0f0', relief='sunken', height=2)
        self.en_passant_label.pack(fill='x', pady=5)

    

    def _toggle_vectors(self):
        try:
            self.show_vectors = bool(self.show_vectors_var.get())
            self.update_board()
        except Exception:
            pass

    def _update_vector_legend(self):
        """Rebuild the legend list from current top_moves."""
        try:
            if not hasattr(self, 'legend_container') or self.legend_container is None:
                return
            for child in list(self.legend_container.children.values()):
                try:
                    child.destroy()
                except Exception:
                    pass
            # Build rows
            for idx in range(5):
                item = self.top_moves[idx]
                row = tk.Frame(self.legend_container)
                row.pack(fill='x', pady=2)
                # color box
                col = self._vector_colors[idx % len(self._vector_colors)]
                box = tk.Canvas(row, width=14, height=14, highlightthickness=0)
                box.create_rectangle(0, 0, 14, 14, fill=col, outline=col)
                box.pack(side='left', padx=(0, 6))
                # text label
                try:
                    fr_txt = rc_to_algebraic(item.get('from'))
                    to_txt = rc_to_algebraic(item.get('to'))
                except Exception:
                    fr_txt = str(item.get('from'))
                    to_txt = str(item.get('to'))
                try:
                    val_txt = f"{float(item.get('val')):.2f}"
                except Exception:
                    val_txt = ""
                lbl = tk.Label(row, text=f"{idx+1}. {fr_txt} -> {to_txt}    {val_txt}")
                lbl.pack(side='left')
        except Exception:
            pass
    
    def _get_moves_for_square(self, row: int, col: int, side: str):
        """Generate moves for a specific square as if it's 'side' to move.
        Temporarily switch turn to gather moves, then restore.
        """
        orig_turn = self.game_state.turn
        try:
            self.game_state.turn = side
            moves = self.game_state.generate_all_moves()
            return [m for m in moves if m[0] == (row, col)]
        finally:
            self.game_state.turn = orig_turn
    
    def update_ai_analysis(self, depth=0, nodes=0, best_move=None, evaluation=0):
        """Update AI analysis display in real-time"""
        self.analysis_data['current_depth'] = depth
        self.analysis_data['total_nodes'] = nodes
        self.analysis_data['best_move'] = best_move
        self.analysis_data['evaluation'] = evaluation
        
        # Update progress based on elapsed time
        if self.analysis_data['start_time'] > 0:
            elapsed = time.time() - self.analysis_data['start_time']
            progress = min(100, (elapsed / self.ai_max_time) * 100)
            self.progress_var.set(progress)
        
        # Update depth and nodes labels
        self.depth_label.config(text=f"Depth: {depth}")
        self.nodes_label.config(text=f"Nodes: {nodes:,}")
        
        # Update analysis text
        if best_move:
            from_sq = rc_to_algebraic(best_move[0])
            to_sq = rc_to_algebraic(best_move[1])
            analysis_text = f"Best: {from_sq}->{to_sq} ({evaluation:.2f})"
        else:
            analysis_text = f"Analyzing... (depth {depth})"
        
        self.analysis_label.config(text=analysis_text)

        # Time/NPS (fallback when detailed progress not provided)
        elapsed_local = (time.time() - self.analysis_data['start_time']) if self.analysis_data['start_time'] else 0.0
        nps_local = int(nodes / elapsed_local) if elapsed_local > 0 else 0
        self.time_nps_label.config(text=f"Time: {elapsed_local:.2f}s | NPS: {nps_local:,}")
        # Keep status as-is
    
    def reset_analysis_display(self):
        """Reset the analysis display"""
        self.progress_var.set(0)
        self.analysis_label.config(text="Waiting...")
        self.depth_label.config(text="Depth: 0")
        self.nodes_label.config(text="Nodes: 0")
        self.time_nps_label.config(text="Time: 0.00s | NPS: 0")
        self.status_label.config(text="")
        self.analysis_data = {
            'start_time': 0,
            'current_depth': 0,
            'total_nodes': 0,
            'best_move': None,
            'evaluation': 0,
            'progress': 0,
            'elapsed': 0.0,
            'remaining': None,
            'nps': 0,
            'status': ''
        }
        # Clear overlay arrows and refresh board
        self.top_moves = []
        # Keep legend in sync
        try:
            self._update_vector_legend()
        except Exception:
            pass
        try:
            self.update_board()
        except Exception:
            pass

    def update_ai_progress(self, evt: dict):
        """Unified handler for engine progress dict (immediate apply)."""
        try:
            event = evt.get('event')
            depth = int(evt.get('depth') or 0)
            nodes = int(evt.get('nodes_total') or 0)
            best_move = evt.get('best_move')
            best_val = float(evt.get('best_val') or 0)
            # optional top move vectors
            top_moves_evt = evt.get('top_moves')
            if top_moves_evt is not None:
                self._set_top_moves(top_moves_evt)
            else:
                # Fallback: top_moves가 없는 경우 best_move만으로 1개 화살표 표시
                bm = best_move
                try:
                    if isinstance(bm, (list, tuple)) and len(bm) >= 2:
                        fr, to = bm[0], bm[1]
                        if (isinstance(fr, (list, tuple)) and len(fr) == 2 and
                            isinstance(to, (list, tuple)) and len(to) == 2):
                            self._set_top_moves([{'from': tuple(fr), 'to': tuple(to), 'val': float(best_val)}])
                except Exception:
                    pass
            elapsed = float(evt.get('elapsed') or 0.0)
            remaining = evt.get('remaining')
            if remaining is not None:
                try:
                    remaining = float(remaining)
                except Exception:
                    remaining = None
            nps = int(nodes / elapsed) if elapsed > 0 else 0

            # Store
            self.analysis_data['elapsed'] = elapsed
            self.analysis_data['remaining'] = remaining
            self.analysis_data['nps'] = nps
            if event in ('early_stop', 'timeout'):
                reason = evt.get('reason', event)
                self.analysis_data['status'] = f"{reason}"
                self.status_label.config(text=self.analysis_data['status'])
                # Also log once
                self.log_message(f"AI {reason} at depth {depth}")
            elif event == 'final':
                self.analysis_data['status'] = 'done'
                self.status_label.config(text='done')
                self.progress_var.set(100)

            # Progress bar update
            if self.ai_max_time > 0:
                progress = min(100, (elapsed / self.ai_max_time) * 100)
                self.progress_var.set(progress)

            # Update text labels (reuse existing updater)
            self.update_ai_analysis(depth, nodes, best_move, best_val)
            # Request board redraw to show candidate vectors
            self.update_board()

            # Time/NPS label with remaining if available
            time_txt = f"Time: {elapsed:.2f}s"
            if remaining is not None:
                time_txt += f" | Rem: {remaining:.2f}s"
            self.time_nps_label.config(text=f"{time_txt} | NPS: {nps:,}")
        except Exception:
            pass
    
    def log_message(self, message):
        """Add message to status log"""
        self.status_text.insert('end', f"{message}\n")
        self.status_text.see('end')
    
    def change_side(self):
        """Change human player side"""
        if not self.ai_thinking:
            self.human_side = self.side_var.get()
            self.log_message(f"You are now playing as {'White' if self.human_side == 'w' else 'Black'}")
            self.update_board()
            if self.game_state.turn != self.human_side:
                self.root.after(500, self.ai_move)
    
    def change_ai_time(self):
        """Change AI thinking time"""
        try:
            self.ai_max_time = float(self.time_var.get())
            self.log_message(f"AI think time set to {self.ai_max_time} seconds")
        except ValueError:
            self.ai_max_time = 2.0
    
    def change_ai_mode(self):
        """Change AI processing mode"""
        mode = self.ai_mode_var.get()
        if mode == 'multiprocessing':
            self.use_multiprocessing = True
            self.worker_frame.pack(fill='x', pady=10)
            self.log_message("AI mode: Multiprocessing enabled")
        else:
            self.use_multiprocessing = False
            self.worker_frame.pack_forget()
            self.log_message("AI mode: Sequential processing")
    
    def change_worker_count(self):
        """Change number of workers for multiprocessing"""
        try:
            self.max_workers = int(self.worker_var.get())
            self.log_message(f"Multiprocessing workers set to {self.max_workers}")
        except ValueError:
            self.max_workers = 4
    
    def new_game(self):
        """Start a new game"""
        if self.ai_thinking:
            self.log_message("Please wait for AI to finish thinking.")
            return

        # 확인 대화상자: 진행 중 게임 초기화 여부 확인
        try:
            has_progress = len(self.move_stack) > 0
        except Exception:
            has_progress = True
        if has_progress:
            if not messagebox.askyesno(
                "New Game",
                "정말로 새 게임을 시작할까요?\n현재 진행 중인 게임이 초기화됩니다."
            ):
                return

        self.game_state = GameState()
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        self.game_state.board = fen_to_board(start_fen)
        self.move_stack = []
        self.selected_square = None
        self.possible_moves = []
        self.selection_side = None
        self.view_only_selection = False
        self.last_move = None

        self.update_board()
        self.update_captured_pieces()
        self.update_castle_rights()
        self.update_en_passant_display()
        self.reset_analysis_display()
        self.log_message("New game started!")

        if self.game_state.turn != self.human_side:
            self.root.after(500, self.ai_move)
    
    def undo_move(self):
        """Undo the last two moves (human and AI)"""
        if self.ai_thinking:
            self.log_message("Please wait for AI to finish thinking.")
            return
        
        if len(self.move_stack) >= 2:
            # Undo AI move
            self.game_state.undo_move(self.move_stack)
            # Undo human move
            self.game_state.undo_move(self.move_stack)
            self.selected_square = None
            self.possible_moves = []
            self.selection_side = None
            self.view_only_selection = False
            self.last_move = None
            self.update_board()
            self.update_captured_pieces()
            self.update_castle_rights()
            self.update_en_passant_display()
            self.log_message("Last moves undone.")
        else:
            self.log_message("No moves to undo.")
    
    def get_hint(self):
        """Get AI hint for current position"""
        if self.ai_thinking:
            self.log_message("AI is already thinking...")
            return
        
        if self.game_state.turn != self.human_side:
            self.log_message("It's not your turn.")
            return
        
        self.log_message("AI is calculating hint...")
        
        # Reset and start analysis visualization for hint
        self.reset_analysis_display()
        self.analysis_data['start_time'] = time.time()
        
        threading.Thread(target=self._calculate_hint, daemon=True).start()
        
        # Start progress update timer for hint calculation
        self._update_progress_timer()
    
    def _calculate_hint(self):
        """Calculate hint in background thread with visualization"""
        try:
            # Start analysis tracking
            self.analysis_data['start_time'] = time.time()
            self.root.after(0, lambda: self.update_ai_analysis(0, 0))
            
            # Use engine's iterative deepening with progress callback
            def _progress(evt: dict):
                # UI 스레드로 안전하게 전달
                try:
                    self.root.after(0, lambda e=dict(evt): self.update_ai_progress(e))
                except Exception:
                    pass
            val, move, nodes, depth = iterative_deepening_search(self.game_state, max_time=self.ai_max_time, start_depth=4, progress=_progress, top_k=5)
            
            if move:
                from_sq = rc_to_algebraic(move[0])
                to_sq = rc_to_algebraic(move[1])
                self.root.after(0, lambda: self.log_message(
                    f"Hint: {from_sq} -> {to_sq} (eval: {val:.2f}, depth: {depth}, nodes: {nodes:,})"
                ))
                # Keep the analysis display for a few seconds
                self.root.after(3000, self.reset_analysis_display)
            else:
                self.root.after(0, lambda: self.log_message("No good moves found."))
                self.root.after(0, self.reset_analysis_display)
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"Error calculating hint: {e}"))
            self.root.after(0, self.reset_analysis_display)
    
    def draw_board(self):
        """Draw the chess board"""
        self.canvas.delete("all")
        
        square_size = 60
        
        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size  # engine row 0 at top (rank 8 at top)
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # Determine square color
                if (row + col) % 2 == 0:
                    color = self.light_color
                else:
                    color = self.dark_color
                
                # Highlight selected square
                if self.selected_square == (row, col):
                    color = self.selected_color
                
                # Highlight possible moves
                elif (row, col) in [(m[1][0], m[1][1]) for m in self.possible_moves]:
                    # Check move type for different colors
                    move_color = self.possible_move_color
                    for m in self.possible_moves:
                        if m[1] == (row, col):
                            # Handle both 3 and 4 element move tuples
                            meta = m[2]
                            if meta.get('castle'):
                                move_color = self.castle_move_color
                                break
                            elif meta.get('en_passant'):
                                move_color = self.en_passant_color
                                break
                    color = move_color
                
                # Highlight last move
                elif self.last_move and ((row, col) == self.last_move[0] or (row, col) == self.last_move[1]):
                    color = self.last_move_color
                
                # Draw square
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                
                # Draw coordinates
                if col == 0:  # Rank numbers on left (8 at top)
                    self.canvas.create_text(x1 + 5, y1 + 5, text=str(8 - row),
                                          font=('Arial', 8), anchor='nw')
                if row == 7:  # File letters at bottom row
                    files = 'abcdefgh'
                    self.canvas.create_text(x2 - 5, y2 - 5, text=files[col],
                                          font=('Arial', 8), anchor='se')
        # Note: candidate move arrows are drawn after pieces in update_board() for visibility
    
    def draw_pieces(self):
        """Draw the chess pieces"""
        square_size = 60
        
        for row in range(8):
            for col in range(8):
                piece = self.game_state.board[row][col]
                if piece:
                    x = col * square_size + square_size // 2
                    y = row * square_size + square_size // 2
                    
                    symbol_key = f"{piece.side}{piece.name}"
                    symbol = self.piece_symbols.get(symbol_key, '?')
                    
                    self.canvas.create_text(x, y, text=symbol, font=('Arial', 30), 
                                          fill='black', anchor='center')

    def _draw_top_move_vectors(self):
        """Overlay top candidate move vectors (arrows) over the board."""
        if not self.top_moves or not self.show_vectors:
            return
        square_size = 60
        # Color map by rank in list
        colors = self._vector_colors
        width = [4, 3, 3, 2, 2]
        for idx, item in enumerate(self.top_moves):
            fr = item.get('from'); to = item.get('to')
            if not fr or not to:
                continue
            fx = fr[1] * square_size + square_size // 2
            fy = fr[0] * square_size + square_size // 2
            tx = to[1] * square_size + square_size // 2
            ty = to[0] * square_size + square_size // 2
            col = colors[idx % len(colors)]
            w = width[idx % len(width)]
            # Main line
            self.canvas.create_line(fx, fy, tx, ty, fill=col, width=w, arrow=tk.LAST)
    
    def update_board(self):
        """Update the board display"""
        self.draw_board()
        self.draw_pieces()
        # Draw candidate move arrows on top of pieces for visibility
        self._draw_top_move_vectors()
        
        # Update turn indicator
        turn_text = f"Turn: {'White' if self.game_state.turn == 'w' else 'Black'}"
        
        # Check for check/checkmate/stalemate
        if self.game_state.is_in_check():
            if self.game_state.is_checkmate():
                turn_text += " - CHECKMATE!"
            else:
                turn_text += " - CHECK!"
        elif self.game_state.is_stalemate():
            turn_text += " - STALEMATE!"
        
        if self.game_state.turn == self.human_side:
            turn_text += " (Your turn)"
        else:
            turn_text += " (AI thinking...)" if self.ai_thinking else " (AI turn)"
        
        self.root.title(f"Chess with Alpha-Beta AI - {turn_text}")
    
    def update_captured_pieces(self):
        """Update captured pieces display"""
        white_captured = ''.join([self.piece_symbols.get(f'w{p.name}', p.name) 
                                 for p in self.game_state.captured['w']])
        black_captured = ''.join([self.piece_symbols.get(f'b{p.name}', p.name) 
                                 for p in self.game_state.captured['b']])
        
        self.captured_white.config(text=f"White: {white_captured}")
        self.captured_black.config(text=f"Black: {black_captured}")
    
    def update_castle_rights(self):
        """Update castling rights display"""
        white_castle = ""
        if self.game_state.castle_check['w'][1]:  # Kingside
            white_castle += "O-O "
        if self.game_state.castle_check['w'][0]:  # Queenside
            white_castle += "O-O-O"
        if not white_castle:
            white_castle = "None"
        
        black_castle = ""
        if self.game_state.castle_check['b'][1]:  # Kingside
            black_castle += "O-O "
        if self.game_state.castle_check['b'][0]:  # Queenside
            black_castle += "O-O-O"
        if not black_castle:
            black_castle = "None"
        
        self.castle_rights.config(text=f"White: {white_castle.strip()} | Black: {black_castle.strip()}")
    
    def update_en_passant_display(self):
        """Update en passant target display"""
        if self.game_state.en_passant_target:
            ep_square = rc_to_algebraic(self.game_state.en_passant_target)
            self.en_passant_label.config(text=ep_square)
        else:
            self.en_passant_label.config(text="None")
    
    def on_square_click(self, event):
        """Handle square clicks.
        - 항상 클릭으로 선택/미리보기 가능(상대 기물 포함)
        - 실제 이동은 (내 차례 + 내 기물 선택 상태)에서만 허용
        """
        
        square_size = 60
        col = event.x // square_size
        row = event.y // square_size
        
        if not (0 <= row < 8 and 0 <= col < 8):
            return
        
        piece = self.game_state.board[row][col]

        # First click - select any piece (own or opponent) for preview
        if self.selected_square is None:
            if piece:
                self.selected_square = (row, col)
                self.selection_side = piece.side
                self.possible_moves = self._get_moves_for_square(row, col, piece.side)
                # view-only if not my turn or not my piece or AI thinking
                self.view_only_selection = not (
                    (self.game_state.turn == self.human_side) and (piece.side == self.human_side) and (not self.ai_thinking)
                )
                self.update_board()
                tag = "(view)" if self.view_only_selection else ""
                self.log_message(f"Selected {piece.name} at {rc_to_algebraic((row, col))} {tag}")
            else:
                # empty square click does nothing
                return
        else:
            # Second click
            from_square = self.selected_square
            to_square = (row, col)

            # If selection is actionable (my turn + my piece), try to move
            if (not self.view_only_selection) and (self.game_state.turn == self.human_side) and (self.selection_side == self.human_side):
                legal_move = None
                for move in self.possible_moves:
                    if move[1] == to_square:
                        legal_move = move
                        break
                if legal_move:
                    self.make_human_move(legal_move)
                    return

                # Otherwise, maybe reselect another of my pieces
                if piece and piece.side == self.human_side:
                    self.selected_square = (row, col)
                    self.selection_side = piece.side
                    self.possible_moves = self._get_moves_for_square(row, col, piece.side)
                    self.view_only_selection = False
                    self.update_board()
                    self.log_message(f"Selected {piece.name} at {rc_to_algebraic((row, col))}")
                else:
                    # Invalid target, clear selection
                    self.selected_square = None
                    self.selection_side = None
                    self.possible_moves = []
                    self.view_only_selection = False
                    self.update_board()
                    self.log_message("Invalid move. Selection cleared.")
            else:
                # View-only mode: clicking acts as preview switcher
                if piece:
                    self.selected_square = (row, col)
                    self.selection_side = piece.side
                    self.possible_moves = self._get_moves_for_square(row, col, piece.side)
                    self.view_only_selection = True
                    self.update_board()
                    self.log_message(f"Preview {piece.side.upper()} {piece.name} at {rc_to_algebraic((row, col))}")
                else:
                    # Empty square clears preview
                    self.selected_square = None
                    self.selection_side = None
                    self.possible_moves = []
                    self.view_only_selection = False
                    self.update_board()
                    self.log_message("Preview cleared.")
    
    def make_human_move(self, move):
        """Execute human player's move"""
        # Handle both 3 and 4 element move tuples
        if len(move) >= 4:
            from_sq, to_sq, meta, piece = move[0], move[1], move[2], move[3]
        else:
            from_sq, to_sq, meta = move[0], move[1], move[2]
        
        # Store for highlighting
        self.last_move = (from_sq, to_sq)
        
        # Apply move
        self.game_state.apply_move(from_sq, to_sq, meta, self.move_stack)

        # 사용자 수 이후: 분석/벡터 초기화 (다음 탐색을 위해 클린 상태)
        self.top_moves = []
        self.reset_analysis_display()

        # Clear selection
        self.selected_square = None
        self.possible_moves = []
        self.selection_side = None
        self.view_only_selection = False
        
        # Update display
        self.update_board()
        self.update_captured_pieces()
        self.update_castle_rights()
        self.update_en_passant_display()
        
        # Log move
        move_str = f"{rc_to_algebraic(from_sq)} -> {rc_to_algebraic(to_sq)}"
        if meta.get('castle'):
            castle_type = meta['castle']
            if castle_type == 'kingside':
                move_str += " (O-O)"
            elif castle_type == 'queenside':
                move_str += " (O-O-O)"
        elif meta.get('en_passant'):
            move_str += " (en passant)"
        elif meta.get('capture'):
            move_str += " (capture)"
        self.log_message(f"You played: {move_str}")
        
        # Check for game over conditions
        if self.game_state.is_checkmate():
            self.log_message("CHECKMATE! You win!")
            messagebox.showinfo("Game Over", "Checkmate! You win!")
            return
        elif self.game_state.is_stalemate():
            self.log_message("STALEMATE! It's a draw!")
            messagebox.showinfo("Game Over", "Stalemate! It's a draw!")
            return
        elif self.game_state.is_in_check():
            self.log_message("AI is in check!")
        
        # Check for game over (no legal moves)
        if not self.game_state.generate_all_moves():
            self.log_message("Game Over - No legal moves!")
            messagebox.showinfo("Game Over", "No legal moves available!")
            return
        
        # AI's turn
        self.root.after(500, self.ai_move)
    
    def ai_move(self):
        """Let AI make a move"""
        if self.game_state.turn == self.human_side:
            return
        
        moves = self.game_state.generate_all_moves()
        if not moves:
            self.log_message("Game Over - AI has no moves!")
            messagebox.showinfo("Game Over", "AI has no legal moves!")
            return
        
        self.ai_thinking = True
        self.update_board()
        self.log_message("AI is thinking...")
        
        # Reset and start analysis visualization
        self.reset_analysis_display()
        self.analysis_data['start_time'] = time.time()
        
        # Run AI calculation in background thread
        threading.Thread(target=self._ai_calculation, daemon=True).start()
        
        # Start progress update timer
        self._update_progress_timer()
    
    def _ai_calculation(self):
        """AI calculation in background thread with real-time updates"""
        try:
            # Start analysis tracking
            self.analysis_data['start_time'] = time.time()
            self.root.after(0, lambda: self.update_ai_analysis(0, 0))
            
            if self.use_multiprocessing:
                # 멀티프로세싱 모드
                self.root.after(0, lambda: self.log_message(f"Using multiprocessing with {self.max_workers} workers"))
                def _mp_progress(evt: dict):
                    # MP 진행 이벤트를 UI 스레드로 전달
                    try:
                        self.root.after(0, lambda e=dict(evt): self.update_ai_progress(e))
                    except Exception:
                        pass
                move, val, nodes, depth = get_multiprocess_move(self.game_state, self.ai_max_time, self.max_workers, progress=_mp_progress, top_k=5)
                
                if not move:
                    # 백업: 순차 처리
                    val, move, nodes, depth = self.iterative_deepening_with_updates()
            else:
                # 순차 처리 모드: engine의 iterative deepening 사용 + 진행 콜백
                def _progress2(evt: dict):
                    try:
                        self.root.after(0, lambda e=dict(evt): self.update_ai_progress(e))
                    except Exception:
                        pass
                val, move, nodes, depth = iterative_deepening_search(self.game_state, max_time=self.ai_max_time, start_depth=4, progress=_progress2, top_k=5)

            # Schedule UI update on main thread
            self.root.after(0, lambda: self._ai_move_complete(val, move, nodes, depth))
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"AI error: {e}"))
            self.ai_thinking = False
            self.root.after(0, self.reset_analysis_display)
    
    # Using alphabeta.iterative_deepening_search
    
    def _update_progress_timer(self):
        """Update progress bar periodically during AI thinking"""
        if self.ai_thinking and self.analysis_data['start_time'] > 0:
            elapsed = time.time() - self.analysis_data['start_time']
            progress = min(100, (elapsed / self.ai_max_time) * 100)
            self.progress_var.set(progress)
            
            # Continue updating every 100ms
            self.root.after(100, self._update_progress_timer)
    
    def _ai_move_complete(self, val, move, nodes, depth):
        """Complete AI move on main thread"""
        self.ai_thinking = False

        # Complete the progress bar
        self.progress_var.set(100)
        self.status_label.config(text="done")

        if not move:
            self.log_message("AI found no move. Game over!")
            messagebox.showinfo("Game Over", "AI cannot move!")
            self.reset_analysis_display()
            return

        # Handle both 3 and 4 element move tuples
        if len(move) >= 4:
            from_sq, to_sq, meta, piece = move[0], move[1], move[2], move[3]
        else:
            from_sq, to_sq, meta = move[0], move[1], move[2]

        # Store for highlighting
        self.last_move = (from_sq, to_sq)

        # Apply AI move
        self.game_state.apply_move(from_sq, to_sq, meta, self.move_stack)

        # Note: don't clear top_moves here; keep arrows until the user makes a move

        # Update display
        self.update_board()
        self.update_captured_pieces()
        self.update_castle_rights()
        self.update_en_passant_display()

        # Log AI move
        move_str = f"{rc_to_algebraic(from_sq)} -> {rc_to_algebraic(to_sq)}"
        if meta.get('castle'):
            castle_type = meta['castle']
            if castle_type == 'kingside':
                move_str += " (O-O)"
            elif castle_type == 'queenside':
                move_str += " (O-O-O)"
        elif meta.get('en_passant'):
            move_str += " (en passant)"
        elif meta.get('capture'):
            move_str += " (capture)"
        self.log_message(f"AI played: {move_str}")
        self.log_message(f"Evaluation: {val:.2f}, Depth: {depth}, Nodes: {nodes:,}")

        # Final analysis update (keep on screen until user moves)
        self.update_ai_analysis(depth, nodes, move, val)

        # Do not reset analysis here; user move will reset/clear overlays

        # Check for game over conditions
        if self.game_state.is_checkmate():
            self.log_message("CHECKMATE! You lose!")
            messagebox.showinfo("Game Over", "Checkmate! You lose!")
            return
        elif self.game_state.is_stalemate():
            self.log_message("STALEMATE! It's a draw!")
            messagebox.showinfo("Game Over", "Stalemate! It's a draw!")
            return
        elif self.game_state.is_in_check():
            self.log_message("You are in check!")

        # Check for game over (no legal moves)
        if not self.game_state.generate_all_moves():
            self.log_message("Game Over - You have no legal moves!")
            messagebox.showinfo("Game Over", "No legal moves available!")
    
    def _set_top_moves(self, top_moves_evt, limit: int = 5):
        """Normalize and store top-move vectors consistently (Sequential/MP).
        top_moves_evt: list of {'from': (r,c)|[r,c], 'to': (r,c)|[r,c], 'val': float}
        """
        try:
            if not isinstance(top_moves_evt, list):
                return
            norm = []
            for item in top_moves_evt[:limit]:
                fr_raw = item.get('from') if isinstance(item, dict) else None
                to_raw = item.get('to') if isinstance(item, dict) else None
                fr = tuple(fr_raw) if isinstance(fr_raw, (list, tuple)) and len(fr_raw) == 2 else None
                to = tuple(to_raw) if isinstance(to_raw, (list, tuple)) and len(to_raw) == 2 else None
                try:
                    val = float(item.get('val')) if isinstance(item, dict) and item.get('val') is not None else 0.0
                except Exception:
                    val = 0.0
                if fr and to:
                    norm.append({'from': fr, 'to': to, 'val': val})
            self.top_moves = norm
            # Sync legend with latest vectors
            try:
                self._update_vector_legend()
            except Exception:
                pass
        except Exception:
            pass
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = ChessGUI()
        app.run()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
