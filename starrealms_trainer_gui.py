from __future__ import annotations

import copy
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError as exc:  # pragma: no cover - depends on local Python install
    raise RuntimeError("Tkinter is required to run the trainer GUI.") from exc

ROOT_DIR = Path(__file__).resolve().parent
VENV_PYTHON_CANDIDATES = [
    ROOT_DIR / ".venv" / "Scripts" / "python.exe",
    ROOT_DIR / ".venv312" / "Scripts" / "python.exe",
]


def _preferred_venv_python() -> Optional[Path]:
    for candidate in VENV_PYTHON_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _ensure_supported_python() -> None:
    preferred_python = _preferred_venv_python()
    if preferred_python is not None and Path(sys.executable).resolve() != preferred_python.resolve():
        try:
            import torch  # type: ignore
        except Exception:
            message = (
                "Star Realms trainer GUI must be run from the project's venv.\n\n"
                f"Current interpreter:\n{sys.executable}\n\n"
                f"Run this instead:\n{preferred_python} {Path(__file__).name}\n"
                "or activate the venv first with:\n"
                f"{preferred_python.parent / 'Activate.ps1'}\n\n"
                "DirectML training currently works best from a Python 3.12 venv with torch-directml installed."
            )
            print(message)
            raise SystemExit(1)


_ensure_supported_python()

import chooser as chooser_ui
import starrealms_selfplay as sp


WINDOW_BG = "#0b1420"
PANEL_BG = "#132235"
ACCENT = "#f4b860"
TEXT = "#ecf3ff"
SUBTEXT = "#c3d3ea"
MUTED = "#87a0c2"
BUTTON_BG = "#25476a"
BUTTON_ACTIVE = "#356796"
POLL_INTERVAL_MS = 50
REFRESH_INTERVAL_COMMAND_MS = 200
REFRESH_INTERVAL_ACTIVE_MS = 350
REFRESH_INTERVAL_IDLE_MS = 1000
REFRESH_INTERVAL_ERROR_MS = 500
STAGE_STARTUP_PROGRESS_CAP = 12.0
STAGE_STARTUP_PROGRESS_SECONDS = 18.0


@dataclass
class HumanChoiceRequest:
    player_name: str
    options: List[Sequence[Any]]
    state: Dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    selection: Optional[int] = None
    error: Optional[str] = None


class HumanChoiceBridge:
    def __init__(self, app: "TrainerGUI") -> None:
        self.app = app

    def choose(self, player_name: str, options: Sequence[Sequence[Any]], state: Dict[str, Any]) -> int:
        request = HumanChoiceRequest(
            player_name=player_name,
            options=copy.deepcopy(list(options)),
            state=copy.deepcopy(dict(state)),
        )
        self.app.choice_queue.put(request)
        request.event.wait()
        if request.error is not None:
            raise RuntimeError(request.error)
        if request.selection is None:
            raise RuntimeError("No selection was returned for the human decision.")
        return request.selection


@dataclass
class GameViewRequest:
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[str] = None


class GameViewBridge:
    def __init__(self, app: "TrainerGUI") -> None:
        self.app = app

    def _round_trip(self, kind: str, **payload: Any) -> Any:
        request = GameViewRequest(kind=kind, payload=payload)
        self.app.choice_queue.put(request)
        request.event.wait()
        if request.error is not None:
            raise RuntimeError(request.error)
        return request.result

    def start_session(self, title: str) -> None:
        self._round_trip("start_session", title=title)

    def policy_choice(
        self,
        player_name: str,
        options: Sequence[Sequence[Any]],
        state: Dict[str, Any],
        selection_index: int,
    ) -> None:
        self._round_trip(
            "policy_choice",
            player_name=player_name,
            options=copy.deepcopy(list(options)),
            state=copy.deepcopy(dict(state)),
            selection_index=selection_index,
        )

    def request_human_choice(self, player_name: str, options: Sequence[Sequence[Any]], state: Dict[str, Any]) -> int:
        return int(
            self._round_trip(
                "human_choice",
                player_name=player_name,
                options=copy.deepcopy(list(options)),
                state=copy.deepcopy(dict(state)),
            )
        )

    def finish_session(self, result: Dict[str, Any]) -> None:
        self._round_trip("finish_session", result=copy.deepcopy(dict(result)))


class HumanDecisionDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, request: HumanChoiceRequest) -> None:
        super().__init__(parent, bg=WINDOW_BG)
        self.request = request
        self.selection: Optional[int] = None
        self.title(f"{request.player_name} - Choose Action")
        self.geometry("1460x920")
        self.minsize(1100, 700)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.option_var = tk.IntVar(value=0 if request.options else -1)

        self._build()
        self._populate()
        self.bind("<Return>", self._on_confirm)
        self.bind("<Escape>", self._on_cancel)

    def _build(self) -> None:
        header = tk.Frame(self, bg=WINDOW_BG, padx=12, pady=10)
        header.pack(fill="x")

        summary = self.request.state
        title_text = (
            f"{self.request.player_name}  |  "
            f"Authority {summary.get('authority')}  |  "
            f"Attack {summary.get('attack')}  |  "
            f"Trade {summary.get('trade')}  |  "
            f"Opponent {summary.get('opponentAuthority')}"
        )
        title_label = tk.Label(
            header,
            text=title_text,
            bg=WINDOW_BG,
            fg=TEXT,
            anchor="w",
            font=("Segoe UI Semibold", 16),
        )
        title_label.pack(fill="x")

        sub_text = (
            f"Must discard {summary.get('mustDiscard')}  |  "
            f"Opponent discard {summary.get('opponentMustDiscard')}  |  "
            f"Next ship top {bool(summary.get('nextShipTop'))}  |  "
            f"Blob play count {summary.get('blobPlayCount')}"
        )
        sub_label = tk.Label(
            header,
            text=sub_text,
            bg=WINDOW_BG,
            fg=SUBTEXT,
            anchor="w",
            font=("Segoe UI", 10),
        )
        sub_label.pack(fill="x", pady=(4, 0))

        body = tk.PanedWindow(self, orient="horizontal", sashrelief="flat", bg=WINDOW_BG, bd=0)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        left_panel = tk.Frame(body, bg=PANEL_BG, padx=8, pady=8)
        right_panel = tk.Frame(body, bg=PANEL_BG, padx=8, pady=8)
        body.add(left_panel, stretch="always")
        body.add(right_panel, stretch="always")

        left_title = tk.Label(
            left_panel,
            text="Game State",
            bg=PANEL_BG,
            fg=ACCENT,
            anchor="w",
            font=("Segoe UI Semibold", 12),
        )
        left_title.pack(fill="x", pady=(0, 6))

        self.state_text = ScrolledText(
            left_panel,
            bg="#0f1a29",
            fg=TEXT,
            insertbackground=TEXT,
            font=("Consolas", 9),
            wrap="word",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.state_text.pack(fill="both", expand=True)

        right_title = tk.Label(
            right_panel,
            text="Available Options",
            bg=PANEL_BG,
            fg=ACCENT,
            anchor="w",
            font=("Segoe UI Semibold", 12),
        )
        right_title.pack(fill="x", pady=(0, 6))

        list_row = tk.Frame(right_panel, bg=PANEL_BG)
        list_row.pack(fill="both", expand=True)

        list_scroll = tk.Scrollbar(list_row, orient="vertical")
        list_scroll.pack(side="right", fill="y")

        self.option_list = tk.Listbox(
            list_row,
            bg="#0f1a29",
            fg=TEXT,
            selectbackground=BUTTON_ACTIVE,
            selectforeground=TEXT,
            activestyle="none",
            font=("Segoe UI", 10),
            exportselection=False,
            yscrollcommand=list_scroll.set,
        )
        self.option_list.pack(side="left", fill="both", expand=True)
        list_scroll.configure(command=self.option_list.yview)
        self.option_list.bind("<<ListboxSelect>>", self._update_option_details)
        self.option_list.bind("<Double-Button-1>", self._on_confirm)

        detail_title = tk.Label(
            right_panel,
            text="Selected Option Details",
            bg=PANEL_BG,
            fg=ACCENT,
            anchor="w",
            font=("Segoe UI Semibold", 11),
        )
        detail_title.pack(fill="x", pady=(8, 4))

        self.detail_text = ScrolledText(
            right_panel,
            height=10,
            bg="#0f1a29",
            fg=SUBTEXT,
            insertbackground=TEXT,
            font=("Consolas", 9),
            wrap="word",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.detail_text.pack(fill="both", expand=False)

        button_row = tk.Frame(self, bg=WINDOW_BG, padx=12, pady=0)
        button_row.pack(fill="x", pady=(0, 12))

        confirm_button = tk.Button(
            button_row,
            text="Choose Selected Option",
            command=self._confirm_selection,
            bg=BUTTON_BG,
            activebackground=BUTTON_ACTIVE,
            fg=TEXT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            font=("Segoe UI Semibold", 10),
        )
        confirm_button.pack(side="left")

        cancel_button = tk.Button(
            button_row,
            text="Cancel Game",
            command=self._on_cancel,
            bg="#4f2a2a",
            activebackground="#7e3d3d",
            fg=TEXT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            font=("Segoe UI Semibold", 10),
        )
        cancel_button.pack(side="right")

    def _populate(self) -> None:
        state_snapshot = self._format_state_snapshot()
        self.state_text.configure(state="normal")
        self.state_text.delete("1.0", "end")
        self.state_text.insert("1.0", state_snapshot)
        self.state_text.configure(state="disabled")

        self.option_list.delete(0, "end")
        for index, option in enumerate(self.request.options):
            label = chooser_ui._format_option(option, self.request.state)
            self.option_list.insert("end", f"{index}. {label}")

        if self.request.options:
            self.option_list.selection_set(0)
            self.option_list.activate(0)
            self.option_list.see(0)
        self._update_option_details()

    def _cards_block(self, title: str, cards: Sequence[Any], in_play: bool = False) -> List[str]:
        lines = [title]
        if not cards:
            lines.append("  (none)")
            return lines
        for index, card in enumerate(cards):
            card_text = chooser_ui._format_card_text(card, in_play=in_play).replace("\n", " | ")
            lines.append(f"  {index}. {card_text}")
        return lines

    def _cards_in_play_block(self, title: str, cards_by_faction: Dict[str, Sequence[Any]]) -> List[str]:
        lines = [title]
        has_cards = False
        for faction in chooser_ui.FACTION_ORDER:
            faction_cards = list((cards_by_faction or {}).get(faction) or [])
            if not faction_cards:
                continue
            has_cards = True
            lines.append(f"  {faction.title()}:")
            for index, card in enumerate(faction_cards):
                card_text = chooser_ui._format_card_text(card, in_play=True).replace("\n", " | ")
                lines.append(f"    {index}. {card_text}")
        if not has_cards:
            lines.append("  (none)")
        return lines

    def _format_state_snapshot(self) -> str:
        state = self.request.state
        lines = [
            f"Authority: {state.get('authority')}  Opponent: {state.get('opponentAuthority')}",
            f"Attack: {state.get('attack')}  Trade: {state.get('trade')}",
            f"Must discard: {state.get('mustDiscard')}  Opponent must discard: {state.get('opponentMustDiscard')}",
            f"Next ship top: {bool(state.get('nextShipTop'))}  Blob plays: {state.get('blobPlayCount')}",
            "",
        ]
        lines.extend(self._cards_in_play_block("Your battlefield", state.get("cardsInPlay") or {}))
        lines.append("")
        lines.extend(self._cards_block("Your hand", state.get("hand") or []))
        lines.append("")
        lines.extend(self._cards_block("Your deck (scrambled)", state.get("scrambleDeck") or []))
        lines.append("")
        lines.extend(self._cards_block("Your known top cards", state.get("topCards") or []))
        lines.append("")
        lines.extend(self._cards_block("Your discard pile", state.get("discardPile") or []))
        lines.append("")
        lines.extend(self._cards_block("Trade row", state.get("tradeRow") or []))
        lines.append("")
        lines.extend(self._cards_in_play_block("Opponent battlefield", state.get("opponentCardsInPlay") or {}))
        lines.append("")
        lines.extend(self._cards_block("Opponent hidden deck + hand", state.get("opponentScrambleDeckAndHand") or []))
        lines.append("")
        lines.extend(self._cards_block("Opponent known top cards", state.get("opponentTopCards") or []))
        lines.append("")
        lines.extend(self._cards_block("Opponent known hand cards", state.get("opponentHandCards") or []))
        lines.append("")
        lines.extend(self._cards_block("Opponent discard pile", state.get("opponentDiscardPile") or []))
        return "\n".join(lines)

    def _selected_index(self) -> Optional[int]:
        selection = self.option_list.curselection()
        if not selection:
            return None
        return int(selection[0])

    def _update_option_details(self, _: Optional[tk.Event] = None) -> None:
        index = self._selected_index()
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        if index is not None:
            option = self.request.options[index]
            pretty = chooser_ui._format_option(option, self.request.state)
            self.detail_text.insert("1.0", pretty + "\n\n" + repr(option))
        self.detail_text.configure(state="disabled")

    def _confirm_selection(self) -> None:
        index = self._selected_index()
        if index is None:
            messagebox.showinfo("Select an option", "Pick an option before confirming.", parent=self)
            return
        self.selection = index
        self.destroy()

    def _on_confirm(self, _: Optional[tk.Event] = None) -> None:
        self._confirm_selection()

    def _on_cancel(self, _: Optional[tk.Event] = None) -> None:
        self.selection = None
        self.destroy()


class TrainerGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Star Realms Self-Play Trainer")
        self.geometry("1620x980")
        self.minsize(1280, 760)
        self.configure(bg=WINDOW_BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.choice_queue: "queue.Queue[Any]" = queue.Queue()
        self.result_queue: "queue.Queue[tuple[str, str, Any, Optional[Callable[[Any], None]]]]" = queue.Queue()
        self.progress_queue: "queue.Queue[tuple[str, Dict[str, Any]]]" = queue.Queue()
        self.command_thread: Optional[threading.Thread] = None
        self.game_view = chooser_ui.GameChooserGUI(parent=self, title="Star Realms Game Viewer")
        self.game_view_bridge = GameViewBridge(self)
        self._suppress_run_events = False
        self._last_refresh_error: Optional[str] = None
        self._last_refresh_error_at = 0.0
        self._runtime_info = sp.runtime_environment()
        self._device_backend_cache: Dict[str, str] = {}
        self._refresh_requested = True
        self._next_refresh_at = 0.0
        self._known_run_names: tuple[str, ...] = ()
        self._last_runs_tree_key: Optional[tuple[tuple[Any, ...], ...]] = None
        self._last_runs_order_key: Optional[tuple[str, ...]] = None
        self._last_checkpoint_tree_key: Optional[tuple[Any, ...]] = None

        self.run_name_var = tk.StringVar(value=sp.LATEST_RUN_NAME)
        self.checkpoint_var = tk.StringVar(value="latest")
        self.match_run_a_var = tk.StringVar(value=sp.LATEST_RUN_NAME)
        self.match_checkpoint_a_var = tk.StringVar(value="latest")
        self.match_run_b_var = tk.StringVar(value=sp.LATEST_RUN_NAME)
        self.match_checkpoint_b_var = tk.StringVar(value="latest")
        self.match_games_var = tk.StringVar(value="24")
        self.chunk_var = tk.StringVar(value="5")
        self.new_run_model_var = tk.StringVar(value=sp.MODEL_TYPE_DEEP)
        self.new_run_device_var = tk.StringVar(value=sp.DEVICE_AUTO)
        self.new_run_workers_var = tk.StringVar(value=str(sp.SIMULATION_WORKERS_AUTO))
        self.new_run_matches_var = tk.StringVar(value="5")
        self.new_run_games_var = tk.StringVar(value="16")
        self.new_run_decisions_var = tk.StringVar(value=sp.TRAINING_DECISIONS_PER_GAME_ALL)
        self.new_run_train_temperature_var = tk.StringVar(value="0.9")
        self.new_run_promotion_games_var = tk.StringVar(value="24")
        self.new_run_promotion_threshold_var = tk.StringVar(value="0.6")
        self.fork_source_run_var = tk.StringVar(value=sp.LATEST_RUN_NAME)
        self.fork_source_checkpoint_var = tk.StringVar(value="latest")
        self.rating_models_var = tk.StringVar(value="8")
        self.rating_games_var = tk.StringVar(value="12")
        self.cross_run_games_var = tk.StringVar(value="200")
        self.include_candidate_var = tk.BooleanVar(value=True)
        self.human_name_var = tk.StringVar(value="Human")
        self.policy_name_var = tk.StringVar(value="Policy")
        self.run_train_temperature_var = tk.StringVar(value="0.9")
        self.run_promotion_threshold_var = tk.StringVar(value="0.6")
        self.run_workers_var = tk.StringVar(value=str(sp.SIMULATION_WORKERS_AUTO))
        self._loaded_run_settings_run_name: Optional[str] = None

        self.status_vars = {
            "status": tk.StringVar(value="idle"),
            "iteration": tk.StringVar(value="0"),
            "matches": tk.StringVar(value="0"),
            "games": tk.StringVar(value="0"),
            "elo": tk.StringVar(value="1000.0"),
            "best_elo": tk.StringVar(value="1000.0"),
            "best_checkpoint": tk.StringVar(value="-"),
            "latest_checkpoint": tk.StringVar(value="-"),
            "runtime": tk.StringVar(value="-"),
            "device": tk.StringVar(value="cpu"),
            "last_error": tk.StringVar(value=""),
        }
        self.summary_progress_value_vars = {
            "iteration": tk.DoubleVar(value=0.0),
            "training": tk.DoubleVar(value=0.0),
            "promotion": tk.DoubleVar(value=0.0),
        }
        self.summary_progress_text_vars = {
            "iteration": tk.StringVar(value="Idle"),
            "training": tk.StringVar(value="Idle"),
            "promotion": tk.StringVar(value="Idle"),
        }
        self.activity_progress_value_vars = {
            "acquire_elo": tk.DoubleVar(value=0.0),
            "policy_match": tk.DoubleVar(value=0.0),
            "rating_pass": tk.DoubleVar(value=0.0),
            "cross_run_rating": tk.DoubleVar(value=0.0),
        }
        self.activity_progress_text_vars = {
            "acquire_elo": tk.StringVar(value="Idle"),
            "policy_match": tk.StringVar(value="Idle"),
            "rating_pass": tk.StringVar(value="Idle"),
            "cross_run_rating": tk.StringVar(value="Idle"),
        }

        self._build_style()
        self._build_layout()
        self.after(50, self._surface_window)
        self.after(10, self._initial_refresh)
        self.after(POLL_INTERVAL_MS, self._poll)

    def _initial_refresh(self) -> None:
        try:
            self.refresh_data()
        except Exception:
            error = traceback.format_exc()
            self.log("Initial GUI refresh failed.\n" + error)
            messagebox.showerror("Startup failed", error, parent=self)

    def _surface_window(self) -> None:
        try:
            self.deiconify()
            self.lift()
            self.attributes("-topmost", True)
            self.after(300, lambda: self.attributes("-topmost", False))
            self.focus_force()
        except tk.TclError:
            pass

    def _request_refresh(self, immediate: bool = False) -> None:
        self._refresh_requested = True
        if immediate:
            self._next_refresh_at = 0.0

    def _refresh_interval_ms(self) -> int:
        if self.command_thread is not None and self.command_thread.is_alive():
            return REFRESH_INTERVAL_COMMAND_MS
        status = self.status_vars["status"].get().strip().lower()
        if status in {"training", "stop_requested"}:
            return REFRESH_INTERVAL_ACTIVE_MS
        return REFRESH_INTERVAL_IDLE_MS

    def _schedule_next_refresh(self, delay_ms: Optional[int] = None) -> None:
        interval_ms = self._refresh_interval_ms() if delay_ms is None else max(0, int(delay_ms))
        self._next_refresh_at = time.monotonic() + (interval_ms / 1000.0)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=WINDOW_BG, foreground=TEXT)
        style.configure("TFrame", background=WINDOW_BG)
        style.configure("TLabelframe", background=WINDOW_BG, foreground=TEXT)
        style.configure("TLabelframe.Label", background=WINDOW_BG, foreground=ACCENT)
        style.configure("TLabel", background=WINDOW_BG, foreground=TEXT)
        style.configure("Treeview", background="#102034", fieldbackground="#102034", foreground=TEXT, rowheight=24)
        style.configure("Treeview.Heading", background="#1a314b", foreground=TEXT)
        style.map("Treeview", background=[("selected", "#335d89")], foreground=[("selected", TEXT)])
        style.configure("TButton", background=BUTTON_BG, foreground=TEXT)
        style.map("TButton", background=[("active", BUTTON_ACTIVE)])
        style.configure("TEntry", fieldbackground="#102034", foreground=TEXT)
        style.configure("TCombobox", fieldbackground="#102034", foreground=TEXT)
        progressbar_styles = {
            "SummaryIteration.Horizontal.TProgressbar": ACCENT,
            "SummaryTraining.Horizontal.TProgressbar": "#5aa9e6",
            "SummaryPromotion.Horizontal.TProgressbar": "#7bd389",
            "ActivityAcquire.Horizontal.TProgressbar": "#f08a5d",
            "ActivityPolicy.Horizontal.TProgressbar": "#5aa9e6",
            "ActivityRating.Horizontal.TProgressbar": "#7bd389",
            "ActivityCrossRun.Horizontal.TProgressbar": "#c9a227",
        }
        for style_name, color in progressbar_styles.items():
            style.configure(
                style_name,
                troughcolor="#102034",
                background=color,
                bordercolor="#102034",
                lightcolor=color,
                darkcolor=color,
            )

    def _build_layout(self) -> None:
        top = tk.Frame(self, bg=WINDOW_BG, padx=12, pady=10)
        top.pack(fill="x")

        controls = ttk.LabelFrame(top, text="Controls", padding=10)
        controls.pack(fill="x")
        for column in range(7):
            controls.grid_columnconfigure(column, weight=1 if column in (1, 3) else 0)

        ttk.Label(controls, text="Run").grid(row=0, column=0, sticky="w")
        self.run_combo = ttk.Combobox(controls, textvariable=self.run_name_var)
        self.run_combo.grid(row=0, column=1, sticky="ew", padx=(6, 12))
        self.run_combo.bind("<<ComboboxSelected>>", self._on_run_combo)
        self.run_combo.bind("<Return>", self._on_run_combo)

        ttk.Label(controls, text="Checkpoint").grid(row=0, column=2, sticky="w")
        self.checkpoint_combo = ttk.Combobox(controls, textvariable=self.checkpoint_var, state="readonly")
        self.checkpoint_combo.grid(row=0, column=3, sticky="ew", padx=(6, 12))

        ttk.Label(controls, text="Chunk Iterations").grid(row=0, column=4, sticky="w")
        chunk_entry = ttk.Entry(controls, textvariable=self.chunk_var, width=8)
        chunk_entry.grid(row=0, column=5, sticky="w", padx=(6, 12))

        refresh_button = tk.Button(
            controls,
            text="Refresh",
            command=self.refresh_data,
            bg=BUTTON_BG,
            activebackground=BUTTON_ACTIVE,
            fg=TEXT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=12,
            pady=6,
        )
        refresh_button.grid(row=0, column=6, sticky="e")

        button_row = tk.Frame(controls, bg=WINDOW_BG)
        button_row.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(10, 0))

        self.train_chunk_button = self._button(button_row, "Train Chunk", self._train_chunk)
        self.train_chunk_button.pack(side="left", padx=(0, 6))
        self.start_button = self._button(button_row, "Start Background", self._start_background)
        self.start_button.pack(side="left", padx=6)
        self.continue_button = self._button(button_row, "Continue Background", self._continue_background)
        self.continue_button.pack(side="left", padx=6)
        self.interrupt_button = self._button(button_row, "Interrupt", self._interrupt_background)
        self.interrupt_button.pack(side="left", padx=6)
        self.create_run_button = self._button(button_row, "Create Run", self._create_run)
        self.create_run_button.pack(side="left", padx=6)
        self.create_run_from_checkpoint_button = self._button(
            button_row,
            "Create Run From Checkpoint",
            self._create_run_from_checkpoint,
        )
        self.create_run_from_checkpoint_button.pack(side="left", padx=6)
        self.rating_pass_button = self._button(button_row, "Run Rating Pass", self._run_rating_pass)
        self.rating_pass_button.pack(side="left", padx=6)
        self.card_acquire_elo_button = self._button(
            button_row,
            "Run Acquire Elo Test",
            self._run_card_acquire_elo_test,
        )
        self.card_acquire_elo_button.pack(side="left", padx=6)
        self.selfplay_button = self._button(button_row, "Run Self-Play Game", self._play_self_game)
        self.selfplay_button.pack(side="left", padx=6)
        self.human_button = self._button(button_row, "Play Against Selected Policy", self._play_human_game)
        self.human_button.pack(side="left", padx=6)

        match_row = tk.Frame(controls, bg=WINDOW_BG)
        match_row.grid(row=2, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        ttk.Label(match_row, text="Match A").pack(side="left")
        self.match_run_a_combo = ttk.Combobox(match_row, textvariable=self.match_run_a_var, width=16)
        self.match_run_a_combo.pack(side="left", padx=(6, 6))
        self.match_run_a_combo.bind("<<ComboboxSelected>>", self._on_match_run_combo)
        self.match_run_a_combo.bind("<Return>", self._on_match_run_combo)
        self.match_checkpoint_a_combo = ttk.Combobox(match_row, textvariable=self.match_checkpoint_a_var, width=16)
        self.match_checkpoint_a_combo.pack(side="left", padx=(0, 16))

        ttk.Label(match_row, text="Match B").pack(side="left")
        self.match_run_b_combo = ttk.Combobox(match_row, textvariable=self.match_run_b_var, width=16)
        self.match_run_b_combo.pack(side="left", padx=(6, 6))
        self.match_run_b_combo.bind("<<ComboboxSelected>>", self._on_match_run_combo)
        self.match_run_b_combo.bind("<Return>", self._on_match_run_combo)
        self.match_checkpoint_b_combo = ttk.Combobox(match_row, textvariable=self.match_checkpoint_b_var, width=16)
        self.match_checkpoint_b_combo.pack(side="left", padx=(0, 16))

        ttk.Label(match_row, text="Games").pack(side="left")
        ttk.Entry(match_row, textvariable=self.match_games_var, width=8).pack(side="left", padx=(6, 16))
        self.policy_match_button = self._button(match_row, "Run Policy Match", self._play_policy_match)
        self.policy_match_button.pack(side="left")

        new_run_row = tk.Frame(controls, bg=WINDOW_BG)
        new_run_row.grid(row=3, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        ttk.Label(new_run_row, text="New Run Model").pack(side="left")
        new_run_model_combo = ttk.Combobox(
            new_run_row,
            textvariable=self.new_run_model_var,
            values=(sp.MODEL_TYPE_DEEP, sp.MODEL_TYPE_DEFAULT),
            state="readonly",
            width=10,
        )
        new_run_model_combo.pack(side="left", padx=(6, 16))
        ttk.Label(new_run_row, text="Device").pack(side="left")
        new_run_device_combo = ttk.Combobox(
            new_run_row,
            textvariable=self.new_run_device_var,
            values=(sp.DEVICE_AUTO, sp.DEVICE_DIRECTML, sp.DEVICE_CPU),
            state="readonly",
            width=10,
        )
        new_run_device_combo.pack(side="left", padx=(6, 16))
        ttk.Label(new_run_row, text="Workers").pack(side="left")
        ttk.Entry(new_run_row, textvariable=self.new_run_workers_var, width=6).pack(side="left", padx=(6, 16))
        ttk.Label(new_run_row, text="Matches / Iter").pack(side="left")
        ttk.Entry(new_run_row, textvariable=self.new_run_matches_var, width=8).pack(side="left", padx=(6, 16))
        ttk.Label(new_run_row, text="Games / Match").pack(side="left")
        ttk.Entry(new_run_row, textvariable=self.new_run_games_var, width=8).pack(side="left", padx=(6, 16))
        ttk.Label(new_run_row, text="Decisions / Game").pack(side="left")
        ttk.Combobox(
            new_run_row,
            textvariable=self.new_run_decisions_var,
            values=sp.TRAINING_DECISIONS_PER_GAME_OPTIONS,
            state="readonly",
            width=6,
        ).pack(side="left", padx=(6, 16))
        ttk.Label(new_run_row, text="Train Temp").pack(side="left")
        ttk.Entry(new_run_row, textvariable=self.new_run_train_temperature_var, width=6).pack(side="left", padx=(6, 16))
        ttk.Label(new_run_row, text="Promotion Games").pack(side="left")
        ttk.Entry(new_run_row, textvariable=self.new_run_promotion_games_var, width=8).pack(side="left", padx=(6, 16))
        ttk.Label(new_run_row, text="Promote Win %").pack(side="left")
        ttk.Entry(new_run_row, textvariable=self.new_run_promotion_threshold_var, width=6).pack(side="left", padx=(6, 16))
        new_run_tip = tk.Label(
            new_run_row,
            text="These settings are used when creating a fresh run or forking from a checkpoint.",
            bg=WINDOW_BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 9),
        )
        new_run_tip.pack(side="left")

        fork_row = tk.Frame(controls, bg=WINDOW_BG)
        fork_row.grid(row=4, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        ttk.Label(fork_row, text="Fork Source Run").pack(side="left")
        self.fork_source_run_combo = ttk.Combobox(
            fork_row,
            textvariable=self.fork_source_run_var,
            width=16,
            state="readonly",
        )
        self.fork_source_run_combo.pack(side="left", padx=(6, 6))
        self.fork_source_run_combo.bind("<<ComboboxSelected>>", self._on_fork_source_combo)
        ttk.Label(fork_row, text="Source Checkpoint").pack(side="left")
        self.fork_source_checkpoint_combo = ttk.Combobox(
            fork_row,
            textvariable=self.fork_source_checkpoint_var,
            width=18,
            state="readonly",
        )
        self.fork_source_checkpoint_combo.pack(side="left", padx=(6, 16))
        fork_tip = tk.Label(
            fork_row,
            text="Creates a brand-new run whose initial weights come from the selected source checkpoint; if the model type above differs from the source, the checkpoint is converted into that architecture when supported.",
            bg=WINDOW_BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 9),
        )
        fork_tip.pack(side="left")

        run_config_row = tk.Frame(controls, bg=WINDOW_BG)
        run_config_row.grid(row=5, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        ttk.Label(run_config_row, text="Selected Run Train Temp").pack(side="left")
        ttk.Entry(run_config_row, textvariable=self.run_train_temperature_var, width=6).pack(side="left", padx=(6, 16))
        ttk.Label(run_config_row, text="Selected Run Promote Win %").pack(side="left")
        ttk.Entry(run_config_row, textvariable=self.run_promotion_threshold_var, width=6).pack(side="left", padx=(6, 16))
        ttk.Label(run_config_row, text="Workers").pack(side="left")
        ttk.Entry(run_config_row, textvariable=self.run_workers_var, width=6).pack(side="left", padx=(6, 16))
        self.apply_run_settings_button = self._button(run_config_row, "Apply Run Settings", self._apply_run_settings)
        self.apply_run_settings_button.pack(side="left", padx=(6, 12))
        tk.Label(
            run_config_row,
            text="Use 0 for automatic worker count.",
            bg=WINDOW_BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(side="left")

        rating_row = tk.Frame(controls, bg=WINDOW_BG)
        rating_row.grid(row=6, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        ttk.Label(rating_row, text="Rating Policies").pack(side="left")
        ttk.Entry(rating_row, textvariable=self.rating_models_var, width=8).pack(side="left", padx=(6, 16))
        ttk.Label(rating_row, text="Games / Pair").pack(side="left")
        ttk.Entry(rating_row, textvariable=self.rating_games_var, width=8).pack(side="left", padx=(6, 16))
        ttk.Checkbutton(rating_row, text="Include Candidate", variable=self.include_candidate_var).pack(side="left")
        ttk.Label(rating_row, text="Cross-Run Games").pack(side="left", padx=(16, 0))
        ttk.Entry(rating_row, textvariable=self.cross_run_games_var, width=8).pack(side="left", padx=(6, 8))
        self.cross_run_rating_button = self._button(rating_row, "Run Cross-Run Rating", self._run_cross_run_rating)
        self.cross_run_rating_button.pack(side="left", padx=(0, 12))
        rating_tip = tk.Label(
            rating_row,
            text="Rating pass is within-run; cross-run rating only compares each run's best policy.",
            bg=WINDOW_BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 9),
        )
        rating_tip.pack(side="left", padx=(16, 0))

        name_row = tk.Frame(controls, bg=WINDOW_BG)
        name_row.grid(row=7, column=0, columnspan=7, sticky="ew", pady=(10, 0))
        ttk.Label(name_row, text="Human Name").pack(side="left")
        ttk.Entry(name_row, textvariable=self.human_name_var, width=18).pack(side="left", padx=(6, 16))
        ttk.Label(name_row, text="Policy Display Name").pack(side="left")
        ttk.Entry(name_row, textvariable=self.policy_name_var, width=18).pack(side="left", padx=(6, 16))
        tip = tk.Label(
            name_row,
            text="Type a new run name above to create it on the first training action.",
            bg=WINDOW_BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 9),
        )
        tip.pack(side="left")

        summary = ttk.LabelFrame(self, text="Selected Run Summary", padding=10)
        summary.pack(fill="x", padx=12, pady=(0, 10))
        for column in range(5):
            summary.grid_columnconfigure(column, weight=1)

        self._summary_pair(summary, 0, 0, "Status", self.status_vars["status"])
        self._summary_pair(summary, 0, 1, "Iteration", self.status_vars["iteration"])
        self._summary_pair(summary, 0, 2, "Matches", self.status_vars["matches"])
        self._summary_pair(summary, 0, 3, "Games", self.status_vars["games"])
        self._summary_pair(summary, 0, 4, "Runtime", self.status_vars["runtime"])
        self._summary_pair(summary, 1, 0, "Current Elo", self.status_vars["elo"])
        self._summary_pair(summary, 1, 1, "Best Elo", self.status_vars["best_elo"])
        self._summary_pair(summary, 1, 2, "Best Checkpoint", self.status_vars["best_checkpoint"])
        self._summary_pair(summary, 1, 3, "Latest Checkpoint", self.status_vars["latest_checkpoint"])
        self._summary_pair(summary, 1, 4, "Last Error", self.status_vars["last_error"])
        self._summary_pair(summary, 2, 0, "Device", self.status_vars["device"])
        summary_progress = tk.Frame(summary, bg=WINDOW_BG)
        summary_progress.grid(row=3, column=0, columnspan=5, sticky="ew", padx=6, pady=(8, 2))
        for column in range(3):
            summary_progress.grid_columnconfigure(column, weight=1)
        self._summary_progress_card(
            summary_progress,
            0,
            "Iteration Progress",
            "iteration",
            "SummaryIteration.Horizontal.TProgressbar",
        )
        self._summary_progress_card(
            summary_progress,
            1,
            "Training Stage",
            "training",
            "SummaryTraining.Horizontal.TProgressbar",
        )
        self._summary_progress_card(
            summary_progress,
            2,
            "Promotion Stage",
            "promotion",
            "SummaryPromotion.Horizontal.TProgressbar",
        )

        body = tk.PanedWindow(self, orient="horizontal", sashrelief="flat", bg=WINDOW_BG, bd=0)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        runs_frame = ttk.LabelFrame(body, text="Runs", padding=8)
        checkpoints_frame = ttk.LabelFrame(body, text="Checkpoints", padding=8)
        details_frame = ttk.LabelFrame(body, text="Details And Logs", padding=8)
        body.add(runs_frame, stretch="always")
        body.add(checkpoints_frame, stretch="always")
        body.add(details_frame, stretch="always")

        self.runs_tree = self._build_tree(
            runs_frame,
            columns=("status", "created", "fork", "iteration", "elo", "best_elo", "cross_elo", "cross_conf", "matches", "games"),
            headings=("Status", "Created", "Forked From", "Iter", "Elo", "Best Elo", "Cross Elo", "+/-", "Matches", "Games"),
            selectmode="extended",
        )
        self.runs_tree.bind("<<TreeviewSelect>>", self._on_run_tree_select)
        self.runs_tree.column("created", width=145, stretch=False)
        self.runs_tree.column("fork", width=190, stretch=True)
        self.runs_tree.column("cross_elo", width=82, stretch=False)
        self.runs_tree.column("cross_conf", width=62, stretch=False)

        self.checkpoints_tree = self._build_tree(
            checkpoints_frame,
            columns=("iteration", "elo", "flags"),
            headings=("Iter", "Elo", "Flags"),
            selectmode="extended",
        )
        self.checkpoints_tree.bind("<<TreeviewSelect>>", self._on_checkpoint_tree_select)
        checkpoint_actions = tk.Frame(checkpoints_frame, bg=WINDOW_BG)
        checkpoint_actions.pack(fill="x", pady=(8, 0))
        self.delete_checkpoint_button = self._button(
            checkpoint_actions,
            "Delete Selected Checkpoints",
            self._delete_selected_checkpoints,
        )
        self.delete_checkpoint_button.pack(side="left")
        tk.Label(
            checkpoint_actions,
            text="Deletes all selected saved checkpoint rows and updates the run metadata.",
            bg=WINDOW_BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(12, 0))

        notebook = ttk.Notebook(details_frame)
        notebook.pack(fill="both", expand=True)

        summary_tab = tk.Frame(notebook, bg=WINDOW_BG)
        log_tab = tk.Frame(notebook, bg=WINDOW_BG)
        notebook.add(summary_tab, text="Run Details")
        notebook.add(log_tab, text="Activity Log")

        self.details_text = ScrolledText(
            summary_tab,
            bg="#102034",
            fg=TEXT,
            insertbackground=TEXT,
            font=("Consolas", 9),
            wrap="word",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.details_text.pack(fill="both", expand=True)

        activity_progress = tk.Frame(log_tab, bg=WINDOW_BG)
        activity_progress.pack(fill="x", pady=(0, 8))
        self._activity_progress_row(
            activity_progress,
            0,
            "Acquire Elo Test",
            "acquire_elo",
            "ActivityAcquire.Horizontal.TProgressbar",
        )
        self._activity_progress_row(
            activity_progress,
            1,
            "Policy Match",
            "policy_match",
            "ActivityPolicy.Horizontal.TProgressbar",
        )
        self._activity_progress_row(
            activity_progress,
            2,
            "Rating Pass",
            "rating_pass",
            "ActivityRating.Horizontal.TProgressbar",
        )
        self._activity_progress_row(
            activity_progress,
            3,
            "Cross-Run Rating",
            "cross_run_rating",
            "ActivityCrossRun.Horizontal.TProgressbar",
        )

        self.log_text = ScrolledText(
            log_tab,
            bg="#102034",
            fg=SUBTEXT,
            insertbackground=TEXT,
            font=("Consolas", 9),
            wrap="word",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.log_text.pack(fill="both", expand=True)
        self.log("Trainer GUI ready.")

    def _button(self, parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=BUTTON_BG,
            activebackground=BUTTON_ACTIVE,
            fg=TEXT,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=12,
            pady=6,
            font=("Segoe UI Semibold", 9),
        )

    def _build_tree(
        self,
        parent: tk.Widget,
        columns: Sequence[str],
        headings: Sequence[str],
        selectmode: str = "browse",
    ) -> ttk.Treeview:
        frame = tk.Frame(parent, bg=WINDOW_BG)
        frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(frame, columns=columns, show="tree headings", selectmode=selectmode)
        tree.pack(side="left", fill="both", expand=True)
        tree.heading("#0", text="Name")
        tree.column("#0", width=190, stretch=True)
        for column, heading in zip(columns, headings):
            tree.heading(column, text=heading)
            tree.column(column, width=90, stretch=True, anchor="center")

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)
        return tree

    def _summary_pair(self, parent: tk.Widget, row: int, column: int, label: str, var: tk.StringVar) -> None:
        wrapper = tk.Frame(parent, bg=WINDOW_BG)
        wrapper.grid(row=row, column=column, sticky="ew", padx=6, pady=4)
        tk.Label(
            wrapper,
            text=label,
            bg=WINDOW_BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(fill="x")
        tk.Label(
            wrapper,
            textvariable=var,
            bg=WINDOW_BG,
            fg=TEXT,
            anchor="w",
            font=("Segoe UI Semibold", 11),
        ).pack(fill="x")

    def _summary_progress_card(
        self,
        parent: tk.Widget,
        column: int,
        label: str,
        key: str,
        style_name: str,
    ) -> None:
        wrapper = tk.Frame(parent, bg=WINDOW_BG)
        wrapper.grid(row=0, column=column, sticky="ew", padx=6, pady=2)
        tk.Label(
            wrapper,
            text=label,
            bg=WINDOW_BG,
            fg=MUTED,
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(fill="x")
        ttk.Progressbar(
            wrapper,
            variable=self.summary_progress_value_vars[key],
            maximum=100.0,
            mode="determinate",
            style=style_name,
        ).pack(fill="x", pady=(4, 4))
        tk.Label(
            wrapper,
            textvariable=self.summary_progress_text_vars[key],
            bg=WINDOW_BG,
            fg=SUBTEXT,
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(fill="x")

    def _activity_progress_row(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        key: str,
        style_name: str,
    ) -> None:
        wrapper = tk.Frame(parent, bg=WINDOW_BG)
        wrapper.pack(fill="x", pady=(0 if row == 0 else 4, 0))
        tk.Label(
            wrapper,
            text=label,
            bg=WINDOW_BG,
            fg=ACCENT,
            anchor="w",
            font=("Segoe UI Semibold", 10),
        ).pack(fill="x")
        ttk.Progressbar(
            wrapper,
            variable=self.activity_progress_value_vars[key],
            maximum=100.0,
            mode="determinate",
            style=style_name,
        ).pack(fill="x", pady=(4, 4))
        tk.Label(
            wrapper,
            textvariable=self.activity_progress_text_vars[key],
            bg=WINDOW_BG,
            fg=SUBTEXT,
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(fill="x")

    def _progress_value(self, completed: int, total: int, *, complete: bool = False) -> float:
        if total <= 0:
            return 100.0 if complete else 0.0
        if complete:
            return 100.0
        return max(0.0, min(100.0, (100.0 * float(completed)) / float(total)))

    def _stage_startup_progress_value(
        self,
        live_progress: Dict[str, Any],
        stage_name: str,
        completed: int,
        total: int,
        *,
        complete: bool = False,
    ) -> float:
        actual = self._progress_value(completed, total, complete=complete)
        if complete or completed > 0:
            return actual
        if str(live_progress.get("stage", "")).strip().lower() != stage_name:
            return actual
        try:
            updated_at = float(live_progress.get("updated_at", time.time()))
        except (TypeError, ValueError):
            updated_at = time.time()
        elapsed = max(0.0, time.time() - updated_at)
        startup = min(
            STAGE_STARTUP_PROGRESS_CAP,
            STAGE_STARTUP_PROGRESS_CAP * elapsed / max(1.0, STAGE_STARTUP_PROGRESS_SECONDS),
        )
        return max(actual, startup)

    def _stage_label(self, stage: str) -> str:
        labels = {
            "training": "Training",
            "optimizing": "Policy Update",
            "promotion": "Promotion",
            "complete": "Complete",
            "idle": "Idle",
        }
        return labels.get(str(stage or "").strip().lower(), str(stage or "Idle"))

    def _set_summary_progress_defaults(self, detail: str = "Idle") -> None:
        for key, value_var in self.summary_progress_value_vars.items():
            value_var.set(0.0)
            self.summary_progress_text_vars[key].set(detail)

    def _set_activity_progress(self, key: str, payload: Dict[str, Any]) -> None:
        value_var = self.activity_progress_value_vars[key]
        text_var = self.activity_progress_text_vars[key]
        status = str(payload.get("status", "running"))
        label = str(payload.get("label", "")).strip()
        completed = max(0, int(payload.get("games_completed", 0)))
        total = max(0, int(payload.get("games_target", 0)))
        pairings_completed = payload.get("pairings_completed")
        pairings_total = payload.get("pairings_total")
        duration_seconds = payload.get("duration_seconds")
        duration_text = ""
        if duration_seconds is not None:
            try:
                duration_text = f" in {sp._format_seconds(float(duration_seconds))}"
            except (TypeError, ValueError):
                duration_text = ""
        if status == "idle":
            value_var.set(0.0)
            text_var.set("Idle")
            return
        if status == "failed":
            value_var.set(self._progress_value(completed, total))
            detail = f"Failed at {completed} / {total} scheduled games"
        elif status == "completed":
            value_var.set(100.0)
            detail = f"Completed {completed} / {total} scheduled games{duration_text}"
        else:
            value_var.set(self._progress_value(completed, total))
            detail = f"{completed} / {total} scheduled games{duration_text}"
        if pairings_completed is not None and pairings_total is not None:
            try:
                detail += f" | {int(pairings_completed)} / {int(pairings_total)} pairings"
            except (TypeError, ValueError):
                pass
        text_var.set(f"{label}: {detail}" if label else detail)

    def _selected_run_name(self) -> str:
        run_name = self.run_name_var.get().strip()
        return run_name or sp.LATEST_RUN_NAME

    def _selected_checkpoint(self) -> str:
        checkpoint = self.checkpoint_var.get().strip()
        return checkpoint or "latest"

    def _selected_match_games(self) -> int:
        try:
            games = int(self.match_games_var.get().strip())
        except ValueError:
            raise ValueError("Match games must be an integer.")
        if games <= 0:
            raise ValueError("Match games must be positive.")
        return games

    def _chunk_iterations(self) -> int:
        try:
            iterations = int(self.chunk_var.get().strip())
        except ValueError:
            raise ValueError("Chunk iterations must be an integer.")
        if iterations <= 0:
            raise ValueError("Chunk iterations must be positive.")
        return iterations

    def _rating_policy_count(self) -> int:
        try:
            count = int(self.rating_models_var.get().strip())
        except ValueError:
            raise ValueError("Rating policy count must be an integer.")
        if count < 2:
            raise ValueError("Rating policy count must be at least 2.")
        return count

    def _rating_games_per_pair(self) -> int:
        try:
            games = int(self.rating_games_var.get().strip())
        except ValueError:
            raise ValueError("Games per pair must be an integer.")
        if games < 2:
            raise ValueError("Games per pair must be at least 2.")
        return games

    def _cross_run_games(self) -> int:
        try:
            games = int(self.cross_run_games_var.get().strip())
        except ValueError:
            raise ValueError("Cross-run games must be an integer.")
        if games <= 0:
            raise ValueError("Cross-run games must be positive.")
        return games

    def _new_run_training_matches(self) -> int:
        try:
            value = int(self.new_run_matches_var.get().strip())
        except ValueError:
            raise ValueError("New run matches per iteration must be an integer.")
        if value <= 0:
            raise ValueError("New run matches per iteration must be positive.")
        return value

    def _new_run_games_per_match(self) -> int:
        try:
            value = int(self.new_run_games_var.get().strip())
        except ValueError:
            raise ValueError("New run games per match must be an integer.")
        if value <= 0:
            raise ValueError("New run games per match must be positive.")
        return value

    def _new_run_decisions_per_game(self) -> str:
        try:
            return sp.normalize_training_decisions_per_game(self.new_run_decisions_var.get().strip())
        except ValueError as exc:
            raise ValueError(str(exc))

    def _new_run_train_temperature(self) -> float:
        try:
            return sp.normalize_train_temperature(self.new_run_train_temperature_var.get().strip())
        except ValueError as exc:
            raise ValueError(str(exc))

    def _new_run_promotion_games(self) -> int:
        try:
            value = int(self.new_run_promotion_games_var.get().strip())
        except ValueError:
            raise ValueError("New run promotion games must be an integer.")
        if value <= 0:
            raise ValueError("New run promotion games must be positive.")
        return value

    def _new_run_promotion_threshold(self) -> float:
        try:
            return sp.normalize_promotion_score_threshold(self.new_run_promotion_threshold_var.get().strip())
        except ValueError as exc:
            raise ValueError(str(exc))

    def _new_run_simulation_workers(self) -> int:
        try:
            return sp.normalize_simulation_workers(self.new_run_workers_var.get().strip())
        except ValueError as exc:
            raise ValueError(str(exc))

    def _selected_run_train_temperature(self) -> float:
        try:
            return sp.normalize_train_temperature(self.run_train_temperature_var.get().strip())
        except ValueError as exc:
            raise ValueError(str(exc))

    def _selected_run_promotion_threshold(self) -> float:
        try:
            return sp.normalize_promotion_score_threshold(self.run_promotion_threshold_var.get().strip())
        except ValueError as exc:
            raise ValueError(str(exc))

    def _selected_run_simulation_workers(self) -> int:
        try:
            return sp.normalize_simulation_workers(self.run_workers_var.get().strip())
        except ValueError as exc:
            raise ValueError(str(exc))

    def _selected_new_run_overrides(self) -> Dict[str, Any]:
        return sp.new_run_overrides(
            model_type=self.new_run_model_var.get().strip() or sp.MODEL_TYPE_DEEP,
            device_preference=self.new_run_device_var.get().strip() or sp.DEVICE_AUTO,
            simulation_workers=self._new_run_simulation_workers(),
            training_matches_per_iteration=self._new_run_training_matches(),
            training_games_per_match=self._new_run_games_per_match(),
            training_decisions_per_game=self._new_run_decisions_per_game(),
            train_temperature=self._new_run_train_temperature(),
            promotion_games=self._new_run_promotion_games(),
            promotion_score_threshold=self._new_run_promotion_threshold(),
        )

    def _new_run_overrides_for(self, run_name: str) -> Optional[Dict[str, Any]]:
        if sp.run_exists(run_name):
            return None
        return self._selected_new_run_overrides()

    def _run_async(
        self,
        label: str,
        func: Callable[[], Any],
        on_success: Optional[Callable[[Any], None]] = None,
    ) -> None:
        if self.command_thread is not None and self.command_thread.is_alive():
            messagebox.showinfo("Busy", "Another command is still running. Wait for it to finish first.", parent=self)
            return

        self.log(f"{label} started.")

        def worker() -> None:
            try:
                result = func()
                self.result_queue.put(("success", label, result, on_success))
            except Exception:
                self.result_queue.put(("error", label, traceback.format_exc(), on_success))

        self.command_thread = threading.Thread(target=worker, name="trainer-gui-command", daemon=True)
        self.command_thread.start()

    def _has_active_command(self) -> bool:
        return self.command_thread is not None and self.command_thread.is_alive()

    def _checkpoint_values_for_run(
        self,
        run_name: str,
        run_names: set[str],
        checkpoint_cache: Dict[str, List[Dict[str, Any]]],
    ) -> tuple[str, ...]:
        checkpoint_values = ["latest", "best"]
        if run_name in run_names:
            checkpoints = checkpoint_cache.get(run_name)
            if checkpoints is None:
                checkpoints = sp.list_checkpoints(run_name)
                checkpoint_cache[run_name] = checkpoints
            checkpoint_values.extend(item["name"] for item in checkpoints)
        return tuple(checkpoint_values)

    def _refresh_match_policy_selectors(
        self,
        run_names: Sequence[str],
        run_name_set: Optional[set[str]] = None,
        checkpoint_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        values = tuple(run_names)
        if tuple(self.match_run_a_combo["values"]) != values:
            self.match_run_a_combo.configure(values=values)
        if tuple(self.match_run_b_combo["values"]) != values:
            self.match_run_b_combo.configure(values=values)

        if run_name_set is None:
            run_name_set = set(run_names)
        if checkpoint_cache is None:
            checkpoint_cache = {}

        if not self.match_run_a_var.get().strip():
            self.match_run_a_var.set(self._selected_run_name())
        if not self.match_run_b_var.get().strip():
            self.match_run_b_var.set(self._selected_run_name())

        for run_var, checkpoint_var, checkpoint_combo in [
            (self.match_run_a_var, self.match_checkpoint_a_var, self.match_checkpoint_a_combo),
            (self.match_run_b_var, self.match_checkpoint_b_var, self.match_checkpoint_b_combo),
        ]:
            run_name = run_var.get().strip() or sp.LATEST_RUN_NAME
            checkpoint_values = self._checkpoint_values_for_run(run_name, run_name_set, checkpoint_cache)
            if tuple(checkpoint_combo["values"]) != checkpoint_values:
                checkpoint_combo.configure(values=checkpoint_values)
            if checkpoint_var.get().strip() not in checkpoint_values:
                checkpoint_var.set("latest")

    def _refresh_fork_source_selector(
        self,
        run_names: Sequence[str],
        run_name_set: Optional[set[str]] = None,
        checkpoint_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        values = tuple(run_names)
        if tuple(self.fork_source_run_combo["values"]) != values:
            self.fork_source_run_combo.configure(values=values)

        if run_name_set is None:
            run_name_set = set(run_names)
        if checkpoint_cache is None:
            checkpoint_cache = {}

        if not self.fork_source_run_var.get().strip():
            self.fork_source_run_var.set(self._selected_run_name())
        if values and self.fork_source_run_var.get().strip() not in values:
            self.fork_source_run_var.set(values[0])

        run_name = self.fork_source_run_var.get().strip() or sp.LATEST_RUN_NAME
        checkpoint_values = self._checkpoint_values_for_run(run_name, run_name_set, checkpoint_cache)
        if tuple(self.fork_source_checkpoint_combo["values"]) != checkpoint_values:
            self.fork_source_checkpoint_combo.configure(values=checkpoint_values)
        if self.fork_source_checkpoint_var.get().strip() not in checkpoint_values:
            self.fork_source_checkpoint_var.set("latest")

    def _load_selected_run_settings(self, run_name: str, state: Dict[str, Any]) -> None:
        if self._loaded_run_settings_run_name == run_name:
            return
        config = state.get("config") or {}
        if config:
            self.run_train_temperature_var.set(str(config.get("train_temperature", 0.9)))
            self.run_promotion_threshold_var.set(str(config.get("promotion_score_threshold", 0.6)))
            self.run_workers_var.set(str(config.get("simulation_workers", sp.SIMULATION_WORKERS_AUTO)))
        else:
            self.run_train_temperature_var.set("0.9")
            self.run_promotion_threshold_var.set("0.6")
            self.run_workers_var.set(str(sp.SIMULATION_WORKERS_AUTO))
        self._loaded_run_settings_run_name = run_name

    def refresh_data(self) -> None:
        runs = sp.list_runs()
        run_names = [item["run_name"] for item in runs]
        run_name_set = set(run_names)
        checkpoint_cache: Dict[str, List[Dict[str, Any]]] = {}
        try:
            cross_run_entries = list(sp.cross_run_rating_summary().get("leaderboard") or [])
            cross_run_by_name = {str(item.get("run_name")): item for item in cross_run_entries}
        except Exception:
            cross_run_by_name = {}

        def cross_run_table_values(run_name: str) -> tuple[str, str]:
            entry = cross_run_by_name.get(run_name) or {}
            elo = entry.get("elo")
            confidence = entry.get("confidence_radius")
            elo_text = "-" if elo is None else f"{float(elo):.1f}"
            confidence_text = "-" if confidence is None else f"{float(confidence):.1f}"
            return elo_text, confidence_text

        self._known_run_names = tuple(run_names)

        if tuple(self.run_combo["values"]) != self._known_run_names:
            self.run_combo.configure(values=self._known_run_names)
        self._refresh_match_policy_selectors(self._known_run_names, run_name_set, checkpoint_cache)
        self._refresh_fork_source_selector(self._known_run_names, run_name_set, checkpoint_cache)

        current_run = self._selected_run_name()
        typed_new_run = bool(current_run) and current_run not in run_names
        selected_run = current_run if current_run else (run_names[0] if run_names else sp.LATEST_RUN_NAME)
        self.run_name_var.set(selected_run)

        current_run_selection = None
        selected_items = self.runs_tree.selection()
        if selected_items:
            focus = self.runs_tree.focus()
            current_run_selection = focus if focus in selected_items else selected_items[-1]

        run_rows = []
        for run in runs:
            run_name = str(run["run_name"])
            values = (
                run["status"],
                run.get("created_datetime") or "-",
                run.get("fork_origin") or "-",
                run["iteration"],
                f"{run['current_elo']:.1f}",
                f"{run['best_elo']:.1f}",
                *cross_run_table_values(run_name),
                run["total_matches"],
                run["total_games"],
            )
            run_rows.append(
                (
                    run_name,
                    values,
                    (
                        run_name,
                        str(run["status"]),
                        str(run.get("created_datetime") or ""),
                        str(run.get("fork_origin") or "-"),
                        str(run.get("last_promotion_at") or ""),
                        int(run["iteration"]),
                        f"{float(run['current_elo']):.1f}",
                        f"{float(run['best_elo']):.1f}",
                        *cross_run_table_values(run_name),
                        int(run["total_matches"]),
                        int(run["total_games"]),
                    ),
                )
            )
        runs_tree_key = tuple(row_key for _, _, row_key in run_rows)
        runs_order_key = tuple(run_name for run_name, _, _ in run_rows)

        self._suppress_run_events = True
        try:
            if runs_order_key != self._last_runs_order_key:
                self.runs_tree.delete(*self.runs_tree.get_children())
                for run_name, values, _ in run_rows:
                    self.runs_tree.insert(
                        "",
                        "end",
                        iid=run_name,
                        text=run_name,
                        values=values,
                    )
                self._last_runs_order_key = runs_order_key
            elif runs_tree_key != self._last_runs_tree_key:
                for run_name, values, _ in run_rows:
                    if run_name in self.runs_tree.get_children():
                        self.runs_tree.item(run_name, text=run_name, values=values)
            self._last_runs_tree_key = runs_tree_key

            if typed_new_run:
                self.runs_tree.selection_remove(self.runs_tree.selection())
                self.runs_tree.focus("")
            elif selected_run in self.runs_tree.get_children():
                surviving_selection = [
                    item
                    for item in selected_items
                    if item in self.runs_tree.get_children()
                ]
                if surviving_selection:
                    if tuple(self.runs_tree.selection()) != tuple(surviving_selection):
                        self.runs_tree.selection_set(surviving_selection)
                    focus_run = (
                        current_run_selection
                        if current_run_selection in surviving_selection
                        else surviving_selection[-1]
                    )
                    self.runs_tree.focus(focus_run)
                    self.runs_tree.see(focus_run)
                else:
                    self.runs_tree.selection_set(selected_run)
                    self.runs_tree.focus(selected_run)
                    self.runs_tree.see(selected_run)
            elif current_run_selection in self.runs_tree.get_children():
                if tuple(self.runs_tree.selection()) != (current_run_selection,):
                    self.runs_tree.selection_set(current_run_selection)
        finally:
            self._suppress_run_events = False

        selected_state: Dict[str, Any] = {}
        selected_checkpoints: List[Dict[str, Any]] = []
        if selected_run in run_name_set:
            selected_state = sp.get_run_state(selected_run)
            selected_checkpoints = checkpoint_cache.get(selected_run)
            if selected_checkpoints is None:
                selected_checkpoints = sp.list_checkpoints(selected_run)
                checkpoint_cache[selected_run] = selected_checkpoints

        self._refresh_checkpoint_selector(selected_run, selected_checkpoints)
        self._refresh_summary(
            selected_run,
            run_names=self._known_run_names,
            state=selected_state,
            checkpoints=selected_checkpoints,
        )
        self._refresh_requested = False
        self._schedule_next_refresh()

    def _refresh_checkpoint_selector(self, run_name: str, checkpoints: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        checkpoints = list(checkpoints or [])
        checkpoint_values = ["latest", "best"] + [item["name"] for item in checkpoints]
        if tuple(self.checkpoint_combo["values"]) != tuple(checkpoint_values):
            self.checkpoint_combo.configure(values=checkpoint_values)

        current_checkpoint = self._selected_checkpoint()
        if current_checkpoint not in checkpoint_values:
            current_checkpoint = "latest"
        self.checkpoint_var.set(current_checkpoint)

        selected_items = list(self.checkpoints_tree.selection())
        selected_checkpoint = selected_items[-1] if selected_items else None

        checkpoint_tree_key = (
            run_name,
            tuple(
                (
                    checkpoint["name"],
                    int(checkpoint.get("iteration", 0)),
                    f"{float(checkpoint.get('elo', sp.INITIAL_ELO)):.1f}",
                    bool(checkpoint.get("is_latest")),
                    bool(checkpoint.get("is_best")),
                    bool(checkpoint.get("is_candidate")),
                )
                for checkpoint in checkpoints
            ),
        )
        if checkpoint_tree_key != self._last_checkpoint_tree_key:
            self.checkpoints_tree.delete(*self.checkpoints_tree.get_children())
            for checkpoint in checkpoints:
                flags = []
                if checkpoint.get("is_latest"):
                    flags.append("latest")
                if checkpoint.get("is_best"):
                    flags.append("best")
                if checkpoint.get("is_candidate"):
                    flags.append("candidate")
                self.checkpoints_tree.insert(
                    "",
                    "end",
                    iid=checkpoint["name"],
                    text=checkpoint["name"],
                    values=(
                        checkpoint.get("iteration", 0),
                        f"{float(checkpoint.get('elo', sp.INITIAL_ELO)):.1f}",
                        ", ".join(flags) or "-",
                    ),
                )
            self._last_checkpoint_tree_key = checkpoint_tree_key

        surviving_selection = [
            checkpoint_name
            for checkpoint_name in selected_items
            if checkpoint_name in self.checkpoints_tree.get_children()
        ]

        if surviving_selection:
            self.checkpoints_tree.selection_set(surviving_selection)
            focus_checkpoint = surviving_selection[-1]
            self.checkpoints_tree.focus(focus_checkpoint)
            self.checkpoints_tree.see(focus_checkpoint)
        elif current_checkpoint not in ("latest", "best") and current_checkpoint in self.checkpoints_tree.get_children():
            self.checkpoints_tree.selection_set(current_checkpoint)
            self.checkpoints_tree.focus(current_checkpoint)
            self.checkpoints_tree.see(current_checkpoint)
        elif selected_checkpoint in self.checkpoints_tree.get_children():
            self.checkpoints_tree.selection_set(selected_checkpoint)

    def _refresh_summary_progress(self, summary: Dict[str, Any], state: Dict[str, Any]) -> None:
        config = state.get("config") or {}
        training_matches = max(1, int(config.get("training_matches_per_iteration", 1) or 1))
        training_games_per_match = max(1, int(config.get("training_games_per_match", 1) or 1))
        promotion_games = max(1, int(config.get("promotion_games", 1) or 1))
        training_total = training_matches * training_games_per_match
        iteration_total = training_total + promotion_games
        live_progress = summary.get("live_progress") or state.get("live_progress") or {}
        if live_progress.get("kind") == "training_iteration":
            stage = self._stage_label(str(live_progress.get("stage", "idle")))
            iteration_number = int(live_progress.get("iteration_number", int(summary.get("iteration", 0)) + 1))
            training_completed = max(0, int(live_progress.get("training_games_completed", 0)))
            promotion_completed = max(0, int(live_progress.get("promotion_games_completed", 0)))
            iteration_completed = max(
                0,
                int(live_progress.get("iteration_games_completed", training_completed + promotion_completed)),
            )
            matches_completed = max(0, int(live_progress.get("training_matches_completed", 0)))
            training_total = max(1, int(live_progress.get("training_games_total", training_total)))
            promotion_games = max(1, int(live_progress.get("promotion_games_total", promotion_games)))
            iteration_total = max(1, int(live_progress.get("iteration_games_total", training_total + promotion_games)))
            training_complete = bool(live_progress.get("training_stage_complete"))
            promotion_complete = bool(live_progress.get("promotion_stage_complete"))
            training_value = self._stage_startup_progress_value(
                live_progress,
                "training",
                training_completed,
                training_total,
                complete=training_complete,
            )
            promotion_value = self._stage_startup_progress_value(
                live_progress,
                "promotion",
                promotion_completed,
                promotion_games,
                complete=promotion_complete,
            )
            visual_training_completed = training_total * training_value / 100.0
            visual_promotion_completed = promotion_games * promotion_value / 100.0
            visual_iteration_completed = max(
                float(iteration_completed),
                min(float(iteration_total), visual_training_completed + visual_promotion_completed),
            )
            self.summary_progress_value_vars["iteration"].set(
                100.0
                if bool(live_progress.get("iteration_complete"))
                else max(0.0, min(100.0, (100.0 * visual_iteration_completed) / float(iteration_total)))
            )
            self.summary_progress_text_vars["iteration"].set(
                f"Iteration {iteration_number}: {iteration_completed} / {iteration_total} scheduled games | {stage}"
            )
            self.summary_progress_value_vars["training"].set(training_value)
            self.summary_progress_text_vars["training"].set(
                f"{training_completed} / {training_total} games | {matches_completed} / {training_matches} matches"
                + (
                    " | starting workers"
                    if str(live_progress.get("stage", "")).strip().lower() == "training"
                    and training_completed == 0
                    and not training_complete
                    and training_value > 0.0
                    else ""
                )
                + (" | complete" if training_complete else "")
            )
            self.summary_progress_value_vars["promotion"].set(promotion_value)
            self.summary_progress_text_vars["promotion"].set(
                f"{promotion_completed} / {promotion_games} games"
                + (
                    " | starting workers"
                    if str(live_progress.get("stage", "")).strip().lower() == "promotion"
                    and promotion_completed == 0
                    and not promotion_complete
                    and promotion_value > 0.0
                    else ""
                )
                + (" | complete" if promotion_complete else f" | {stage}")
            )
            return

        last_match = summary.get("last_match") or {}
        last_eval = summary.get("last_eval") or {}
        if int(summary.get("iteration", 0)) > 0 and last_eval:
            training_completed = max(0, int(last_match.get("games_played", 0)))
            promotion_completed = max(0, int(last_eval.get("games_played", 0)))
            self.summary_progress_value_vars["iteration"].set(100.0)
            self.summary_progress_text_vars["iteration"].set(
                f"Last completed iteration: {training_completed + promotion_completed} / {iteration_total} scheduled games"
            )
            self.summary_progress_value_vars["training"].set(100.0)
            self.summary_progress_text_vars["training"].set(
                f"{training_completed} / {training_total} games | {training_matches} / {training_matches} matches | complete"
            )
            self.summary_progress_value_vars["promotion"].set(100.0)
            self.summary_progress_text_vars["promotion"].set(
                f"{promotion_completed} / {promotion_games} games | complete"
            )
            return

        self._set_summary_progress_defaults("Not running")

    def _refresh_summary(
        self,
        run_name: str,
        run_names: Optional[Sequence[str]] = None,
        state: Optional[Dict[str, Any]] = None,
        checkpoints: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        run_name_set = set(run_names or self._known_run_names)
        if run_name not in run_name_set:
            for key, value in self.status_vars.items():
                value.set("" if key == "last_error" else "-")
            self.status_vars["status"].set("new run")
            self.status_vars["last_error"].set("")
            self._set_summary_progress_defaults("New run")
            self._write_details(
                "\n".join(
                    [
                        f"Run '{run_name}' does not exist yet.",
                        "",
                        "It will be created automatically the first time you train or start background training.",
                    ]
                )
            )
            return

        if state is None:
            state = sp.get_run_state(run_name)
        summary = self._summary_from_state(run_name, state)
        self._load_selected_run_settings(run_name, state)

        self.status_vars["status"].set(str(summary.get("status", "idle")))
        self.status_vars["iteration"].set(str(summary.get("iteration", 0)))
        self.status_vars["matches"].set(str(summary.get("total_matches", 0)))
        self.status_vars["games"].set(str(summary.get("total_games", 0)))
        self.status_vars["elo"].set(str(summary.get("current_elo", sp.INITIAL_ELO)))
        self.status_vars["best_elo"].set(str(summary.get("best_elo", sp.INITIAL_ELO)))
        self.status_vars["best_checkpoint"].set(str(summary.get("best_checkpoint", "-")))
        self.status_vars["latest_checkpoint"].set(str(summary.get("latest_checkpoint", "-")))
        self.status_vars["runtime"].set(str(summary.get("runtime", "-")))
        self.status_vars["device"].set(str(summary.get("device_backend", "cpu")))
        self.status_vars["last_error"].set(str(summary.get("last_error") or ""))
        self._refresh_summary_progress(summary, state)

        checkpoints = list(checkpoints or [])
        checkpoint_lines = []
        for checkpoint in checkpoints:
            flags = []
            if checkpoint.get("is_latest"):
                flags.append("latest")
            if checkpoint.get("is_best"):
                flags.append("best")
            checkpoint_lines.append(
                f"- {checkpoint['name']}: iter {checkpoint.get('iteration', 0)}, "
                f"elo {float(checkpoint.get('elo', sp.INITIAL_ELO)):.1f}"
                + (f" ({', '.join(flags)})" if flags else "")
            )

        config = state.get("config") or {}
        config_lines = [f"- {key}: {config[key]}" for key in sorted(config.keys())]
        runtime_info = self._runtime_info

        last_match = summary.get("last_match") or {}
        last_update = summary.get("last_update") or {}
        last_eval = summary.get("last_eval") or {}
        last_rating_pass = summary.get("last_rating_pass") or {}
        candidate = summary.get("candidate") or {}
        forked_from = state.get("forked_from") or {}
        cross_run_summary: Dict[str, Any] = {}
        cross_run_entry: Dict[str, Any] = {}
        try:
            cross_run_summary = sp.cross_run_rating_summary()
            for item in list(cross_run_summary.get("leaderboard") or []):
                if item.get("run_name") == run_name:
                    cross_run_entry = dict(item)
                    break
        except Exception as exc:
            cross_run_summary = {"status": f"unavailable: {type(exc).__name__}: {exc}"}
        cross_confidence = cross_run_entry.get("confidence_radius")
        cross_confidence_text = "-" if cross_confidence is None else f"+/- {float(cross_confidence):.1f}"
        cross_elo = cross_run_entry.get("elo")
        cross_elo_text = "-" if cross_elo is None else f"{float(cross_elo):.1f}"
        details = [
            f"Run: {run_name}",
            f"Directory: {summary.get('run_dir', '-')}",
            f"Created: {summary.get('created_datetime') or state.get('created_datetime') or '-'}",
            f"Forked from: {summary.get('fork_origin') or state.get('fork_origin') or sp._fork_origin_label(forked_from)}",
            "",
            "Progress",
            f"- Status: {summary.get('status')}",
            f"- Iteration: {summary.get('iteration')}",
            f"- Matches: {summary.get('total_matches')}",
            f"- Games: {summary.get('total_games')}",
            f"- Current Elo: {summary.get('current_elo')}",
            f"- Best Elo: {summary.get('best_elo')}",
            f"- Best checkpoint: {summary.get('best_checkpoint')}",
            f"- Latest checkpoint: {summary.get('latest_checkpoint')}",
            f"- Promotions: {summary.get('promotions', 0)}",
            f"- Last promotion iteration: {state.get('last_promotion_iteration', 0)}",
            f"- Last promotion time: {summary.get('last_promotion_datetime') or '-'}",
            f"- Runtime: {summary.get('runtime')}",
            f"- Device backend: {summary.get('device_backend', '-')}",
            f"- Device repr: {summary.get('device_repr', '-')}",
            f"- Device preference: {summary.get('device_preference', '-')}",
            f"- Device requested backend: {summary.get('device_requested_backend', '-')}",
            f"- Device reason: {summary.get('device_reason', '-')}",
            f"- Simulation workers setting: {summary.get('simulation_workers', '-')}",
            f"- Resolved simulation workers: {summary.get('resolved_simulation_workers', '-')}",
            f"- Active simulation runs: {', '.join(summary.get('active_simulation_runs') or []) or '-'}",
            f"- Logical processors: {summary.get('logical_processors', '-')}",
            f"- Python: {runtime_info.get('python_version', '-')}",
            f"- Interpreter: {runtime_info.get('interpreter', '-')}",
            "",
            "Last Match",
            f"- Wins: {last_match.get('wins', '-')}",
            f"- Losses: {last_match.get('losses', '-')}",
            f"- Games played: {last_match.get('games_played', '-')}",
            f"- Return value: {last_match.get('return_value', '-')}",
            f"- Duration seconds: {last_match.get('duration_seconds', '-')}",
            "",
            "Last Update",
            f"- Samples: {last_update.get('samples', '-')}",
            f"- Policy loss: {last_update.get('policy_loss', '-')}",
            f"- Value loss: {last_update.get('value_loss', '-')}",
            f"- Clip fraction: {last_update.get('clip_fraction', '-')}",
            f"- Average ratio: {last_update.get('avg_ratio', '-')}",
            f"- Average value prediction: {last_update.get('avg_value_prediction', '-')}",
            f"- Learning rate: {last_update.get('learning_rate', '-')}",
            f"- Epsilon random: {last_update.get('epsilon_random', '-')}",
            f"- Train temperature: {last_update.get('train_temperature', '-')}",
            f"- PPO clip: {last_update.get('ppo_clip', '-')}",
            f"- Simulation workers: {last_update.get('simulation_workers', '-')}",
            f"- Resolved simulation workers: {last_update.get('resolved_simulation_workers', '-')}",
            f"- Active simulation runs: {', '.join(last_update.get('active_simulation_runs') or []) or '-'}",
            f"- Iterations since promotion: {last_update.get('iterations_since_promotion', '-')}",
            f"- Promotion drought progress: {last_update.get('promotion_drought_progress', '-')}",
            f"- Learning-rate multiplier: {last_update.get('learning_rate_multiplier', '-')}",
            f"- Epsilon multiplier: {last_update.get('epsilon_multiplier', '-')}",
            f"- Temperature multiplier: {last_update.get('temperature_multiplier', '-')}",
            "",
            "Last Eval",
            f"- Wins: {last_eval.get('wins', '-')}",
            f"- Losses: {last_eval.get('losses', '-')}",
            f"- Games played: {last_eval.get('games_played', '-')}",
            f"- Score: {last_eval.get('score', '-')}",
            f"- Action: {last_eval.get('action', '-')}",
            f"- Promoted: {last_eval.get('promoted', '-')}",
            f"- Promoted checkpoint: {last_eval.get('promoted_checkpoint', '-')}",
            "",
            "Last Rating Pass",
            f"- Status: {last_rating_pass.get('status', '-')}",
            f"- Participant count: {last_rating_pass.get('participant_count', '-')}",
            f"- Pairings played: {last_rating_pass.get('pairings_played', '-')}",
            f"- Pairings total: {last_rating_pass.get('pairings_total', '-')}",
            f"- Games completed: {last_rating_pass.get('games_completed', '-')}",
            f"- Games target: {last_rating_pass.get('games_target', '-')}",
            f"- Games per pair: {last_rating_pass.get('games_per_pair', '-')}",
            f"- Included candidate: {last_rating_pass.get('include_candidate', '-')}",
            f"- Duration seconds: {last_rating_pass.get('duration_seconds', '-')}",
            "",
            "Cross-Run Rating",
            f"- Status: {cross_run_summary.get('status', '-')}",
            f"- Elo: {cross_elo_text}",
            f"- Confidence: {cross_confidence_text}",
            f"- Rank: {cross_run_entry.get('rank', '-')}",
            f"- Games: {cross_run_entry.get('games', '-')}",
            f"- Rated best checkpoint: {cross_run_entry.get('best_checkpoint', '-')}",
            f"- Total cross-run games: {cross_run_summary.get('total_games', '-')}",
            f"- Confidence level: {cross_run_summary.get('confidence_level', '-')}",
            "",
            "Candidate",
            f"- Base checkpoint: {candidate.get('base_checkpoint', '-')}",
            f"- Attempts since reset: {candidate.get('attempts_since_reset', '-')}",
            f"- Total attempts: {candidate.get('total_attempts', '-')}",
            f"- Resets: {candidate.get('resets', '-')}",
            f"- Promotions: {candidate.get('promotions', '-')}",
            f"- Rating pass Elo: {candidate.get('rating_pass_elo', '-')}",
            f"- Last score: {candidate.get('last_score', '-')}",
            f"- Last result: {candidate.get('last_result', '-')}",
            f"- Last reset reason: {candidate.get('last_reset_reason', '-')}",
            "",
            "Origin",
            f"- Source run: {forked_from.get('run_name', '-')}",
            f"- Source checkpoint: {forked_from.get('checkpoint', '-')}",
            f"- Resolved source checkpoint: {forked_from.get('resolved_checkpoint', '-')}",
            f"- Source architecture: {forked_from.get('source_architecture', '-')}",
            f"- Source hidden size: {forked_from.get('source_hidden_size', '-')}",
            f"- Target architecture: {forked_from.get('target_architecture', '-')}",
            f"- Target hidden size: {forked_from.get('target_hidden_size', '-')}",
            f"- Requested model type: {forked_from.get('requested_model_type', '-')}",
            f"- Conversion mode: {forked_from.get('conversion_mode', '-')}",
            f"- Fork created: {sp._format_timestamp(forked_from.get('created_at')) or '-'}",
            "",
            "Config",
            *(config_lines or ["- No saved config"]),
            "",
            "Checkpoints",
            *(checkpoint_lines or ["- No checkpoints yet"]),
        ]
        leaderboard = list(last_rating_pass.get("leaderboard") or [])
        if leaderboard:
            details.extend(
                [
                    "",
                    "Rating Pass Leaderboard",
                    *[
                        f"- {item.get('name', '-')}: elo {round(float(item.get('elo', sp.INITIAL_ELO)), 2)}, "
                        f"iter {item.get('iteration', '-')}, kind {item.get('kind', '-')}"
                        for item in leaderboard
                    ],
                ]
            )
        cross_run_leaderboard = list(cross_run_summary.get("leaderboard") or [])
        if cross_run_leaderboard:
            details.extend(
                [
                    "",
                    "Cross-Run Leaderboard",
                    *[
                        f"- #{item.get('rank', '-')} {item.get('run_name', '-')}: elo "
                        f"{float(item.get('elo', sp.INITIAL_ELO)):.1f}, "
                        f"confidence "
                        f"{'-' if item.get('confidence_radius') is None else '+/- ' + format(float(item.get('confidence_radius')), '.1f')}, "
                        f"games {item.get('games', 0)}, best {item.get('best_checkpoint', '-')}"
                        for item in cross_run_leaderboard[:12]
                    ],
                ]
            )
        device_benchmark = summary.get("device_benchmark") or {}
        if device_benchmark:
            details.extend(
                [
                    "",
                    "Device Benchmark",
                    f"- Preferred backend: {device_benchmark.get('preferred_backend', '-')}",
                    f"- CPU inference ms: {device_benchmark.get('cpu_inference_ms', '-')}",
                    f"- DirectML inference ms: {device_benchmark.get('directml_inference_ms', '-')}",
                    f"- CPU training ms: {device_benchmark.get('cpu_training_ms', '-')}",
                    f"- DirectML training ms: {device_benchmark.get('directml_training_ms', '-')}",
                    f"- Benchmark sample count: {device_benchmark.get('sample_count', '-')}",
                    f"- Benchmark minibatch size: {device_benchmark.get('minibatch_size', '-')}",
                ]
            )
        if summary.get("last_error"):
            details.extend(["", "Last Error", str(summary.get("last_error"))])

        self._write_details("\n".join(details))

    def _summary_from_state(self, run_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        if not state:
            return {
                "run_name": run_name,
                "status": "idle",
                "iteration": 0,
                "total_matches": 0,
                "total_games": 0,
                "current_elo": sp.INITIAL_ELO,
                "best_elo": sp.INITIAL_ELO,
                "best_checkpoint": "-",
                "latest_checkpoint": "-",
                "last_match": None,
                "last_update": None,
                "last_eval": None,
                "last_rating_pass": None,
                "live_progress": None,
                "last_error": None,
                "created_at": None,
                "created_datetime": "-",
                "last_promotion_at": None,
                "last_promotion_datetime": "",
                "forked_from": None,
                "fork_origin": "-",
                "runtime": "-",
                "device_backend": "cpu",
                "device_repr": "cpu",
                "device_preference": "auto",
                "device_requested_backend": "auto",
                "device_reason": "",
                "device_benchmark": None,
                "simulation_workers": sp.SIMULATION_WORKERS_AUTO,
                "resolved_simulation_workers": sp.default_simulation_workers(),
                "logical_processors": self._runtime_info.get("cpu_count", 1),
                "run_dir": str((Path(sp.RUNS_DIR) / run_name).resolve()),
            }

        created_at = state.get("created_at")
        runtime = "-"
        if created_at is not None:
            runtime = sp._format_seconds(sp._timestamp() - float(created_at))
        config = state.get("config", {}) or {}
        device_preference = str(config.get("device_preference", "auto"))
        device_plan = dict(state.get("device_plan") or {})
        device_backend = device_plan.get("backend")
        if not device_backend:
            device_backend = self._device_backend_cache.get(device_preference)
            if device_backend is None:
                try:
                    device_backend = sp.resolve_device_backend(device_preference)
                except Exception:
                    device_backend = device_preference
                self._device_backend_cache[device_preference] = device_backend
        return {
            "run_name": state.get("run_name", run_name),
            "status": state.get("status", "idle"),
            "created_at": state.get("created_at"),
            "created_datetime": state.get("created_datetime") or sp._format_timestamp(state.get("created_at")),
            "last_promotion_at": state.get("last_promotion_at"),
            "last_promotion_datetime": state.get("last_promotion_datetime")
            or sp._format_timestamp(state.get("last_promotion_at")),
            "forked_from": state.get("forked_from"),
            "fork_origin": state.get("fork_origin") or sp._fork_origin_label(state.get("forked_from")),
            "iteration": int(state.get("iteration", 0)),
            "total_matches": int(state.get("total_matches", 0)),
            "total_games": int(state.get("total_games", 0)),
            "current_elo": round(float(state.get("current_elo", sp.INITIAL_ELO)), 2),
            "best_elo": round(float(state.get("best_elo", sp.INITIAL_ELO)), 2),
            "best_checkpoint": state.get("best_checkpoint"),
            "latest_checkpoint": state.get("latest_checkpoint"),
            "promotions": int(state.get("promotions", 0)),
            "candidate": state.get("candidate"),
            "last_match": state.get("last_match"),
            "last_update": state.get("last_update"),
            "last_eval": state.get("last_eval"),
            "last_rating_pass": state.get("last_rating_pass"),
            "live_progress": state.get("live_progress"),
            "last_error": state.get("last_error"),
            "runtime": runtime,
            "device_backend": device_backend,
            "device_repr": device_plan.get("repr", "-"),
            "device_preference": device_preference,
            "device_requested_backend": device_plan.get("requested_backend", device_preference),
            "device_reason": device_plan.get("reason", ""),
            "device_benchmark": device_plan.get("benchmark"),
            "simulation_workers": int(config.get("simulation_workers", sp.SIMULATION_WORKERS_AUTO)),
            "resolved_simulation_workers": sp.resolve_simulation_workers(
                config.get("simulation_workers", sp.SIMULATION_WORKERS_AUTO)
            ),
            "logical_processors": int(self._runtime_info.get("cpu_count", 1) or 1),
            "run_dir": state.get("run_dir", str((Path(sp.RUNS_DIR) / run_name).resolve())),
        }

    def _write_details(self, text: str) -> None:
        current_text = self.details_text.get("1.0", "end-1c")
        if current_text == text:
            return
        yview = self.details_text.yview()
        xview = self.details_text.xview()
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", text)
        self.details_text.configure(state="disabled")
        if yview:
            self.details_text.yview_moveto(yview[0])
        if xview:
            self.details_text.xview_moveto(xview[0])

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _show_text_report(self, title: str, text: str) -> None:
        window = tk.Toplevel(self, bg=WINDOW_BG)
        window.title(title)
        window.geometry("860x980")
        window.minsize(680, 540)

        header = tk.Frame(window, bg=WINDOW_BG, padx=12, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text=title,
            bg=WINDOW_BG,
            fg=TEXT,
            anchor="w",
            font=("Segoe UI Semibold", 14),
        ).pack(fill="x")

        body = ScrolledText(
            window,
            wrap="none",
            bg=PANEL_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            padx=12,
            pady=12,
            font=("Consolas", 11),
        )
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        body.insert("1.0", text)
        body.configure(state="disabled")

    def _on_run_tree_select(self, _: Optional[tk.Event] = None) -> None:
        if self._suppress_run_events:
            return
        selection = self.runs_tree.selection()
        if not selection:
            return
        focus = self.runs_tree.focus()
        run_name = focus if focus in selection else selection[-1]
        self.run_name_var.set(run_name)
        checkpoints = sp.list_checkpoints(run_name)
        state = sp.get_run_state(run_name)
        self._refresh_checkpoint_selector(run_name, checkpoints)
        self._refresh_summary(
            run_name,
            run_names=self._known_run_names,
            state=state,
            checkpoints=checkpoints,
        )

    def _on_run_combo(self, _: Optional[tk.Event] = None) -> None:
        self.refresh_data()

    def _on_match_run_combo(self, _: Optional[tk.Event] = None) -> None:
        self.refresh_data()

    def _on_fork_source_combo(self, _: Optional[tk.Event] = None) -> None:
        self.refresh_data()

    def _on_checkpoint_tree_select(self, _: Optional[tk.Event] = None) -> None:
        selection = self.checkpoints_tree.selection()
        if not selection:
            return
        self.checkpoint_var.set(selection[-1])

    def _delete_selected_checkpoints(self) -> None:
        selection = list(self.checkpoints_tree.selection())
        if not selection:
            messagebox.showinfo(
                "No checkpoints selected",
                "Select one or more checkpoints in the Checkpoints table first.",
                parent=self,
            )
            return

        run_name = self._selected_run_name()
        checkpoints = [checkpoint for checkpoint in selection if checkpoint != "candidate"]
        if not checkpoints:
            messagebox.showinfo(
                "Candidate cannot be deleted here",
                "The candidate row is an in-progress policy, not a saved checkpoint file. Select one or more saved checkpoint rows instead.",
                parent=self,
            )
            return

        skipped_candidate = len(checkpoints) != len(selection)
        count = len(checkpoints)
        selected_text = ", ".join(checkpoints[:4])
        if count > 4:
            selected_text += f", and {count - 4} more"
        confirm = messagebox.askyesno(
            "Delete checkpoints",
            f"Delete {count} checkpoint(s) from run '{run_name}'?\n\nSelected: {selected_text}\n\nThis removes the checkpoint files and updates training_state.json to match.",
            parent=self,
        )
        if not confirm:
            return

        def action() -> Dict[str, Any]:
            return sp.delete_checkpoints(run_name=run_name, checkpoints=checkpoints)

        def on_success(result: Dict[str, Any]) -> None:
            replacement = result.get("replacement_checkpoint")
            if replacement:
                replacement_text = f" Replaced active champion snapshot with '{replacement}'."
            else:
                replacement_text = ""
            skipped_text = " Candidate row was skipped." if skipped_candidate else ""
            self.log(
                f"Deleted {len(result.get('deleted_checkpoints') or [])} checkpoint(s) from '{run_name}': "
                f"{', '.join(result.get('deleted_checkpoints') or [])}. "
                f"Latest is now '{result.get('latest_checkpoint')}', best is '{result.get('best_checkpoint')}'."
                f"{replacement_text}{skipped_text}"
            )

        self._run_async(f"Delete {count} checkpoint(s) from '{run_name}'", action, on_success=on_success)

    def _train_chunk(self) -> None:
        try:
            iterations = self._chunk_iterations()
            overrides = self._new_run_overrides_for(self._selected_run_name())
        except ValueError as exc:
            messagebox.showerror("Invalid chunk size", str(exc), parent=self)
            return
        run_name = self._selected_run_name()

        def action() -> Dict[str, Any]:
            return sp.train_iterations(iterations, run_name=run_name, **(overrides or {}))

        def on_success(result: Dict[str, Any]) -> None:
            self.log(
                f"Train chunk finished for '{run_name}': iteration {result.get('iteration')}, "
                f"elo {result.get('current_elo')}."
            )

        self._run_async(f"Train {iterations} iteration(s) for '{run_name}'", action, on_success=on_success)

    def _start_background(self) -> None:
        if self._has_active_command():
            messagebox.showinfo("Busy", "Another command is still running. Wait for it to finish first.", parent=self)
            return
        run_name = self._selected_run_name()
        try:
            overrides = self._new_run_overrides_for(run_name)
        except ValueError as exc:
            messagebox.showerror("Invalid new run settings", str(exc), parent=self)
            return
        summary = sp.start_training(run_name=run_name, background=True, **(overrides or {}))
        self.log(f"Background training started for '{run_name}'. Status: {summary.get('status')}.")
        self.refresh_data()

    def _continue_background(self) -> None:
        if self._has_active_command():
            messagebox.showinfo("Busy", "Another command is still running. Wait for it to finish first.", parent=self)
            return
        run_name = self._selected_run_name()
        try:
            overrides = self._new_run_overrides_for(run_name)
        except ValueError as exc:
            messagebox.showerror("Invalid new run settings", str(exc), parent=self)
            return
        message = sp.continue_training(run_name=run_name, **(overrides or {}))
        self.log(message)
        self.refresh_data()

    def _create_run(self) -> None:
        run_name = self._selected_run_name()
        try:
            overrides = self._selected_new_run_overrides()
        except ValueError as exc:
            messagebox.showerror("Invalid new run settings", str(exc), parent=self)
            return

        def action() -> Dict[str, Any]:
            return sp.create_run(run_name=run_name, **overrides)

        def on_success(result: Dict[str, Any]) -> None:
            config = sp.get_run_state(run_name).get("config") or {}
            self.log(
                f"Created run '{run_name}' with model {config.get('model_architecture')} "
                f"on {config.get('device_preference')} "
                f"and {config.get('training_matches_per_iteration')} match(es)/iter, "
                f"{config.get('training_games_per_match')} game(s)/match, "
                f"{config.get('training_decisions_per_game')} decision(s)/game, "
                f"train_temperature={config.get('train_temperature')}, "
                f"promotion_threshold={config.get('promotion_score_threshold')}, "
                f"{config.get('promotion_games')} promotion game(s), "
                f"simulation_workers={config.get('simulation_workers')}."
            )
            self.refresh_data()

        self._run_async(f"Create run '{run_name}'", action, on_success=on_success)

    def _create_run_from_checkpoint(self) -> None:
        run_name = self._selected_run_name()
        source_run_name = self.fork_source_run_var.get().strip() or sp.LATEST_RUN_NAME
        source_checkpoint = self.fork_source_checkpoint_var.get().strip() or "latest"
        try:
            overrides = self._selected_new_run_overrides()
        except ValueError as exc:
            messagebox.showerror("Invalid new run settings", str(exc), parent=self)
            return

        def action() -> Dict[str, Any]:
            return sp.create_run_from_checkpoint(
                run_name=run_name,
                source_run_name=source_run_name,
                source_checkpoint=source_checkpoint,
                model_type=self.new_run_model_var.get().strip() or sp.MODEL_TYPE_DEEP,
                device_preference=overrides["device_preference"],
                training_matches_per_iteration=overrides["training_matches_per_iteration"],
                training_games_per_match=overrides["training_games_per_match"],
                training_decisions_per_game=overrides["training_decisions_per_game"],
                train_temperature=overrides["train_temperature"],
                promotion_games=overrides["promotion_games"],
                promotion_score_threshold=overrides["promotion_score_threshold"],
                simulation_workers=overrides["simulation_workers"],
            )

        def on_success(result: Dict[str, Any]) -> None:
            state = sp.get_run_state(run_name)
            config = state.get("config") or {}
            forked_from = state.get("forked_from") or {}
            self.log(
                f"Created run '{run_name}' from {forked_from.get('run_name', source_run_name)} / "
                f"{forked_from.get('checkpoint', source_checkpoint)} "
                f"using architecture {config.get('model_architecture')} "
                f"(conversion={forked_from.get('conversion_mode', 'cloned')}) "
                f"and {config.get('training_matches_per_iteration')} match(es)/iter, "
                f"{config.get('training_games_per_match')} game(s)/match, "
                f"{config.get('training_decisions_per_game')} decision(s)/game, "
                f"train_temperature={config.get('train_temperature')}, "
                f"promotion_threshold={config.get('promotion_score_threshold')}, "
                f"{config.get('promotion_games')} promotion game(s), "
                f"simulation_workers={config.get('simulation_workers')}."
            )
            self.refresh_data()

        self._run_async(
            f"Create run '{run_name}' from '{source_run_name}' / '{source_checkpoint}'",
            action,
            on_success=on_success,
        )

    def _apply_run_settings(self) -> None:
        run_name = self._selected_run_name()
        if not sp.run_exists(run_name):
            messagebox.showerror("Run does not exist", f"Run '{run_name}' does not exist yet.", parent=self)
            return
        try:
            train_temperature = self._selected_run_train_temperature()
            promotion_threshold = self._selected_run_promotion_threshold()
            simulation_workers = self._selected_run_simulation_workers()
        except ValueError as exc:
            messagebox.showerror("Invalid run settings", str(exc), parent=self)
            return

        def action() -> Dict[str, Any]:
            return sp.update_run_config(
                run_name=run_name,
                train_temperature=train_temperature,
                promotion_score_threshold=promotion_threshold,
                simulation_workers=simulation_workers,
            )

        def on_success(result: Dict[str, Any]) -> None:
            state = sp.get_run_state(run_name)
            config = state.get("config") or {}
            self._loaded_run_settings_run_name = None
            self.log(
                f"Updated run settings for '{run_name}': "
                f"train_temperature={config.get('train_temperature')}, "
                f"promotion_score_threshold={config.get('promotion_score_threshold')}, "
                f"simulation_workers={config.get('simulation_workers')}."
            )
            self.refresh_data()

        self._run_async(f"Apply run settings for '{run_name}'", action, on_success=on_success)

    def _interrupt_background(self) -> None:
        run_name = self._selected_run_name()

        def action() -> str:
            return sp.interrupt_training(run_name=run_name)

        def on_success(result: str) -> None:
            self.log(result)

        self._run_async(f"Interrupt background training for '{run_name}'", action, on_success=on_success)

    def _run_rating_pass(self) -> None:
        run_name = self._selected_run_name()
        try:
            max_policies = self._rating_policy_count()
            games_per_pair = self._rating_games_per_pair()
        except ValueError as exc:
            messagebox.showerror("Invalid rating pass settings", str(exc), parent=self)
            return
        include_candidate = bool(self.include_candidate_var.get())
        pairings_total = max_policies * (max_policies - 1) // 2
        games_target = pairings_total * games_per_pair
        activity_label = f"{run_name}: {max_policies} policies"
        self._set_activity_progress(
            "rating_pass",
            {
                "status": "running",
                "label": activity_label,
                "games_completed": 0,
                "games_target": games_target,
                "pairings_completed": 0,
                "pairings_total": pairings_total,
            },
        )

        def action() -> Dict[str, Any]:
            def progress_callback(progress: Dict[str, Any]) -> None:
                self.progress_queue.put(
                    (
                        "rating_pass",
                        {
                            "status": "running",
                            "label": activity_label,
                            **progress,
                        },
                    )
                )

            try:
                return sp.run_rating_pass(
                    run_name=run_name,
                    max_policies=max_policies,
                    games_per_pair=games_per_pair,
                    include_candidate=include_candidate,
                    progress_callback=progress_callback,
                )
            except Exception:
                self.progress_queue.put(
                    (
                        "rating_pass",
                        {
                            "status": "failed",
                            "label": activity_label,
                        },
                    )
                )
                raise

        def on_success(result: Dict[str, Any]) -> None:
            leaderboard = list(result.get("leaderboard") or [])
            top_line = "-"
            if leaderboard:
                top_entry = leaderboard[0]
                top_line = f"{top_entry.get('name')} at {round(float(top_entry.get('elo', sp.INITIAL_ELO)), 2)}"
            self._set_activity_progress(
                "rating_pass",
                {
                    "status": "completed",
                    "label": activity_label,
                    "games_completed": int(result.get("games_completed", result.get("pairings_played", 0) * games_per_pair)),
                    "games_target": int(result.get("games_target", games_target)),
                    "pairings_completed": int(result.get("pairings_played", 0)),
                    "pairings_total": int(result.get("pairings_total", pairings_total)),
                    "duration_seconds": result.get("duration_seconds"),
                },
            )
            self.log(
                f"Rating pass finished for '{run_name}': {result.get('participant_count', 0)} models, "
                f"{result.get('pairings_played', 0)} pairings, top rated {top_line}."
            )

        self._run_async(f"Run rating pass for '{run_name}'", action, on_success=on_success)

    def _run_cross_run_rating(self) -> None:
        try:
            games = self._cross_run_games()
        except ValueError as exc:
            messagebox.showerror("Invalid cross-run rating settings", str(exc), parent=self)
            return
        selected_run_names = [
            str(item)
            for item in self.runs_tree.selection()
            if str(item) in set(self._known_run_names)
        ]
        if len(selected_run_names) < 2:
            messagebox.showerror(
                "Select runs",
                "Select at least two runs in the Runs table before starting cross-run rating.",
                parent=self,
            )
            return

        activity_label = f"{len(selected_run_names)} run(s), {games} game(s)"
        self._set_activity_progress(
            "cross_run_rating",
            {
                "status": "running",
                "label": activity_label,
                "games_completed": 0,
                "games_target": games,
                "pairings_completed": 0,
                "pairings_total": 0,
            },
        )

        def action() -> Dict[str, Any]:
            def progress_callback(progress: Dict[str, Any]) -> None:
                self.progress_queue.put(
                    (
                        "cross_run_rating",
                        {
                            "status": "running",
                            "label": activity_label,
                            **progress,
                        },
                    )
                )

            try:
                return sp.run_cross_run_calibration_games(
                    games=games,
                    run_names=selected_run_names,
                    progress_callback=progress_callback,
                )
            except Exception:
                self.progress_queue.put(
                    (
                        "cross_run_rating",
                        {
                            "status": "failed",
                            "label": activity_label,
                        },
                    )
                )
                raise

        def on_success(result: Dict[str, Any]) -> None:
            leaderboard = list(result.get("leaderboard") or [])
            top_line = "-"
            if leaderboard:
                top_entry = leaderboard[0]
                confidence = top_entry.get("confidence_radius")
                confidence_text = "-" if confidence is None else f"+/- {float(confidence):.1f}"
                top_line = f"{top_entry.get('run_name')} at {float(top_entry.get('elo', sp.INITIAL_ELO)):.1f} ({confidence_text})"
            self._set_activity_progress(
                "cross_run_rating",
                {
                    "status": "completed",
                    "label": activity_label,
                    "games_completed": int(result.get("games_completed", games)),
                    "games_target": int(result.get("games_target", games)),
                    "pairings_completed": int(result.get("pairings_played", 0)),
                    "pairings_total": int(result.get("pairings_total", 0)),
                    "duration_seconds": result.get("duration_seconds"),
                },
            )
            self.log(
                f"Cross-run rating finished: {result.get('games_completed', 0)} game(s), "
                f"{result.get('pairings_played', 0)} sampled pairing(s), top rated {top_line}."
            )
            report_lines = [
                "Cross-Run Rating",
                f"Games completed: {result.get('games_completed', 0)} / {result.get('games_target', games)}",
                f"Pairings sampled: {result.get('pairings_played', 0)} / {result.get('pairings_total', 0)}",
                f"Participants: {result.get('participant_count', 0)}",
                f"Selected runs: {', '.join(result.get('selected_run_names') or selected_run_names)}",
                f"Workers: {result.get('resolved_simulation_workers', '-')}",
                f"Duration: {sp._format_seconds(float(result.get('duration_seconds', 0.0)))}",
                f"Confidence level: {result.get('confidence_level', '-')}",
                "",
                "Leaderboard",
            ]
            for item in leaderboard:
                confidence = item.get("confidence_radius")
                confidence_text = "-" if confidence is None else f"+/- {float(confidence):.1f}"
                report_lines.append(
                    f"#{item.get('rank', '-')} {item.get('run_name', '-')}: "
                    f"{float(item.get('elo', sp.INITIAL_ELO)):.1f} Elo ({confidence_text}), "
                    f"{item.get('games', 0)} games, best {item.get('best_checkpoint', '-')}"
                )
            self._show_text_report("Cross-Run Rating", "\n".join(report_lines))

        self._run_async("Run cross-run rating", action, on_success=on_success)

    def _run_card_acquire_elo_test(self) -> None:
        run_name = self._selected_run_name()
        checkpoint = self._selected_checkpoint()
        if not sp.run_exists(run_name):
            messagebox.showerror("Run does not exist", f"Run '{run_name}' does not exist yet.", parent=self)
            return
        activity_label = f"{run_name} / {checkpoint}"
        self._set_activity_progress(
            "acquire_elo",
            {
                "status": "running",
                "label": activity_label,
                "games_completed": 0,
                "games_target": sp.CARD_ACQUIRE_ELO_TEST_GAMES,
            },
        )

        def action() -> Dict[str, Any]:
            def progress_callback(progress: Dict[str, Any]) -> None:
                self.progress_queue.put(
                    (
                        "acquire_elo",
                        {
                            "status": "running",
                            "label": activity_label,
                            **progress,
                        },
                    )
                )

            try:
                return sp.run_card_acquire_elo_test(
                    run_name=run_name,
                    checkpoint=checkpoint,
                    games=sp.CARD_ACQUIRE_ELO_TEST_GAMES,
                    progress_callback=progress_callback,
                )
            except Exception:
                self.progress_queue.put(
                    (
                        "acquire_elo",
                        {
                            "status": "failed",
                            "label": activity_label,
                        },
                    )
                )
                raise

        def on_success(result: Dict[str, Any]) -> None:
            leaderboard = list(result.get("leaderboard") or [])
            top_entry = leaderboard[0] if leaderboard else {}
            self._set_activity_progress(
                "acquire_elo",
                {
                    "status": "completed",
                    "label": f"{run_name} / {result.get('resolved_checkpoint', checkpoint)}",
                    "games_completed": int(result.get("games", 0)),
                    "games_target": sp.CARD_ACQUIRE_ELO_TEST_GAMES,
                    "duration_seconds": result.get("duration_seconds"),
                },
            )
            self.log(
                f"Acquire Elo test finished for '{run_name}' / '{checkpoint}': "
                f"{result.get('scored_decisions', 0)} scored decision(s) from "
                f"{result.get('eligible_single_acquire_turns', 0)} eligible single-acquire turn(s) across "
                f"{result.get('games', 0)} game(s). "
                f"Top card: {top_entry.get('card_name', '-')} at {round(float(top_entry.get('elo', 0.0)), 2)}. "
                f"Report saved to {result.get('report_path', '-')}"
            )
            self._show_text_report(
                f"Acquire Elo Test: {run_name} / {result.get('resolved_checkpoint', checkpoint)}",
                str(result.get("report_text") or ""),
            )

        self._run_async(
            f"Run acquire Elo test for '{run_name}' / '{checkpoint}'",
            action,
            on_success=on_success,
        )

    def _play_self_game(self) -> None:
        run_name = self._selected_run_name()
        checkpoint = self._selected_checkpoint()

        def action() -> Dict[str, Any]:
            return sp.play_self_game(
                run_name=run_name,
                checkpoint=checkpoint,
                verbose=False,
                ui_observer=self.game_view_bridge,
            )

        def on_success(result: Dict[str, Any]) -> None:
            self.log(
                f"Self-play finished for '{run_name}' / '{checkpoint}': "
                f"winner {result.get('winner')}, turns {result.get('turns_taken')}, "
                f"ended_by_limit={result.get('ended_by_limit')}."
            )

        self._run_async(f"Run self-play for '{run_name}' / '{checkpoint}'", action, on_success=on_success)

    def _play_policy_match(self) -> None:
        run_name_a = self.match_run_a_var.get().strip() or sp.LATEST_RUN_NAME
        checkpoint_a = self.match_checkpoint_a_var.get().strip() or "latest"
        run_name_b = self.match_run_b_var.get().strip() or sp.LATEST_RUN_NAME
        checkpoint_b = self.match_checkpoint_b_var.get().strip() or "latest"
        try:
            games_per_match = self._selected_match_games()
        except ValueError as exc:
            messagebox.showerror("Invalid match settings", str(exc), parent=self)
            return
        activity_label = f"{run_name_a} / {checkpoint_a} vs {run_name_b} / {checkpoint_b}"
        self._set_activity_progress(
            "policy_match",
            {
                "status": "running",
                "label": activity_label,
                "games_completed": 0,
                "games_target": games_per_match,
            },
        )

        def action() -> Dict[str, Any]:
            def progress_callback(progress: Dict[str, Any]) -> None:
                self.progress_queue.put(
                    (
                        "policy_match",
                        {
                            "status": "running",
                            "label": activity_label,
                            **progress,
                        },
                    )
                )

            try:
                return sp.play_policy_match(
                    run_name_a=run_name_a,
                    checkpoint_a=checkpoint_a,
                    run_name_b=run_name_b,
                    checkpoint_b=checkpoint_b,
                    games_per_match=games_per_match,
                    progress_callback=progress_callback,
                )
            except Exception:
                self.progress_queue.put(
                    (
                        "policy_match",
                        {
                            "status": "failed",
                            "label": activity_label,
                        },
                    )
                )
                raise

        def on_success(result: Dict[str, Any]) -> None:
            policy_a = result.get("policy_a") or {}
            policy_b = result.get("policy_b") or {}
            winner = result.get("winner")
            if winner == "policy_a":
                winner_text = f"{policy_a.get('run_name')} / {policy_a.get('checkpoint')}"
            elif winner == "policy_b":
                winner_text = f"{policy_b.get('run_name')} / {policy_b.get('checkpoint')}"
            else:
                winner_text = "draw"
            self._set_activity_progress(
                "policy_match",
                {
                    "status": "completed",
                    "label": activity_label,
                    "games_completed": int(result.get("games_played", 0)),
                    "games_target": games_per_match,
                    "duration_seconds": result.get("duration_seconds"),
                },
            )
            self.log(
                "Policy match finished: "
                f"{policy_a.get('run_name')} / {policy_a.get('checkpoint')} "
                f"{result.get('wins_a')}-{result.get('wins_b')} "
                f"{policy_b.get('run_name')} / {policy_b.get('checkpoint')} "
                f"over {result.get('games_played')} game(s). Winner: {winner_text}."
            )

        self._run_async(
            f"Run policy match for '{run_name_a}' / '{checkpoint_a}' vs '{run_name_b}' / '{checkpoint_b}'",
            action,
            on_success=on_success,
        )

    def _play_human_game(self) -> None:
        run_name = self._selected_run_name()
        checkpoint = self._selected_checkpoint()
        human_name = self.human_name_var.get().strip() or "Human"
        policy_name = self.policy_name_var.get().strip() or "Policy"

        def action() -> Dict[str, Any]:
            return sp.play_human_game(
                run_name=run_name,
                checkpoint=checkpoint,
                human_name=human_name,
                policy_name=policy_name,
                human_choose_fn=self.game_view_bridge.request_human_choice,
                verbose=False,
                ui_observer=self.game_view_bridge,
            )

        def on_success(result: Dict[str, Any]) -> None:
            self.log(
                f"Human game finished for '{run_name}' / '{checkpoint}': "
                f"winner {result.get('winner')}, turns {result.get('turns_taken')}, "
                f"ended_by_limit={result.get('ended_by_limit')}."
            )

        self._run_async(f"Play human game against '{run_name}' / '{checkpoint}'", action, on_success=on_success)

    def _handle_choice_requests(self) -> None:
        while True:
            try:
                request = self.choice_queue.get_nowait()
            except queue.Empty:
                break
            try:
                if isinstance(request, HumanChoiceRequest):
                    dialog = HumanDecisionDialog(self, request)
                    self.wait_window(dialog)
                    if dialog.selection is None:
                        request.error = "The human decision dialog was closed. The game was cancelled."
                    else:
                        request.selection = dialog.selection
                elif isinstance(request, GameViewRequest):
                    if request.kind == "start_session":
                        self.game_view.begin_session(str(request.payload.get("title", "Star Realms Game")), clear_log=True)
                    elif request.kind == "policy_choice":
                        player_name = str(request.payload["player_name"])
                        options = request.payload["options"]
                        state = request.payload["state"]
                        selection_index = int(request.payload["selection_index"])
                        self.game_view.show_state(
                            player_name,
                            options,
                            state,
                            selectable=False,
                            selected_index=selection_index,
                            mode_text="Policy-selected action.",
                        )
                        selected_text = chooser_ui._format_option(options[selection_index], state)
                        self.game_view.append_log(f"{player_name}: {selected_text}")
                    elif request.kind == "human_choice":
                        player_name = str(request.payload["player_name"])
                        options = request.payload["options"]
                        state = request.payload["state"]
                        selection = self.game_view.choose(
                            player_name,
                            options,
                            state,
                            mode_text="Choose the action to take.",
                        )
                        request.result = selection
                        selected_text = chooser_ui._format_option(options[selection], state)
                        self.game_view.append_log(f"{player_name}: {selected_text}")
                    elif request.kind == "finish_session":
                        result = dict(request.payload.get("result") or {})
                        self.game_view.append_log(
                            "Result: winner {winner}, turns {turns}, ended_by_limit={ended}.".format(
                                winner=result.get("winner", "-"),
                                turns=result.get("turns_taken", "-"),
                                ended=result.get("ended_by_limit", "-"),
                            )
                        )
                    else:
                        request.error = f"Unknown game view request kind: {request.kind}"
                else:
                    request.error = f"Unsupported queued request type: {type(request).__name__}"
            except Exception as exc:
                request.error = f"{type(exc).__name__}: {exc}"
            finally:
                request.event.set()

    def _handle_worker_results(self) -> bool:
        handled_result = False
        while True:
            try:
                status, label, payload, callback = self.result_queue.get_nowait()
            except queue.Empty:
                break

            handled_result = True

            if status == "success":
                self.log(f"{label} finished.")
                if callback is not None:
                    callback(payload)
            else:
                self.log(f"{label} failed.\n{payload}")
                messagebox.showerror("Command failed", payload, parent=self)
        return handled_result

    def _handle_progress_updates(self) -> None:
        while True:
            try:
                kind, payload = self.progress_queue.get_nowait()
            except queue.Empty:
                break
            if kind in self.activity_progress_value_vars:
                self._set_activity_progress(kind, payload)

    def _poll(self) -> None:
        self._handle_choice_requests()
        self._handle_progress_updates()
        if self._handle_worker_results():
            self._request_refresh(immediate=True)
        if self._refresh_requested or time.monotonic() >= self._next_refresh_at:
            try:
                self.refresh_data()
                self._last_refresh_error = None
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                now = time.time()
                if self._last_refresh_error != message or (now - self._last_refresh_error_at) >= 5.0:
                    self.log(f"Refresh skipped due to transient error: {message}")
                    self._last_refresh_error = message
                    self._last_refresh_error_at = now
                self._schedule_next_refresh(REFRESH_INTERVAL_ERROR_MS)
        self.after(POLL_INTERVAL_MS, self._poll)

    def _on_close(self) -> None:
        run_name = self._selected_run_name()
        state = sp.get_run_state(run_name)
        summary = self._summary_from_state(run_name, state)

        if summary.get("status") == "training":
            confirmed = messagebox.askyesno(
                "Training is running",
                "Background training is still running in this GUI process. Closing the window will stop it. Close anyway?",
                parent=self,
            )
            if not confirmed:
                return

        while True:
            try:
                request = self.choice_queue.get_nowait()
            except queue.Empty:
                break
            request.error = "The trainer GUI was closed."
            request.event.set()

        self.destroy()


def launch_gui() -> None:
    print(f"Launching Star Realms trainer GUI with {sys.executable}")
    app = TrainerGUI()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
