from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - depends on local Python install
    tk = None


FACTION_ORDER = ("green", "blue", "yellow", "red", "none")
FACTION_COLORS = {
    "green": "#27543d",
    "blue": "#23466b",
    "yellow": "#655020",
    "red": "#6b2b2b",
    "none": "#3b4252",
}
WINDOW_BG = "#0f1724"
PANEL_BG = "#162235"
SECTION_BG = "#1d2d45"
TEXT_COLOR = "#f5f7fb"
SUBTEXT_COLOR = "#c6d3e6"
MUTED_TEXT = "#8da4c3"
ACCENT = "#f3b63a"
BUTTON_BG = "#28456a"
BUTTON_ACTIVE_BG = "#356296"


def _console_choose(player_name: str, options: Sequence[Sequence[Any]], state: Dict[str, Any]) -> int:
    print(player_name + "'s turn:")
    print("Authority: " + str(state["authority"]) + "/" + str(state["opponentAuthority"]))
    print("Attack: " + str(state["attack"]) + ". Trade: " + str(state["trade"]))

    def _print_cards(label: str, cards: Iterable[Any]) -> None:
        cards = list(cards or [])
        values = ", ".join(_card_name(card) for card in cards) or "none"
        print(f"{label}: {values}")

    cards_in_play = []
    for faction_cards in (state.get("cardsInPlay") or {}).values():
        cards_in_play.extend(card[0] for card in faction_cards)
    _print_cards("Cards in play", cards_in_play)

    opponent_cards_in_play = []
    for faction_cards in (state.get("opponentCardsInPlay") or {}).values():
        opponent_cards_in_play.extend(card[0] for card in faction_cards)
    _print_cards("Opponent cards in play", opponent_cards_in_play)

    _print_cards("Trade row", state.get("tradeRow") or [])
    _print_cards("Hand", state.get("hand") or [])
    _print_cards("Deck", list(state.get("scrambleDeck") or []) + list(state.get("topCards") or []))
    _print_cards("Discard pile", state.get("discardPile") or [])
    _print_cards(
        "Opponent deck+hand",
        list(state.get("opponentScrambleDeckAndHand") or [])
        + list(state.get("opponentTopCards") or [])
        + list(state.get("opponentHandCards") or []),
    )
    _print_cards("Opponent discard pile", state.get("opponentDiscardPile") or [])

    print(
        "Must discard: "
        + str(state.get("mustDiscard"))
        + ", Opponent must discard: "
        + str(state.get("opponentMustDiscard"))
        + ", nextShipTop: "
        + str(state.get("nextShipTop"))
        + ", blobPlayCount: "
        + str(state.get("blobPlayCount"))
    )
    print("Select an option:")
    for index, option in enumerate(options):
        print(str(index) + ": " + _format_option(option, state))
    return int(input())


def _card_name(card: Any) -> str:
    if isinstance(card, list) and card and isinstance(card[0], (list, tuple)):
        return _card_name(card[0])
    if not isinstance(card, (list, tuple)) or not card:
        return "Unknown card"
    if card[0] == "none":
        return "Empty slot"
    if len(card) > 1 and isinstance(card[1], int) and len(card) > 6:
        return f"{card[0]} ({card[1]})"
    return str(card[0])


def _card_details(card: Any) -> Sequence[Any]:
    if isinstance(card, list) and card and isinstance(card[0], (list, tuple)):
        return card[0]
    if isinstance(card, (list, tuple)):
        return card
    return ()


def _faction_name(card: Any) -> str:
    details = _card_details(card)
    if len(details) > 5:
        return str(details[5])
    return "none"


def _count_label(cards: Sequence[Any]) -> str:
    count = len(cards)
    return "1 card" if count == 1 else f"{count} cards"


def _amount_label(value: Any, noun: str) -> str:
    if value == 1:
        return f"1 {noun}"
    return f"{value} {noun}s"


def _humanize_ability(ability_name: str, amount: Optional[Any] = None) -> str:
    clean_name = ability_name.lstrip("-")
    match clean_name:
        case "gainattack":
            return f"gain {amount} attack" if amount is not None else "gain attack"
        case "trade":
            return f"gain {amount} trade" if amount is not None else "gain trade"
        case "authority":
            return f"gain {amount} authority" if amount is not None else "gain authority"
        case "draw":
            draw_amount = 1 if amount in (None, 0) else amount
            return f"draw {_amount_label(draw_amount, 'card')}"
        case "draw2":
            return "draw 2 cards"
        case "rowscrap":
            return "scrap a trade row card"
        case "scrapany":
            return "scrap a card from hand or discard"
        case "scraptwo":
            return "scrap up to 2 cards, then draw that many"
        case "drawscrap":
            return "draw a card, then scrap from hand"
        case "killbase":
            return "destroy a target base"
        case "5ordraws":
            return "choose +5 attack or draw cards"
        case "5or0or3":
            return "choose +3 trade or +5 attack"
        case "copyship":
            return "copy another ship"
        case "recycle":
            return "discard up to 2 cards, then draw that many"
        case "0or2or2":
            return "choose +2 authority or +2 trade"
        case "2or3or0":
            return "choose +2 attack or +3 authority"
        case "0or1or1":
            return "choose +1 authority or +1 trade"
        case "freebuy":
            return "acquire a ship for free"
        case "destroyscrap":
            return "destroy a base, then scrap a trade row card"
        case "drawdestroy":
            return "destroy a base, then draw a card"
        case "opdiscard":
            return "make the opponent discard"
        case "shiptop":
            return "put the next ship you buy on top of your deck"
        case "bases2d2":
            return "draw 2 cards if you have at least 2 bases"
        case "allally":
            return "treat all cards as allied"
        case "fleethq":
            return "give ships +1 attack"
        case _:
            return clean_name


def _format_card_text(card: Any, in_play: bool = False) -> str:
    details = _card_details(card)
    if not details:
        return "Unknown card"
    if details[0] == "none":
        return "Empty slot"

    summary_parts = [f"C{details[1]}" if len(details) > 1 else "C?"]
    if len(details) > 6:
        summary_parts.append(str(details[6]).upper())
    if len(details) > 5:
        summary_parts.append(str(details[5]).title())

    stat_parts = []
    if len(details) > 2 and details[2]:
        stat_parts.append(f"ATK {details[2]}")
    if len(details) > 4 and details[4]:
        stat_parts.append(f"TRA {details[4]}")
    if len(details) > 3 and details[3]:
        stat_parts.append(f"AUTH {details[3]}")
    if len(details) > 12 and details[12]:
        stat_parts.append(f"SH {details[12]}")
    if not stat_parts:
        stat_parts.append("No passive")

    ability_parts = []
    if len(details) > 7 and details[7] != "none":
        ability_parts.append(f"Act {_humanize_ability(str(details[7]))}")
    if len(details) > 8 and details[8] != "none":
        ally_amount = details[9] if len(details) > 9 and details[9] else None
        ability_parts.append(f"Ally {_humanize_ability(str(details[8]), ally_amount)}")
    if len(details) > 10 and details[10] != "none":
        scrap_amount = details[11] if len(details) > 11 and details[11] else None
        ability_parts.append(f"Scrap {_humanize_ability(str(details[10]), scrap_amount)}")
    if not ability_parts:
        ability_parts.append("No abilities")

    lines = [f"{' | '.join(summary_parts)} | {' | '.join(stat_parts)}", " | ".join(ability_parts)]

    if in_play and isinstance(card, list) and len(card) >= 5:
        status_parts = []
        if card[1]:
            status_parts.append("Ally used")
        if card[2] not in (None, "none", "used"):
            status_parts.append(f"Ready {_humanize_ability(str(card[2]))}")
        elif card[2] == "used":
            status_parts.append("Option used")
        if not card[3]:
            status_parts.append("Removed from play")
        if card[4]:
            status_parts.append("Stealth copy")
        if status_parts:
            lines[-1] = f"{lines[-1]} | {' | '.join(status_parts)}"

    return "\n".join(lines)


def _lookup_in_play_name(cards_in_play: Dict[str, Sequence[Any]], faction: Any, position: Any) -> str:
    faction_cards = (cards_in_play or {}).get(faction) or []
    if isinstance(position, int) and 0 <= position < len(faction_cards):
        return _card_name(faction_cards[position])
    return "Unknown target"


def _format_option(option: Sequence[Any], state: Dict[str, Any]) -> str:
    if not option:
        return "Unknown option"

    action = option[0]
    match action:
        case "play":
            return f"Play {_card_name(option[2])}"
        case "abilityOption":
            card_name = _lookup_in_play_name(state.get("cardsInPlay") or {}, option[1], option[2])
            return f"Use {_humanize_ability(str(option[3]))} from {card_name}"
        case "scrapFromPlay":
            amount = option[5] if len(option) > 5 else None
            return f"Scrap {_card_name(option[3])} for {_humanize_ability(str(option[4]), amount)}"
        case "attack":
            target_name = _lookup_in_play_name(state.get("opponentCardsInPlay") or {}, option[1], option[2])
            return f"Attack {target_name} for {option[3]} damage"
        case "attackOpponent":
            return f"Attack opponent for {option[1]} damage"
        case "acquire":
            return f"Acquire {_card_name(option[2])} for {option[2][1]} trade"
        case "freeAcquire":
            return f"Acquire {_card_name(option[2])} for free"
        case "endTurn":
            return "End turn"
        case "gainattack":
            return f"Take +{option[1]} attack"
        case "draw":
            return f"Draw {_amount_label(option[1], 'card')}"
        case "trade":
            return f"Take +{option[1]} trade"
        case "authority":
            return f"Take +{option[1]} authority"
        case "copyship":
            return f"Copy {_card_name(option[1])} with Stealth Needle"
        case "nocopy":
            return "Do not copy a ship"
        case "killbase":
            return f"Destroy {_card_name(option[3])}"
        case "nokill":
            return "Do not destroy a base"
        case "rowscrap":
            return f"Scrap {_card_name(option[2])} from the trade row"
        case "noRowScrap":
            return "Leave the trade row alone"
        case "nodiscard":
            return "Do not discard"
        case "noScrapFromHand":
            return "Do not scrap a card"
        case _:
            action_name = str(action)
            if action_name.startswith("discard"):
                return f"Discard {_card_name(option[2])}"
            if action_name.startswith("scrapFromHand"):
                return f"Scrap {_card_name(option[2])} from hand"
            if action_name.startswith("scrapFromDiscard"):
                return f"Scrap {_card_name(option[2])} from discard pile"
            return str(option)


if tk is not None:
    class ScrollableFrame(tk.Frame):
        def __init__(self, parent: tk.Widget, bg: str) -> None:
            super().__init__(parent, bg=bg, highlightthickness=0, bd=0)
            self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
            scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=scrollbar.set)

            self.content = tk.Frame(self.canvas, bg=bg, highlightthickness=0, bd=0)
            self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

            self.canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            self.content.bind("<Configure>", self._on_content_configure)
            self.canvas.bind("<Configure>", self._on_canvas_configure)
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
            self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux_up, add="+")
            self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux_down, add="+")

        def _on_content_configure(self, _: tk.Event) -> None:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def _on_canvas_configure(self, event: tk.Event) -> None:
            self.canvas.itemconfigure(self.window_id, width=event.width)

        def _pointer_inside(self) -> bool:
            widget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
            while widget is not None:
                if widget == self:
                    return True
                widget = widget.master
            return False

        def _scroll(self, units: int) -> None:
            if self.canvas.bbox("all") is None:
                return
            self.canvas.yview_scroll(units, "units")

        def _on_mousewheel(self, event: tk.Event) -> None:
            if not self._pointer_inside():
                return
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return
            units = -max(1, abs(delta) // 120) if delta > 0 else max(1, abs(delta) // 120)
            self._scroll(units)

        def _on_mousewheel_linux_up(self, _: tk.Event) -> None:
            if self._pointer_inside():
                self._scroll(-1)

        def _on_mousewheel_linux_down(self, _: tk.Event) -> None:
            if self._pointer_inside():
                self._scroll(1)


    class GameChooserGUI:
        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title("AstroSynapse Game Chooser")
            self.root.geometry("1540x940")
            self.root.minsize(1200, 760)
            self.root.configure(bg=WINDOW_BG)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)

            self._closed = False
            self._selection_var: Optional[tk.IntVar] = None

            self._build_layout()

        def _build_layout(self) -> None:
            header = tk.Frame(self.root, bg=WINDOW_BG, padx=14, pady=10)
            header.pack(fill="x")

            self.turn_label = tk.Label(
                header,
                text="Waiting for game state",
                bg=WINDOW_BG,
                fg=TEXT_COLOR,
                font=("Segoe UI Semibold", 18),
                anchor="w",
            )
            self.turn_label.pack(fill="x")

            self.summary_label = tk.Label(
                header,
                text="",
                bg=WINDOW_BG,
                fg=SUBTEXT_COLOR,
                font=("Segoe UI", 10),
                justify="left",
                anchor="w",
            )
            self.summary_label.pack(fill="x", pady=(2, 0))

            body = tk.Frame(self.root, bg=WINDOW_BG, padx=14, pady=0)
            body.pack(fill="both", expand=True)
            body.grid_columnconfigure(0, weight=3)
            body.grid_columnconfigure(1, weight=2)
            body.grid_rowconfigure(0, weight=1)

            self.state_panel = ScrollableFrame(body, bg=WINDOW_BG)
            self.state_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))

            self.options_panel = ScrollableFrame(body, bg=WINDOW_BG)
            self.options_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 10))

            self._build_state_sections()
            self._build_options_sections()

        def _build_state_sections(self) -> None:
            self.match_summary_section = self._create_section(self.state_panel.content, "Match Summary")
            self.opponent_board_section = self._create_section(self.state_panel.content, "Opponent Battlefield")
            self.trade_row_section = self._create_section(self.state_panel.content, "Trade Row")
            self.player_board_section = self._create_section(self.state_panel.content, "Your Battlefield")
            self.hand_section = self._create_section(self.state_panel.content, "Your Hand")
            self.deck_section = self._create_section(self.state_panel.content, "Your Deck")
            self.discard_section = self._create_section(self.state_panel.content, "Your Discard Pile")
            self.opponent_hidden_section = self._create_section(self.state_panel.content, "Opponent Known Deck And Hand")
            self.opponent_discard_section = self._create_section(self.state_panel.content, "Opponent Discard Pile")

        def _build_options_sections(self) -> None:
            body = self._create_section(self.options_panel.content, "Available Actions")
            self.options_intro_label = tk.Label(
                body,
                text="Click the action you want to send back to the game.",
                bg=SECTION_BG,
                fg=SUBTEXT_COLOR,
                justify="left",
                anchor="w",
                wraplength=420,
                font=("Segoe UI", 10),
            )
            self.options_intro_label.pack(fill="x")

            self.options_body = tk.Frame(body, bg=SECTION_BG)
            self.options_body.pack(fill="both", expand=True, pady=(6, 0))

        def _create_section(self, parent: tk.Widget, title: str) -> tk.Frame:
            container = tk.Frame(parent, bg=PANEL_BG, highlightbackground="#263953", highlightthickness=1, bd=0)
            container.pack(fill="x", pady=(0, 6))

            title_label = tk.Label(
                container,
                text=title,
                bg=PANEL_BG,
                fg=TEXT_COLOR,
                anchor="w",
                padx=8,
                pady=6,
                font=("Segoe UI Semibold", 12),
            )
            title_label.pack(fill="x")

            body = tk.Frame(container, bg=SECTION_BG, padx=8, pady=8)
            body.pack(fill="x")
            return body

        def _on_close(self) -> None:
            self._closed = True
            if self._selection_var is not None:
                self._selection_var.set(-1)

        def choose(self, player_name: str, options: Sequence[Sequence[Any]], state: Dict[str, Any]) -> int:
            if self._closed:
                raise SystemExit("The chooser window was closed.")

            self._selection_var = tk.IntVar(value=-1)
            self._render(player_name, options, state)

            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.root.update_idletasks()
            self.root.wait_variable(self._selection_var)

            selected = self._selection_var.get()
            self._selection_var = None
            if self._closed or selected < 0:
                raise SystemExit("The chooser window was closed.")
            return selected

        def _render(self, player_name: str, options: Sequence[Sequence[Any]], state: Dict[str, Any]) -> None:
            summary_lines = [
                f"You {state.get('authority')} auth | {state.get('attack')} atk | {state.get('trade')} trade | discard {state.get('mustDiscard')}",
                f"Opp {state.get('opponentAuthority')} auth | opp discard {state.get('opponentMustDiscard')}",
                f"Next-top {bool(state.get('nextShipTop'))} | blob plays {state.get('blobPlayCount')}",
            ]
            self.turn_label.configure(text=f"{player_name}'s decision")
            self.summary_label.configure(text="    |    ".join(summary_lines))

            self._render_match_summary(state)
            self._render_battlefield(self.opponent_board_section, state.get("opponentCardsInPlay") or {})
            self._render_single_cards_section(
                self.trade_row_section,
                "Trade row",
                state.get("tradeRow") or [],
                columns=6,
            )
            self._render_battlefield(self.player_board_section, state.get("cardsInPlay") or {})
            self._render_single_cards_section(
                self.hand_section,
                f"Hand ({_count_label(state.get('hand') or [])})",
                state.get("hand") or [],
                columns=6,
            )
            self._render_player_deck(state)
            self._render_single_cards_section(
                self.discard_section,
                f"Discard pile ({_count_label(state.get('discardPile') or [])})",
                state.get("discardPile") or [],
                columns=7,
            )
            self._render_opponent_hidden(state)
            self._render_single_cards_section(
                self.opponent_discard_section,
                f"Opponent discard ({_count_label(state.get('opponentDiscardPile') or [])})",
                state.get("opponentDiscardPile") or [],
                columns=7,
            )
            self._render_options(options, state)

        def _render_match_summary(self, state: Dict[str, Any]) -> None:
            self._clear(self.match_summary_section)

            summary_text = " | ".join(
                [
                    f"You {state.get('authority')} auth / {state.get('attack')} atk / {state.get('trade')} trade",
                    f"Opp {state.get('opponentAuthority')} auth",
                    f"Discard {state.get('mustDiscard')} / opp {state.get('opponentMustDiscard')}",
                    f"Next-top {bool(state.get('nextShipTop'))}",
                    f"Blob plays {state.get('blobPlayCount')}",
                ]
            )

            summary_label = tk.Label(
                self.match_summary_section,
                text=summary_text,
                bg=SECTION_BG,
                fg=TEXT_COLOR,
                justify="left",
                anchor="w",
                wraplength=920,
                font=("Segoe UI", 10),
            )
            summary_label.pack(fill="x")

        def _render_battlefield(self, section: tk.Frame, cards_by_faction: Dict[str, Sequence[Any]]) -> None:
            self._clear(section)

            any_cards = False
            for faction in FACTION_ORDER:
                faction_cards = list((cards_by_faction or {}).get(faction) or [])
                if not faction_cards:
                    continue

                any_cards = True
                heading = tk.Label(
                    section,
                    text=f"{faction.title()} ({_count_label(faction_cards)})",
                    bg=SECTION_BG,
                    fg=ACCENT,
                    anchor="w",
                    font=("Segoe UI Semibold", 10),
                )
                heading.pack(fill="x", pady=(0, 4))

                group = tk.Frame(section, bg=SECTION_BG)
                group.pack(fill="x", pady=(0, 4))
                self._populate_cards(group, faction_cards, columns=6, in_play=True)

            if not any_cards:
                self._render_empty_text(section, "No cards in play.")

        def _render_opponent_hidden(self, state: Dict[str, Any]) -> None:
            self._clear(self.opponent_hidden_section)

            self._render_cards_group(
                self.opponent_hidden_section,
                f"Unknown opponent deck + hidden hand ({_count_label(state.get('opponentScrambleDeckAndHand') or [])})",
                state.get("opponentScrambleDeckAndHand") or [],
                columns=7,
            )
            self._render_cards_group(
                self.opponent_hidden_section,
                f"Known top cards ({_count_label(state.get('opponentTopCards') or [])})",
                state.get("opponentTopCards") or [],
                columns=7,
            )
            self._render_cards_group(
                self.opponent_hidden_section,
                f"Known hand cards ({_count_label(state.get('opponentHandCards') or [])})",
                state.get("opponentHandCards") or [],
                columns=7,
            )

        def _render_player_deck(self, state: Dict[str, Any]) -> None:
            self._clear(self.deck_section)

            self._render_cards_group(
                self.deck_section,
                f"Scrambled deck ({_count_label(state.get('scrambleDeck') or [])})",
                state.get("scrambleDeck") or [],
                columns=7,
            )
            self._render_cards_group(
                self.deck_section,
                f"Known top cards ({_count_label(state.get('topCards') or [])})",
                state.get("topCards") or [],
                columns=7,
            )

        def _render_single_cards_section(
            self,
            section: tk.Frame,
            title: str,
            cards: Sequence[Any],
            columns: int = 5,
        ) -> None:
            self._clear(section)
            self._render_cards_group(section, title, cards, columns=columns)

        def _render_cards_group(
            self,
            parent: tk.Frame,
            title: str,
            cards: Sequence[Any],
            columns: int = 5,
        ) -> None:
            group = tk.Frame(parent, bg=SECTION_BG)
            group.pack(fill="x", pady=(0, 6))

            heading = tk.Label(
                group,
                text=title,
                bg=SECTION_BG,
                fg=ACCENT,
                anchor="w",
                font=("Segoe UI Semibold", 10),
            )
            heading.pack(fill="x", pady=(0, 4))

            cards = list(cards or [])
            if cards:
                card_holder = tk.Frame(group, bg=SECTION_BG)
                card_holder.pack(fill="x")
                self._populate_cards(card_holder, cards, columns=columns)
            else:
                self._render_empty_text(group, "No known cards.")

        def _populate_cards(
            self,
            parent: tk.Frame,
            cards: Sequence[Any],
            columns: int = 5,
            in_play: bool = False,
        ) -> None:
            for index, card in enumerate(cards):
                row = index // columns
                column = index % columns
                tile = self._create_card_tile(parent, card, in_play=in_play)
                tile.grid(row=row, column=column, sticky="nsew", padx=3, pady=3)
                parent.grid_columnconfigure(column, weight=1)

        def _create_card_tile(self, parent: tk.Widget, card: Any, in_play: bool = False) -> tk.Frame:
            faction = _faction_name(card)
            card_bg = FACTION_COLORS.get(faction, FACTION_COLORS["none"])
            text = _format_card_text(card, in_play=in_play)

            frame = tk.Frame(parent, bg=card_bg, highlightbackground="#0a1018", highlightthickness=1, bd=0, padx=6, pady=5)
            name_label = tk.Label(
                frame,
                text=_card_name(card),
                bg=card_bg,
                fg=TEXT_COLOR,
                justify="left",
                anchor="w",
                wraplength=150,
                font=("Segoe UI Semibold", 10),
            )
            name_label.pack(fill="x")

            details_label = tk.Label(
                frame,
                text=text,
                bg=card_bg,
                fg=SUBTEXT_COLOR,
                justify="left",
                anchor="w",
                wraplength=150,
                font=("Segoe UI", 8),
            )
            details_label.pack(fill="x", pady=(2, 0))
            return frame

        def _render_options(self, options: Sequence[Sequence[Any]], state: Dict[str, Any]) -> None:
            self._clear(self.options_body)
            self.options_panel.canvas.yview_moveto(0)

            for index, option in enumerate(options):
                wrapper = tk.Frame(
                    self.options_body,
                    bg=PANEL_BG,
                    highlightbackground="#294160",
                    highlightthickness=1,
                    bd=0,
                    padx=8,
                    pady=6,
                )
                wrapper.pack(fill="x", pady=(0, 5))

                button = tk.Button(
                    wrapper,
                    text=f"{index}. {_format_option(option, state)}",
                    command=lambda i=index: self._selection_var.set(i),
                    bg=BUTTON_BG,
                    activebackground=BUTTON_ACTIVE_BG,
                    fg=TEXT_COLOR,
                    activeforeground=TEXT_COLOR,
                    relief="flat",
                    bd=0,
                    padx=10,
                    pady=6,
                    justify="left",
                    anchor="w",
                    wraplength=390,
                    font=("Segoe UI Semibold", 9),
                )
                button.pack(fill="x")

                raw_option = tk.Label(
                    wrapper,
                    text=str(option),
                    bg=PANEL_BG,
                    fg=MUTED_TEXT,
                    justify="left",
                    anchor="w",
                    wraplength=390,
                    font=("Consolas", 8),
                )
                raw_option.pack(fill="x", pady=(3, 0))

        def _render_empty_text(self, parent: tk.Widget, text: str) -> None:
            empty_label = tk.Label(
                parent,
                text=text,
                bg=SECTION_BG,
                fg=MUTED_TEXT,
                anchor="w",
                justify="left",
                font=("Segoe UI", 9),
            )
            empty_label.pack(fill="x", pady=(0, 2))

        def _clear(self, widget: tk.Widget) -> None:
            for child in widget.winfo_children():
                child.destroy()


_GUI: Optional[GameChooserGUI] = None
_GUI_DISABLED = False


def choose(playerName: str, options: Sequence[Sequence[Any]], state: Dict[str, Any]) -> int:
    global _GUI, _GUI_DISABLED

    if tk is None or _GUI_DISABLED:
        return _console_choose(playerName, options, state)

    if _GUI is None:
        try:
            _GUI = GameChooserGUI()
        except Exception:  # pragma: no cover - depends on local display availability
            _GUI_DISABLED = True
            return _console_choose(playerName, options, state)

    return _GUI.choose(playerName, options, state)
