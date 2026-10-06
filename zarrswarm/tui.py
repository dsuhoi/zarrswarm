"""zs-tui - torrent-client style terminal UI for the local zs node.

Main list = datasets ("torrents": one row per grid the node seeds, holds or is downloading), filters on the left,
details below (general / content tree from metadata / piece map / peers / log). Dialogs: add download (pick
variables and a slice like files in a torrent), seed a path (scan preview), search the metadata index, settings
(edits config.toml). Keys: a add, s seed, / search, p pause/resume, del remove, o settings, q quit.
"""
import asyncio
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label, ListItem, ListView, Log, Select,
                             SelectionList, Static, Switch, TabbedContent, TabPane, TextArea, Tree)

from .common import state_home
from .store import CTL, http, open_views

FILTERS = [("all", "Все"), ("down", "Загружаются"), ("pause", "На паузе"), ("seed", "Раздаются"), ("done", "Скачаны"), ("err", "Ошибки")]
STATE = {"down": ("⇣ загрузка", "cyan"), "seed": ("⇡ раздача", "green"), "done": ("✓ скачано · раздаётся", "green"),
         "err": ("✗ ошибка", "red"), "part": ("◐ частично", "yellow"), "pause": ("⏸ пауза", "yellow")}


def _mb(b: float) -> str:
    for u, k in (("ГБ", 1e9), ("МБ", 1e6), ("КБ", 1e3)):
        if b >= k:
            return f"{b / k:,.1f} {u}"
    return f"{b:.0f} Б"


def _bar(frac: float, width: int = 16) -> Text:
    frac = max(0.0, min(frac, 1.0))
    n = int(round(frac * width))
    t = Text("█" * n, style="green" if frac >= 1 else "cyan")
    t.append("░" * (width - n), style="grey37")
    t.append(f" {100 * frac:5.1f}%")
    return t


def _eta(s: float | None) -> str:
    if s is None or s <= 0 or s > 1e7:
        return "∞" if s else ""
    return f"{int(s // 3600)}ч{int(s % 3600 // 60):02d}м" if s >= 3600 else f"{int(s // 60)}м{int(s % 60):02d}с"


def _ts(x) -> str:
    return datetime.fromtimestamp(x, timezone.utc).strftime("%Y-%m-%d %H:%M") if x is not None else "—"


def piece_row(bins: list) -> Text:
    """[[holders, local_frac]] -> colored strip: █ here · ▒ ≥2 peers · ░ one peer (rare) · · nobody."""
    t = Text()
    for h, loc in bins:
        if loc >= 1:
            t.append("█", "green")
        elif loc > 0:
            t.append("▓", "green")
        elif h >= 2:
            t.append("▒", "blue")
        elif h == 1:
            t.append("░", "yellow")
        else:
            t.append("·", "red")
    return t


def _title(v: dict) -> str | None:
    """Dataset title from the root group attributes (v2 .zattrs or v3 zarr.json)."""
    g = v.get("gdocs") or {}
    attrs = g.get(".zattrs") or (g.get("zarr.json") or {}).get("attributes") or {}
    t = attrs.get("title")
    return str(t) if t else None


def build_rows(status: dict, views: dict, rates: dict) -> list[dict]:
    """One row per grid: seeded paths, downloaded caches and grids with download jobs."""
    rows: dict[str, dict] = {}

    def row(g):
        return rows.setdefault(g, {"grid": g, "paths": [], "arrays": set(), "bytes": 0, "chunks": 0, "jobs": []})
    for s_ in status["seeds"]:
        r = row(s_["grid"])
        r["paths"].append(s_["path"])
        r["arrays"].update(s_["arrays"])
    for g, x in status["grids"].items():
        r = row(g)
        r["bytes"], r["chunks"] = x["bytes"], x["chunks"]
        r["arrays"].update(x["arrays"])
    for j in status["jobs"]:
        row(j["grid"])["jobs"].append(j)
    out = []
    for g, r in rows.items():
        v = views.get(g) or {}
        dims = set((v.get("grid") or {}).get("dims") or {})
        data = sorted(n for n in (r["arrays"] | set(v.get("arrays") or {})) if n not in dims and "#" not in n)
        running = [j for j in r["jobs"] if j["state"] == "running"]
        last = r["jobs"][-1] if r["jobs"] else None
        if running:
            kind = "down"
        elif any(j["state"] == "paused" for j in r["jobs"]):
            kind = "pause"
        elif last and last["state"].startswith("error"):
            kind = "err"
        elif r["paths"]:
            kind = "seed"
        elif last and last["state"] == "partial":
            kind = "part"
        else:
            kind = "done"
        swarm = v.get("nchunks") or 0
        prog = (sum(j["done"] for j in running) / max(sum(j["total"] - j["missing"] for j in running), 1)) if running \
            else (r["chunks"] / swarm if swarm else 1.0)
        down = sum(rates.get(j["id"], 0) for j in running)
        left = sum(j["bytes"] / max(j["done"], 1) * (j["total"] - j["done"] - j["missing"]) for j in running)
        name = _title(v) or Path(r["paths"][0]).name if r["paths"] or _title(v) else (",".join(data[:3]) + ("…" if len(data) > 3 else "")) or g[:12]
        out.append({"grid": g, "name": name, "kind": kind, "size": r["bytes"], "chunks": r["chunks"], "swarm": swarm,
                    "progress": prog, "down": down, "eta": left / down if down else None,
                    "peers": v.get("npeers", 0), "paths": r["paths"], "vars": data, "jobs": r["jobs"]})
    return sorted(out, key=lambda x: (x["kind"] != "down", x["name"]))


# ---------------------------------------------------------------------------------------------------- dialogs
class AddDialog(ModalScreen):
    """Like 'add torrent': load metadata, pick variables (= files) and a slice, optionally export afterwards."""
    DEFAULT_CSS = """
    AddDialog { align: center middle; }
    #box { width: 96; height: auto; max-height: 90%; border: thick $accent; background: $panel; padding: 1 2; }
    #box Input { margin-bottom: 1; }
    #vars { height: 10; border: round $primary; }
    #info { color: $text-muted; height: auto; margin-bottom: 1; }
    .row { height: auto; }
    .row Input, .row Select { width: 1fr; }
    #buttons { height: 3; align-horizontal: right; }
    """
    BINDINGS = [("escape", "dismiss(None)", "Отмена")]

    def __init__(self, ctl: str, link: str = ""):
        super().__init__()
        self.ctl, self.link0, self.meta = ctl, link, None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="box"):
            yield Label("[b]Добавить загрузку[/b]")
            with Horizontal(classes="row"):
                yield Input(self.link0, placeholder="zs://…  или  zs://имя@ключ", id="link")
                yield Button("Метаданные", id="meta", variant="primary")
            yield Static("введите ссылку и нажмите «Метаданные»", id="info")
            yield SelectionList(id="vars")
            with Horizontal(classes="row"):
                yield Input(placeholder="время: 2020-01-01:2020-02-01", id="time")
                yield Input(placeholder="регион: lat=40:60,lon=0:30", id="sel")
            with Horizontal(classes="row"):
                yield Select([("JLPS: быстрые пиры, любые раскладки", "jlps"), ("минимум байт", "bytes")],
                             value="jlps", allow_blank=False, id="cover")
                yield Select([("оптимальный порядок", "optimal"), ("прогрессивный (грубо → точно)", "progressive")],
                             value="optimal", allow_blank=False, id="order")
            with Horizontal(classes="row"):
                yield Input(placeholder="сохранить в файл после загрузки: out.zarr / out.nc (необязательно)", id="out")
                yield Input(placeholder="подписка: держать свежими последние 7d / 12h", id="follow")
            with Horizontal(id="buttons"):
                yield Button("Отмена", id="cancel")
                yield Button("Загрузить", id="ok", variant="success")

    def on_mount(self):
        if self.link0:
            self.load_meta()

    @on(Button.Pressed, "#meta")
    @on(Input.Submitted, "#link")
    def _meta(self):
        self.load_meta()

    @work(thread=True, exclusive=True)
    def load_meta(self):
        link = self.query_one("#link", Input).value.strip()
        info = self.query_one("#info", Static)
        self.app.call_from_thread(info.update, "загружаю метаданные…")
        try:
            views = open_views(link, self.ctl)
        except Exception as e:
            self.app.call_from_thread(info.update, f"[red]не удалось: {e}[/red]")
            return
        items, lines = [], []
        for g, view in views:
            dims = view.v["grid"]["dims"]
            for n, a in sorted(view.arrays.items()):
                if n in dims or "#" in n:
                    continue
                attrs = a.get("attrs") or {}
                desc = attrs.get("long_name") or attrs.get("standard_name") or ""
                items.append((f"{n}  [{', '.join(a['dims'])}]  {desc} {attrs.get('units', '') and '(' + attrs['units'] + ')'}",
                              n, True))
            t = view.v["grid"].get("time")
            if t and view.v["gmin"] is not None:
                lines.append(f"сетка {g[:12]}: {' × '.join(f'{k}={v}' for k, v in dims.items())}, время "
                             f"{_ts(view.v['gmin'] * t['dt'] + t['rphase'])} … {_ts((view.v['gmax'] - 1) * t['dt'] + t['rphase'])}"
                             f" (шаг {t['dt'] / 3600:g} ч), чанков в рое {view.v['nchunks']}, пиров {view.v['npeers']}")
            else:
                lines.append(f"сетка {g[:12]}: {dims}, чанков {view.v['nchunks']}, пиров {view.v['npeers']}")
        self.meta = views

        def show():
            sl = self.query_one("#vars", SelectionList)
            sl.clear_options()
            sl.add_options(items)
            info.update("\n".join(lines) or "пусто")
        self.app.call_from_thread(show)

    @on(Button.Pressed, "#cancel")
    def _cancel(self):
        self.dismiss(None)

    @on(Button.Pressed, "#ok")
    def _ok(self):
        link = self.query_one("#link", Input).value.strip()
        if not link:
            self.query_one("#info", Static).update("[red]нужна ссылка[/red]")
            return
        self.dismiss({"link": link, "vars": list(self.query_one("#vars", SelectionList).selected) or None,
                      "time": self.query_one("#time", Input).value.strip(),
                      "sel": self.query_one("#sel", Input).value.strip(),
                      "cover": self.query_one("#cover", Select).value, "order": self.query_one("#order", Select).value,
                      "out": self.query_one("#out", Input).value.strip(),
                      "follow": self.query_one("#follow", Input).value.strip()})


class SeedDialog(ModalScreen):
    DEFAULT_CSS = """
    SeedDialog { align: center middle; }
    #box { width: 90; height: auto; border: thick $accent; background: $panel; padding: 1 2; }
    #preview { height: auto; max-height: 16; margin: 1 0; }
    #buttons { height: 3; align-horizontal: right; }
    """
    BINDINGS = [("escape", "dismiss(None)", "Отмена")]

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Label("[b]Раздавать датасет[/b]  (данные не копируются; ссылка вычисляется по структуре)")
            yield Input(placeholder="путь к каталогу .zarr", id="path")
            yield Static("", id="preview")
            with Horizontal(id="buttons"):
                yield Button("Проверить", id="scan")
                yield Button("Отмена", id="cancel")
                yield Button("Раздавать", id="ok", variant="success")

    @on(Button.Pressed, "#scan")
    @on(Input.Submitted, "#path")
    def _scan(self):
        self.scan()

    @work(thread=True, exclusive=True)
    def scan(self):
        from .scan import scan
        path = os.path.abspath(os.path.expanduser(self.query_one("#path", Input).value.strip()))
        pv = self.query_one("#preview", Static)
        self.app.call_from_thread(pv.update, "сканирую…")
        try:
            r = scan(path)
        except Exception as e:
            self.app.call_from_thread(pv.update, f"[red]{e}[/red]")
            return
        lines = []
        for g, sg in r["subgrids"].items():
            nb = sum(c[2] for c in sg["chunks"].values())
            lines.append(f"[b]zs://{g[:16]}…[/b]  {' × '.join(f'{k}={v}' for k, v in sg['grid']['dims'].items())}"
                         f"  {len(sg['chunks'])} чанков, {_mb(nb)}")
            for n, a in sorted(sg["arrays"].items()):
                if n not in sg["grid"]["dims"]:
                    lines.append(f"   {n} [{', '.join(a['dims'])}]  раскладки: {', '.join(a['layouts'])}")
        self.app.call_from_thread(pv.update, "\n".join(lines))

    @on(Button.Pressed, "#cancel")
    def _cancel(self):
        self.dismiss(None)

    @on(Button.Pressed, "#ok")
    def _ok(self):
        p = self.query_one("#path", Input).value.strip()
        self.dismiss(os.path.abspath(os.path.expanduser(p)) if p else None)


class SearchDialog(ModalScreen):
    DEFAULT_CSS = """
    SearchDialog { align: center middle; }
    #box { width: 120; height: 80%; border: thick $accent; background: $panel; padding: 1 2; }
    #found { height: 1fr; }
    """
    BINDINGS = [("escape", "dismiss(None)", "Закрыть")]

    def __init__(self, ctl: str):
        super().__init__()
        self.ctl = ctl

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Label("[b]Поиск по индексу метаданных[/b]  (имя переменной, standard_name, long_name) · Enter — добавить")
            yield Input(placeholder="напр. t2m или air_temperature", id="tag")
            yield DataTable(id="found", cursor_type="row")

    def on_mount(self):
        self.query_one("#found", DataTable).add_columns("ссылка", "сидов", "переменные", "период", "сетка")
        self.query_one("#tag", Input).focus()

    @on(Input.Submitted, "#tag")
    def _search(self, ev: Input.Submitted):
        self.search(ev.value.strip())

    @work(thread=True, exclusive=True)
    def search(self, tag: str):
        hits = http(self.ctl, "GET", "/api/search?tag=" + urllib.parse.quote(tag)) if tag else []

        def show():
            t = self.query_one("#found", DataTable)
            t.clear()
            for h in hits:
                tr = f"{_ts(h['tr'][0])[:10]} … {_ts(h['tr'][1])[:10]}" if h.get("tr") else "—"
                t.add_row(f"zs://{h['grid']}", str(h["seeders"]), ",".join(h["vars"]), tr,
                          " × ".join(f"{k}={v}" for k, v in (h.get("dims") or {}).items()))
            if hits:
                t.focus()
        self.app.call_from_thread(show)

    @on(DataTable.RowSelected, "#found")
    def _pick(self, ev: DataTable.RowSelected):
        self.dismiss(str(ev.data_table.get_row(ev.row_key)[0]))


class SettingsScreen(ModalScreen):
    """Edits ~/.zs/config.toml (the node reads it at start: restart the node to apply)."""
    DEFAULT_CSS = """
    SettingsScreen { align: center middle; }
    #box { width: 100; height: 90%; border: thick $accent; background: $panel; padding: 1 2; }
    .f { height: auto; }
    .f Label { width: 26; padding-top: 1; }
    .f Input { width: 1fr; }
    .f Switch { width: auto; }
    TextArea { height: 6; }
    #buttons { height: 3; align-horizontal: right; }
    """
    BINDINGS = [("escape", "dismiss(None)", "Закрыть")]

    def __init__(self, home: Path):
        super().__init__()
        from .cli import _config
        self.home, self.cfg = home, _config(home)

    def _field(self, label, widget):
        with Horizontal(classes="f"):
            yield Label(label)
            yield widget

    def compose(self) -> ComposeResult:
        c = self.cfg
        with VerticalScroll(id="box"):
            yield Label(f"[b]Настройки узла[/b]  {self.home / 'config.toml'}  (применяются после перезапуска узла)")
            yield Static(f"сеть: {c.get('network') or '—'}", classes="f")
            yield from self._field("Потолок отдачи, МБ/с", Input(str(c.get("upload_mbps", 0)), id="upload_mbps",
                                                                 type="number"))
            yield from self._field("Кэш скачанного, ГБ", Input(str(c.get("cache_max_gb", 0)), id="cache_max_gb",
                                                               type="number", placeholder="0 — без ограничения"))
            yield from self._field("Порт данных", Input(str(c.get("port", 7881)), id="port", type="integer"))
            yield from self._field("Адрес привязки", Input(c.get("host", "127.0.0.1"), id="host"))
            yield from self._field("Bootstrap (через ,)", Input(",".join(c.get("bootstrap", [])), id="bootstrap"))
            yield from self._field("Релей для узла за NAT", Input(c.get("relay") or "", id="relay"))
            yield from self._field("Быть релеем", Switch(bool(c.get("relay_server")), id="relay_server"))
            yield from self._field("Автоопределение адреса", Switch(bool(c.get("auto")), id="auto"))
            yield from self._field("Ключ закрытой сети", Input(c.get("network_key") or "", id="network_key",
                                                               password=True, placeholder="пусто — открытая сеть"))
            yield from self._field("Доверенные ключи (через ,)", Input(",".join(c.get("trust", [])), id="trust"))
            yield Label("Раздавать при старте (путь или glob, по строке):")
            yield TextArea("\n".join(x["path"] for x in c.get("seed", [])), id="seeds")
            yield Label("Тонкая настройка (ключ = значение, по строке; см. docs/config.md):")
            yield TextArea("\n".join(f"{k} = {v}" for k, v in c.get("tuning", {}).items()), id="tuning")
            yield Static("", id="err")
            with Horizontal(id="buttons"):
                yield Button("Отмена", id="cancel")
                yield Button("Сохранить", id="save", variant="success")

    @on(Button.Pressed, "#cancel")
    def _cancel(self):
        self.dismiss(None)

    @on(Button.Pressed, "#save")
    def _save(self):
        from .cli import TUNING, _write_config
        q = lambda i: self.query_one(f"#{i}", Input).value.strip()
        split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
        try:
            tuning = {}
            for line in self.query_one("#tuning", TextArea).text.splitlines():
                if line.strip():
                    k, _, v = line.partition("=")
                    k = k.strip()
                    if k not in TUNING:
                        raise ValueError(f"неизвестный ключ {k!r}; допустимые: {', '.join(TUNING)}")
                    tuning[k] = float(v) if "." in v or "e" in v.lower() else int(v)
            cfg = dict(self.cfg, upload_mbps=float(q("upload_mbps") or 0), cache_max_gb=float(q("cache_max_gb") or 0), port=int(q("port") or 7881),
                       host=q("host") or "127.0.0.1", bootstrap=split(q("bootstrap")), relay=q("relay") or None,
                       relay_server=self.query_one("#relay_server", Switch).value,
                       auto=self.query_one("#auto", Switch).value, network_key=q("network_key"),
                       trust=split(q("trust")), tuning=tuning,
                       seed=[{"path": x.strip()} for x in self.query_one("#seeds", TextArea).text.splitlines() if x.strip()])
            self.home.mkdir(parents=True, exist_ok=True)
            _write_config(self.home, cfg)
        except Exception as e:
            self.query_one("#err", Static).update(f"[red]{e}[/red]")
            return
        self.dismiss(str(self.home / "config.toml"))


class ConfirmDialog(ModalScreen):
    DEFAULT_CSS = """
    ConfirmDialog { align: center middle; }
    #box { width: 70; height: auto; border: thick $error; background: $panel; padding: 1 2; }
    #buttons { height: 3; align-horizontal: right; }
    """
    BINDINGS = [("escape", "dismiss(False)", "Нет")]

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Static(self.text)
            with Horizontal(id="buttons"):
                yield Button("Нет", id="no")
                yield Button("Да", id="yes", variant="error")

    @on(Button.Pressed)
    def _b(self, ev: Button.Pressed):
        self.dismiss(ev.button.id == "yes")


# ---------------------------------------------------------------------------------------------------- app
class ZsTui(App):
    TITLE = "ZarrSwarm"
    CSS = """
    #top { height: 1fr; }
    #filters { width: 22; border-right: solid $primary-darken-2; }
    #list { width: 1fr; }
    #details { height: 55%; border-top: solid $accent; }
    #general, #pieces { padding: 0 1; }
    #statusbar { height: 1; background: $primary-darken-3; padding: 0 1; }
    """
    BINDINGS = [Binding("a", "add", "Добавить"), Binding("s", "seed", "Раздать"), Binding("slash", "search", "Поиск"),
                Binding("delete", "remove", "Удалить"), Binding("p", "pause", "Пауза/пуск"), Binding("o", "settings", "Настройки"),
                Binding("r", "refresh", "Обновить", show=False), Binding("q", "quit", "Выход")]

    def __init__(self, ctl: str, home: str | Path | None = None):
        super().__init__()
        self.ctl = ctl
        self.home = state_home(home)
        self.status: dict = {}
        self.views: dict[str, dict] = {}   # grid -> /api/view summary (refreshed in the background)
        self._view_ts: dict[str, float] = {}
        self.rows: list[dict] = []
        self.filter = "all"
        self.selected: str | None = None
        self._prev: dict = {}              # job id / "up" -> (t, bytes) for instantaneous rates
        self.rates: dict[str, float] = {}
        self._content_for = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top"):
            yield ListView(*[ListItem(Label(n), id=f"f_{k}") for k, n in FILTERS], id="filters")
            yield DataTable(id="list", cursor_type="row", zebra_stripes=True)
        with TabbedContent(id="details"):
            with TabPane("Общее", id="d_general"):
                yield Static("выберите датасет", id="general")
            with TabPane("Содержимое", id="d_content"):
                yield Tree("датасет", id="content")
            with TabPane("Карта частей", id="d_pieces"):
                yield Static("", id="pieces")
            with TabPane("Пиры", id="d_peers"):
                yield DataTable(id="peers", cursor_type="row")
            with TabPane("Журнал", id="d_log"):
                yield Log(id="log")
        yield Static("подключение к узлу…", id="statusbar")
        yield Footer()

    def on_mount(self):
        self.query_one("#list", DataTable).add_columns("Название", "Размер", "Прогресс", "Статус", "↓ скорость",
                                                       "Пиры", "Осталось", "Ссылка")
        self.query_one("#peers", DataTable).add_columns("пир", "адрес", "чанков", "скорость", "путь")
        self.set_interval(1.0, self.action_refresh)
        self.set_interval(3.0, self.refresh_details)
        self.call_after_refresh(self.action_refresh)

    def log_line(self, s: str):
        self.query_one("#log", Log).write_line(f"{time.strftime('%H:%M:%S')} {s}")

    # ------------------------------------------------------------------ polling
    async def action_refresh(self):
        try:
            s = await asyncio.to_thread(http, self.ctl, "GET", "/api/status", None, 5)
        except Exception as e:
            self.query_one("#statusbar", Static).update(Text(f"узел недоступен ({self.ctl}): {e}", style="red"))
            return
        self.status, now = s, time.monotonic()
        for j in s["jobs"]:
            t0, b0 = self._prev.get(j["id"], (now, j["bytes"]))
            self.rates[j["id"]] = (j["bytes"] - b0) / (now - t0) if now > t0 else 0.0
            self._prev[j["id"]] = (now, j["bytes"])
        up_b = s["served"].get("bytes", 0) + s["served"].get("relayed_bytes", 0)
        t0, b0 = self._prev.get("up", (now, up_b))
        up = (up_b - b0) / (now - t0) if now > t0 else 0.0
        self._prev["up"] = (now, up_b)
        grids = {x["grid"] for x in s["seeds"]} | set(s["grids"]) | {j["grid"] for j in s["jobs"]}
        stale = [g for g in grids if time.monotonic() - self._view_ts.get(g, 0) > 10]
        if stale:
            self.fetch_views(stale)
        self.rows = build_rows(s, self.views, self.rates)
        self._render_list()
        down = sum(r["down"] for r in self.rows)
        net = "закрытая сеть 🔒" if s.get("private") else "открытая сеть"
        where = "публичный" if not s["ro"] else f"за NAT, релей {'✓' if s['relay_up'] else '✗'}"
        bar = Text()
        bar.append(f"↓ {_mb(down)}/с  ", "cyan")
        bar.append(f"↑ {_mb(up)}/с  ", "green")
        bar.append(f"DHT {s['contacts']}  ·  {net}  ·  {where}  ·  узел {s['id'][:10]}  ·  {s['addr']}")
        self.query_one("#statusbar", Static).update(bar)

    @work(thread=True, group="views")
    def fetch_views(self, grids):
        for g in grids:
            self._view_ts[g] = time.monotonic()
            try:
                self.views[g] = http(self.ctl, "GET", f"/api/view/{g}", None, 20)
            except Exception:
                pass

    def _visible(self):
        f = self.filter
        return [r for r in self.rows if f == "all" or r["kind"] == f or (f == "done" and r["kind"] in ("done", "part"))
                or (f == "seed" and r["kind"] in ("seed", "done"))]

    def _render_list(self):
        t = self.query_one("#list", DataTable)
        keep = self.selected
        t.clear()
        for r in self._visible():
            label, color = STATE[r["kind"]]
            size = _mb(r["size"]) + (f" · {r['chunks']}/{r['swarm']} ч." if r["swarm"] else f" · {r['chunks']} ч.")
            t.add_row(Text(r["name"], style="bold"), size, _bar(r["progress"]), Text(label, style=color),
                      f"{_mb(r['down'])}/с" if r["down"] else "", str(r["peers"] or ""), _eta(r["eta"]),
                      f"zs://{r['grid'][:14]}…", key=r["grid"])
        if keep is not None:
            try:
                t.move_cursor(row=t.get_row_index(keep))
            except Exception:
                pass
        self._render_general()

    def _row(self):
        return next((r for r in self.rows if r["grid"] == self.selected), None)

    # ------------------------------------------------------------------ details
    @on(DataTable.RowHighlighted, "#list")
    def _highlight(self, ev: DataTable.RowHighlighted):
        if ev.row_key is None or ev.row_key.value == self.selected:
            return
        self.selected = ev.row_key.value
        self._render_general()
        self.refresh_details()

    @on(ListView.Highlighted, "#filters")
    def _filter(self, ev: ListView.Highlighted):
        if ev.item is not None:
            self.filter = ev.item.id.removeprefix("f_")
            self._render_list()

    def _render_general(self):
        r = self._row()
        if not r:
            return
        v = self.views.get(r["grid"]) or {}
        t = (v.get("grid") or {}).get("time")
        lines = [f"ссылка      zs://{r['grid']}",
                 f"переменные  {', '.join(r['vars']) or '—'}",
                 f"сетка       {' × '.join(f'{k}={n}' for k, n in ((v.get('grid') or {}).get('dims') or {}).items()) or '—'}"]
        if t and v.get("gmin") is not None:
            lines.append(f"период      {_ts(v['gmin'] * t['dt'] + t['rphase'])} … "
                         f"{_ts((v['gmax'] - 1) * t['dt'] + t['rphase'])}  (шаг {t['dt'] / 3600:g} ч)")
        lines.append(f"у меня      {r['chunks']} чанков, {_mb(r['size'])}   в рое: {r['swarm'] or '?'} чанков, "
                     f"{r['peers'] or '?'} пиров")
        for p in r["paths"]:
            lines.append(f"раздаётся   {p}")
        for j in r["jobs"][-5:]:
            el = (j["t1"] or time.time()) - j["t0"]
            lines.append(f"задача {j['id']}  {j['label'][:40]:40}  {j['done']}/{j['total']} ч.  {_mb(j['bytes'])}  "
                         f"{el:5.1f} с  {j['state']}" + (f"  нет реплик: {j['missing']}" if j["missing"] else ""))
        head = Text(r["name"], "bold")
        head.append("   " + STATE[r["kind"]][0], STATE[r["kind"]][1])
        self.query_one("#general", Static).update(Text("\n").join([head, *map(Text, lines)]))

    def refresh_details(self):
        if not self.selected:
            return
        tab = self.query_one("#details", TabbedContent).active
        if tab == "d_pieces":
            self.load_pieces(self.selected)
        elif tab == "d_peers":
            self.load_peers(self.selected)
        elif tab == "d_content" and self._content_for != self.selected:
            self.load_content(self.selected)

    @on(TabbedContent.TabActivated, "#details")
    def _tab(self):
        self.refresh_details()

    @work(thread=True, exclusive=True, group="pieces")
    def load_pieces(self, grid):
        width = max(20, self.size.width - 22)
        try:
            p = http(self.ctl, "GET", f"/api/pieces/{grid}?width={width}", None, 20)
        except Exception as e:
            self.app.call_from_thread(self.query_one("#pieces", Static).update, f"[red]{e}[/red]")
            return
        out = Text()
        out.append(f"{_ts(p['t0'])}{' ' * max(1, width - 32)}{_ts(p['t1'])}\n", "grey62")
        for n, bins in sorted(p["vars"].items()):
            if "#" in n:
                continue
            here = sum(1 for _, l in bins if l >= 1) / max(len(bins), 1)
            out.append(f"{n}  (у меня {100 * here:.0f}%)\n", "bold")
            out.append_text(piece_row(bins))
            out.append("\n")
        out.append("\n█ у меня  ▓ частично  ▒ ≥2 пиров  ░ один пир (редкое)  · ни у кого\n", "grey62")
        self.app.call_from_thread(self.query_one("#pieces", Static).update, out)

    @work(thread=True, exclusive=True, group="peers")
    def load_peers(self, grid):
        try:
            peers = http(self.ctl, "GET", f"/api/peers/{grid}", None, 20)["peers"]
        except Exception:
            return
        me = self.status.get("id")

        def show():
            t = self.query_one("#peers", DataTable)
            t.clear()
            for p in sorted(peers, key=lambda p: -p["chunks"]):
                t.add_row(p["id"][:12] + (" (я)" if p["id"] == me else ""), p["addr"], str(p["chunks"]),
                          f"{_mb(p['bw'])}/с" if p["bw"] else "—", "через релей" if "/r/" in p["addr"] else "напрямую")
        self.app.call_from_thread(show)

    @work(thread=True, exclusive=True, group="content")
    def load_content(self, grid):
        """Metadata tree like an xarray repr: dimensions, coordinates (ranges), variables (attrs, layouts)."""
        from . import open_dataset
        self._content_for = grid
        try:
            view = open_views(f"zs://{grid}", self.ctl)[0][1]
            ds = open_dataset(f"zs://{grid}", ctl=self.ctl)
        except Exception as e:
            self._content_for = None
            self.app.call_from_thread(self.log_line, f"метаданные {grid[:12]}: {e}")
            return

        def rng(c):
            try:
                v = c.values
                if not v.size:
                    return ""
                if v.dtype.kind == "M":
                    import numpy as np
                    return f"{np.datetime_as_string(v.min(), 'm')} … {np.datetime_as_string(v.max(), 'm')}"
                return f"{v.min():g} … {v.max():g}" if v.dtype.kind in "fiu" else f"{v.min()} … {v.max()}"
            except Exception:
                return "не загружено"

        def show():
            tree = self.query_one("#content", Tree)
            tree.clear()
            tree.root.set_label(Text(f"zs://{grid[:16]}…  ({ds.nbytes / 1e6:,.1f} МБ в распакованном виде)", "bold"))
            d = tree.root.add("Измерения", expand=True)
            for k, n in ds.sizes.items():
                d.add_leaf(f"{k}: {n}")
            c = tree.root.add("Координаты", expand=True)
            for k, co in ds.coords.items():
                c.add_leaf(f"{k} ({', '.join(co.dims)})  {co.dtype}  {rng(co)}")
            vs = tree.root.add("Переменные", expand=True)
            for k, da in ds.data_vars.items():
                a = da.attrs
                node = vs.add(Text.assemble((k, "bold cyan"), f"  ({', '.join(da.dims)})  {da.dtype}  "
                                            f"{_mb(da.nbytes)}  {a.get('long_name', '')} {a.get('units', '') and '[' + a['units'] + ']'}"))
                for ak, av in a.items():
                    if not ak.startswith("_"):
                        node.add_leaf(f"{ak}: {av}")
                arr = view.arrays.get(k) or {}
                for lay, li in (arr.get("layouts") or {}).items():
                    node.add_leaf(f"раскладка {lay}: чанк {tuple(li['chunks'])}, чанков {li.get('n', '?')}")
            at = tree.root.add("Атрибуты", expand=False)
            for ak, av in ds.attrs.items():
                if not ak.startswith(("zs_", "zt_")):
                    at.add_leaf(f"{ak}: {av}")
            tree.root.expand()
        self.app.call_from_thread(show)

    # ------------------------------------------------------------------ actions
    def action_add(self, link: str = ""):
        self.push_screen(AddDialog(self.ctl, link), self._added)

    def _added(self, req):
        if req:
            self.start_download(req)

    @work(thread=True)
    def start_download(self, req):
        from .cli import _dur, _isel_from_sel, _time, export
        try:
            t0, t1 = _time(req["time"])
            isel = _isel_from_sel(req["link"], self.ctl, req["sel"] or None)
            if req.get("follow"):  # subscription instead of a one-off slice
                r = http(self.ctl, "POST", "/api/follow", {"link": req["link"], "vars": req["vars"],
                                                           "last_s": _dur(req["follow"]), "isel": isel})
                self.call_from_thread(self.log_line, f"подписка {r['id']}: последние {req['follow']} держатся свежими "
                                                     f"({len(r['jobs'])} задач догрузки)")
                return
            jobs = []
            for grid, view in open_views(req["link"], self.ctl):
                names = [n for n in view.arrays if n not in view.v["grid"]["dims"] and "#" not in n
                         and (not req["vars"] or n in req["vars"])]
                for var in names:
                    jobs.append(http(self.ctl, "POST", "/api/download", {
                        "grid": grid, "cover": req["cover"], "order": req["order"],
                        "region": {"var": var, "t0": t0, "t1": t1, "isel": isel},
                        "label": f"{var} {req['time']}".strip()})["job"])
            self.call_from_thread(self.log_line, f"поставлено в загрузку: {len(jobs)} задач ({req['link'][:30]}…)")
            if req["out"]:
                from .store import wait_job
                for j in jobs:
                    wait_job(self.ctl, j)
                export(req["link"], self.ctl, req["vars"], t0, t1, isel, req["out"])
                self.call_from_thread(self.log_line, f"сохранено: {req['out']}")
        except Exception as e:
            self.call_from_thread(self.log_line, f"ошибка загрузки: {e}")

    def action_seed(self):
        self.push_screen(SeedDialog(), self._seeded)

    def _seeded(self, path):
        if path:
            self.do_seed(path)

    @work(thread=True)
    def do_seed(self, path):
        try:
            r = http(self.ctl, "POST", "/api/seed", {"path": path})
            self.call_from_thread(self.log_line, f"раздаётся: {r['link']}  ({r['chunks']} чанков, {','.join(r['arrays'])})")
        except Exception as e:
            self.call_from_thread(self.log_line, f"ошибка раздачи: {e}")

    def action_search(self):
        self.push_screen(SearchDialog(self.ctl), lambda link: link and self.action_add(link))

    def action_settings(self):
        self.push_screen(SettingsScreen(self.home),
                         lambda p: p and self.log_line(f"настройки сохранены в {p}; перезапустите узел"))

    def action_pause(self):
        r = self._row()
        if r:
            self.do_pause(r)

    @work(thread=True)
    def do_pause(self, r):
        running = [j for j in r["jobs"] if j["state"] == "running"]
        paused = [j for j in r["jobs"] if j["state"] == "paused"]
        for j in running:
            http(self.ctl, "POST", f"/api/pause/{j['id']}")
        for j in paused if not running else []:
            http(self.ctl, "POST", f"/api/resume/{j['id']}")
        msg = f"пауза: {len(running)} задач" if running else f"продолжено: {len(paused)} задач" if paused else \
            "нет загрузок для паузы"
        self.call_from_thread(self.log_line, f"{r['name']}: {msg}")

    def action_remove(self):
        r = self._row()
        if not r:
            return
        running = [j for j in r["jobs"] if j["state"] in ("running", "paused")]
        what = (f"остановить {len(running)} загрузок" if running else "") + \
               (" и " if running and r["paths"] else "") + (f"снять с раздачи {', '.join(r['paths'])}" if r["paths"] else "")
        if not what:
            self.log_line("нечего удалять: скачанные данные остаются в кэше узла")
            return
        self.push_screen(ConfirmDialog(f"{r['name']}: {what}?"), lambda ok: ok and self.do_remove(r))

    @work(thread=True)
    def do_remove(self, r):
        for j in r["jobs"]:
            if j["state"] in ("running", "paused"):
                http(self.ctl, "POST", f"/api/cancel/{j['id']}")
        for p in r["paths"]:
            http(self.ctl, "POST", "/api/unseed", {"path": p})
        self.call_from_thread(self.log_line, f"удалено: {r['name']}")


ZtTui = ZsTui  # compatibility for existing callers


def main():
    ctl = sys.argv[1] if len(sys.argv) > 1 else CTL
    ZsTui(ctl).run()


if __name__ == "__main__":
    main()
