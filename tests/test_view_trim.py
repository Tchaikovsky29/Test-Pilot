"""Checks for the trimmed page view: modal-only, viewport windowing, shorter changes and history, and the two
perception pieces that feed them (layout_positions, and the walker's modal/position tracking on a fake AX tree).
Run with `pytest tests/test_view_trim.py`, or `python tests/test_view_trim.py`."""
from test_pilot.evidence import Changes, render_changes
from test_pilot.explorer import Event, format_event
from test_pilot.models import Element, Line, Observation
from test_pilot.perceive import DomInfo, _fill_positions, _FrameWalker, layout_positions
from test_pilot.render import render


def el(i, widget="button", name="", **kw):
    return Element(id=i, role=widget, widget=widget, name=name or f"b{i}", **kw)


def ctl(e, x=20.0, y=None):
    return Line(kind="control", element_id=e.id, x=x, y=y)


def obs(lines, elements, **kw):
    return Observation(url="https://x.test/", title="T", lines=lines, elements=elements,
                       viewport_w=1280, viewport_h=800, **kw)


def dialog(elements_in, prose="Please read this"):
    ok = el(900, name="OK")
    return [Line(kind="overlay", text="dialog blocking the page: handle it first"), Line(kind="heading", text="Notice"),
            Line(kind="text", text=prose), ctl(ok), Line(kind="overlay", text="end of dialog")], [ok]


# ---- popup: what is behind a modal is not sent ----------------------------------------------------------------------
def test_modal_hides_the_page_behind_it():
    dl, de = dialog([])
    page = [el(1, name="Search"), el(2, name="Login")]
    o = obs(dl + [ctl(page[0], y=100), ctl(page[1], y=140)], de + page, blocked=True, modal=True)
    out = render(o)
    assert '[900] button "OK"' in out and "Please read this" in out
    assert '"Search"' not in out and '"Login"' not in out
    assert "page behind the dialog is hidden: 2 controls" in out


def test_soft_overlay_keeps_the_page():  # a fixed layer that is not role=dialog: the page may still be usable
    dl, de = dialog([])
    page = [el(1, name="Search")]
    out = render(obs(dl + [ctl(page[0], y=100)], de + page, blocked=True, modal=False))
    assert '"Search"' in out and '[900] button "OK"' in out


def test_modal_without_a_control_keeps_the_page():  # hiding the page here could strand the model
    page = [el(1, name="Search")]
    lines = [Line(kind="overlay", text="dialog blocking the page: handle it first"), Line(kind="text", text="Loading…"),
             Line(kind="overlay", text="end of dialog"), ctl(page[0], y=100)]
    assert '"Search"' in render(obs(lines, page, blocked=True, modal=True))


def test_window_false_lists_everything():  # what the explorer uses after two failed actions
    dl, de = dialog([])
    page = [el(1, name="Search")]
    out = render(obs(dl + [ctl(page[0], y=100)], de + page, blocked=True, modal=True), window=False)
    assert '"Search"' in out


# ---- scrolling: only the part near the viewport is listed --------------------------------------------------------------
def _long_page():
    top, far, far2 = el(1, name="Search"), el(2, name="Subscribe"), el(3, name="Privacy")
    lines = [Line(kind="heading", text="Find flights", x=20, y=60), ctl(top, y=100),
             Line(kind="heading", text="Newsletter", x=20, y=3000), ctl(far, y=3040),
             Line(kind="heading", text="Footer", x=20, y=3400), ctl(far2, y=3440)]
    return lines, [top, far, far2]


def test_below_the_fold_becomes_an_outline():
    lines, els = _long_page()
    out = render(obs(lines, els))
    assert '[1] button "Search"' in out
    assert '"Subscribe"' not in out and '"Privacy"' not in out
    assert "below the view, 2 controls" in out and "## Newsletter (1)" in out and "## Footer (1)" in out


def test_scrolling_moves_the_window():
    lines, els = _long_page()
    out = render(obs(lines, els, scroll_y=2800))
    assert '"Subscribe"' in out and '"Search"' not in out and "above the view, 1 control" in out


def test_focused_and_invalid_fields_are_never_left_out():
    lines, els = _long_page()
    far = el(7, "textbox", "City", states=["focused"])  # an autocomplete box is last in the DOM but belongs to the view
    bad = el(8, "textbox", "Email", states=["invalid"], validation="Enter an email")
    out = render(obs(lines + [ctl(far, y=3500), ctl(bad, y=3600)], els + [far, bad]))
    assert '[7] textbox "City" focused' in out and "(error:" in out


def test_carousel_cards_off_to_the_side_are_counted():
    cards = [el(10 + k, name=f"Card {k}") for k in range(6)]
    lines = [ctl(c, x=40 + 400 * k, y=300) for k, c in enumerate(cards)]
    out = render(obs(lines, cards))
    assert '"Card 0"' in out and '"Card 5"' not in out and "off to the side" in out


def test_unknown_positions_are_always_listed():
    e = el(1, name="Frame button")
    assert '"Frame button"' in render(obs([ctl(e, x=None, y=None)], [e]))
    assert '"Frame button"' in render(Observation(url="u", lines=[ctl(e, x=None, y=None)], elements=[e]))  # no viewport known


# ---- what the model is told about the last action ---------------------------------------------------------------------
def test_changes_say_how_many_lines_went_not_which():
    out = render_changes(Changes(disappeared=[f"line {i}" for i in range(12)], appeared=["Type to search…"]))
    assert "+ Type to search…" in out and "12 text lines gone" in out and "line 3" not in out


def test_history_line_has_no_empty_brackets_and_newest_skips_changes():
    ev = Event(n=2, url="u", page="p", action={"kind": "click"}, status="ok", message="clicked [99] x -> page changed",
               changes=Changes(appeared=["Type to search…"]))
    assert "Type to search" in format_event(ev)
    assert "Type to search" not in format_event(ev, with_changes=False)
    gone_only = Event(n=1, url="u", page="p", action={"kind": "click"}, status="ok", message="clicked",
                      changes=Changes(disappeared=["a"]))
    assert format_event(gone_only).endswith("clicked")


# ---- perception: positions and modal detection --------------------------------------------------------------------------
def test_layout_positions_reads_main_document_only():
    snap = {"documents": [
        {"nodes": {"backendNodeId": [1, 2, 3]}, "layout": {"nodeIndex": [1, 2], "bounds": [[10, 200, 5, 5], [30, 900, 5, 5]]},
         "scrollOffsetY": 120},
        {"nodes": {"backendNodeId": [77]}, "layout": {"nodeIndex": [0], "bounds": [[0, 0, 1, 1]]}}]}
    pos, scroll = layout_positions(snap)
    assert pos == {2: (10.0, 200.0), 3: (30.0, 900.0)} and scroll == 120.0
    assert layout_positions({}) == ({}, 0.0)


def _node(nid, role, name="", kids=(), bid=None, **props):
    return {"nodeId": nid, "role": {"value": role}, "name": {"value": name}, "childIds": list(kids),
            "backendDOMNodeId": bid, "ignored": False,
            "properties": [{"name": k, "value": {"value": v}} for k, v in props.items()]}


def _walk(dom, overlay, root_kids, dialog_role="dialog"):
    nodes = [_node("1", "RootWebArea", kids=root_kids, bid=1),
             _node("2", "generic", kids=["3"], bid=2), _node("3", dialog_role, kids=["4", "5"], bid=3),
             _node("4", "heading", "Notice", bid=4), _node("5", "button", "OK", bid=5),
             _node("10", "heading", "Search flights", bid=10),
             _node("20", "textbox", "From", bid=20, focusable=True, editable=True),
             _node("31", "StaticText", "Footer text", bid=31)]
    tops = {10: (20.0, 100.0), 20: (20.0, 160.0), 31: (20.0, 2500.0), 5: (400.0, 300.0)}
    lines, elements = [], []
    w = _FrameWalker(nodes, dom, None, "https://x.test/", "", None, lines, elements, set(), overlay, tops)
    w.walk(nodes[0], "")
    w.flush()
    _fill_positions(lines)
    return w, lines


def test_walker_marks_real_modals_and_positions_lines():
    dom = {2: DomInfo("DIV", {}), 3: DomInfo("DIV", {"role": "dialog"}), 5: DomInfo("BUTTON", {}),
           20: DomInfo("INPUT", {"type": "text"})}
    w, lines = _walk(dom, 2, ["2", "10", "20", "31"])
    assert w.overlay_hard and [l.text for l in w.overlay_lines if l.kind == "heading"] == ["Notice"]
    assert [(l.kind, l.y) for l in lines] == [("heading", 100.0), ("control", 160.0), ("text", 2500.0)]


def test_walker_does_not_call_a_plain_fixed_layer_a_modal():
    dom = {2: DomInfo("DIV", {}), 3: DomInfo("DIV", {}), 5: DomInfo("BUTTON", {})}
    w, _ = _walk(dom, 2, ["2", "10", "20", "31"], dialog_role="generic")
    assert w.overlay_lines and not w.overlay_hard


def test_aria_modal_attribute_counts():
    dom = {2: DomInfo("DIV", {"aria-modal": "true"}), 5: DomInfo("BUTTON", {})}
    w, _ = _walk(dom, 2, ["2", "10", "20", "31"])
    assert w.overlay_hard


def test_walker_excludes_month_grid_wrapper_but_keeps_day_controls():
    nodes = [
        _node("1", "RootWebArea", kids=["2", "3", "4"], bid=1),
        _node("2", "button", "December 2026 SU MO TU WE TH FR SA", bid=2),
        _node("3", "button", "December 3, 2026", bid=3),
        _node("4", "button", "Next month", bid=4),
    ]
    dom = {2: DomInfo("DIV", {}), 3: DomInfo("BUTTON", {}), 4: DomInfo("BUTTON", {})}
    lines, elements = [], []
    w = _FrameWalker(nodes, dom, None, "https://x.test/", "", None, lines, elements, {2, 3, 4}, None,
                     {2: (20.0, 100.0), 3: (40.0, 120.0), 4: (60.0, 100.0)})
    w.walk(nodes[0], "")
    w.flush()
    assert not any(e.name == "December 2026 SU MO TU WE TH FR SA" for e in elements)
    assert any(e.name == "December 3, 2026" for e in elements)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in tests:
        f()
        print("ok  ", n)
    print(f"{len(tests)} passed")