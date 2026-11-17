"""Role tables. Same idea as Browser Use's ax module: a small set of roles are 'controls' the agent can act on."""

CONTROLS = frozenset(
    "button link textbox searchbox combobox listbox option checkbox radio switch slider spinbutton "
    "tab menuitem menuitemcheckbox menuitemradio treeitem".split()
)
# Text inside these is one line; text across them is not joined.
BLOCKS = frozenset({"paragraph", "listitem", "row", "LabelText"})
# Chrome's AX roles for native date/time inputs: their children are the browser's own widget parts.
NATIVE_INPUT_ROLES = frozenset({"Date", "DateTime", "InputTime"})
DATE_INPUT_TYPES = frozenset({"date", "time", "datetime-local", "month", "week"})
CHECKABLE = frozenset({"checkbox", "radio", "switch", "menuitemcheckbox", "menuitemradio"})
# Widgets that make sense with get_by_label.
LABELLED = frozenset({"textbox", "searchbox", "combobox", "select", "date", "checkbox", "radio", "spinbutton", "switch"})
