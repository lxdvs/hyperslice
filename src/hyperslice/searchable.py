"""Searchable dropdowns, from panel-material-ui, without its app-wide restyle.

Importing panel_material_ui switches ``pn.config.design`` to Material for
every Panel component, restyling the tabs, buttons and sliders too. The
previous design is put back straight after the import, so only the selects
made here look Material.
"""

from __future__ import annotations

from typing import Any

import panel as pn

_design = pn.config.design
import panel_material_ui as pmui  # noqa: E402

pn.config.design = _design


#: Bokeh's widget font, which the filter sliders' labels use, so the
#: dropdowns read as part of the same column rather than as Material inserts.
WIDGET_FONT = "Helvetica, Arial, sans-serif"
WIDGET_FONT_SIZE = "12px"


def searchable_select(label: str, **params: Any) -> pmui.Select:
    """A compact dropdown whose list opens with a box to type into and narrow it.

    The theme reaches the dropdown's menu too, which Material renders apart
    from the box itself.
    """
    font = {"fontFamily": WIDGET_FONT, "fontSize": WIDGET_FONT_SIZE}
    return pmui.Select(  # type: ignore[no-untyped-call]
        label=label,
        searchable=True,
        size="small",
        theme_config={"typography": {"fontFamily": WIDGET_FONT, "body1": font}},
        **params,
    )
