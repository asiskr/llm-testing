"""Tools the returns agent can call.

Two things the prompt cannot supply: today's date (changes every request) and
a specific order's record (different per customer). ORDERS is a stand-in for a
database - fixed rows so tests stay deterministic.
"""

from datetime import date

ORDERS = {
    "4821": {"delivered": "2026-03-03", "on_sale": False},
    "5100": {"delivered": "2026-08-20", "on_sale": False},
    "6402": {"delivered": "2026-08-25", "on_sale": True},
}


def get_order(order_id):
    """Look up one order. Returns None when the id is unknown."""
    return ORDERS.get(str(order_id).strip())


def days_since(iso_date):
    """Whole days between an ISO date and today."""
    return (date.today() - date.fromisoformat(iso_date)).days


def today():
    """Today's date as an ISO string. The model has no clock."""
    return date.today().isoformat()


# What the model sees. The model never runs these - it replies with a name and
# arguments, our loop runs the real function and feeds the result back.
# Descriptions say when to use a tool, not just what it does - that text is the
# only thing the model uses to choose.
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up a ShopEasy order by its id. Returns the "
            "delivery date and whether the item was on sale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order number, digits only, e.g. 4821",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "days_since",
            "description": "Number of whole days between a delivery date and "
            "today. Use this instead of calculating dates yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "iso_date": {
                        "type": "string",
                        "description": "A date in YYYY-MM-DD format",
                    }
                },
                "required": ["iso_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "today",
            "description": "Today's date in YYYY-MM-DD format. The model has no "
            "clock - call this instead of stating a date from memory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
