"""Golden tasks for the returns agent.

Data, not test functions: adding a case is a dict, not another copy of the
same three lines. Only 'id' and 'question' are required - each optional key is
asserted only when a row has it, so a case can pin a trajectory, a verdict,
both, or neither.

Trajectories here must not depend on today's date. Order 5100 was delivered
2026-08-20 and will fall outside the 30-day window, so its verdict is
deliberately not pinned - only its trajectory is.
"""

AGENT_GOLDENS = [
    {
        "id": "in_window",
        "question": "Can I return order 5100?",
        "expect_tools": ["get_order", "days_since"],
        "expect_args": {"get_order": {"order_id": "5100"}},
    },
    {
        "id": "final_sale",
        "question": "Can I return order 6402?",
        "expect_first_tool": "get_order",
        "expect_max_calls": 3,
        "expect_verdict": "REFUSED",
    },
    {
        "id": "expired_window",
        "question": "Can I return order 4821?",
        "expect_first_tool": "get_order",
        "expect_verdict": "REFUSED",
    },
    {
        "id": "unknown_order",
        "question": "Can I return order 9999?",
        "expect_first_tool": "get_order",
        "expect_args": {"get_order": {"order_id": "9999"}},
    },
    {
        "id": "messy_order_id",
        "question": "can I return order # 5100 please",
        "expect_first_tool": "get_order",
        "expect_args": {"get_order": {"order_id": "5100"}},
    },
    {
        "id": "no_order_id",
        "question": "Can I return something?",
        "expect_no_tools": True,
    },
]
