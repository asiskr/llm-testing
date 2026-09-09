"""MCP server exposing the order tools over the protocol, so any agent
can use them without importing this file."""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("orders")

ORDERS = {
    "A-1029": "Shipped, arriving Sep 11",
    "A-1030": "Processing",
}


@mcp.tool()
def get_order_status(order_id: str) -> str:
    """Get the delivery status of a customer order."""
    return ORDERS.get(order_id, "Order not found")


if __name__ == "__main__":
    mcp.run()