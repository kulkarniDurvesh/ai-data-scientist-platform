"""python -m mcp_server [--http PORT]: run the MCP server over stdio (default) or streamable HTTP."""

import argparse

from .server import server


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Data Scientist Platform MCP server")
    parser.add_argument("--http", type=int, metavar="PORT", help="Serve streamable HTTP on this port instead of stdio.")
    args = parser.parse_args()
    if args.http:
        server.run("streamable-http", port=args.http)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
