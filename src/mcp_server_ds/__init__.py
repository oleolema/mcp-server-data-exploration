from . import server
import asyncio

from .script_runner import ScriptRunner


def main():
    """Main entry point for the package."""

    # 日志级别 debug
    server.logger.setLevel(server.logging.DEBUG)
    script_runner = ScriptRunner('/Users/leqiuhong/PycharmProjects/mcp-server-data-exploration/src/test/data')
    asyncio.run(server.main(script_runner))


# Optionally expose other important items at package level
__all__ = ['main', 'server']