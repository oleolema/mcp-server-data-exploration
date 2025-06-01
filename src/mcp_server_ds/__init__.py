from . import server
import asyncio

def main():
    """Main entry point for the package."""

    # 日志级别 debug
    server.logger.setLevel(server.logging.DEBUG)

    asyncio.run(server.main())


# Optionally expose other important items at package level
__all__ = ['main', 'server']