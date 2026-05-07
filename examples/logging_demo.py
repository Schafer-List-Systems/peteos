#!/usr/bin/env python3
"""
Logging Demo

This example demonstrates how to use the Peteos logging system with different
log levels and debug mode.

Usage:
    # Default info level
    PYTHONPATH=/home/frygge/projects/private/peteos python examples/logging_demo.py

    # Debug level
    PYTHONPATH=/home/frygge/projects/private/peteos python examples/logging_demo.py --debug

    # Write to file
    PYTHONPATH=/home/frygge/projects/private/peteos python examples/logging_demo.py --log-file /tmp/peteos.log
"""

import argparse
import asyncio
import sys

from peteos.logger import setup_logging, get_logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Peteos Logging Demo")
    parser.add_argument(
        "--level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Write logs to file instead of stdout"
    )

    args = parser.parse_args()

    # Configure logging
    setup_logging(
        level=args.level,
        debug=args.debug,
        log_to_file=args.log_file
    )

    logger = get_logger(__name__)

    print("=" * 60)
    print("  Peteos Logging Demo")
    print("=" * 60)
    print(f"Log level: {args.level}")
    print(f"Debug mode: {args.debug}")
    print(f"Output: {'file' if args.log_file else 'stdout'}")
    print("=" * 60)
    print()

    # Demonstrate different log levels
    logger.debug("This is a DEBUG message - only shown in debug mode")
    logger.info("This is an INFO message - shown by default")
    logger.warning("This is a WARNING message - always shown")
    logger.error("This is an ERROR message - always shown")
    logger.critical("This is a CRITICAL message - always shown")

    # Show example modules that use logging
    print("\n" + "-" * 60)
    print("Example modules using logging:")
    print("-" * 60)

    # Import modules that have logging to demonstrate they're configured
    from peteos import rolemanager

    logger.debug("Testing rolemanager module...")
    try:
        from peteos.role import Role
        rm = rolemanager.RoleManager()
        rm.register_role(Role(name="test", description="Test role"))
        logger.debug(f"Registered role: {rm.get_role('test').name}")
    except Exception as e:
        logger.error(f"Error: {e}")

    print("\n" + "-" * 60)
    print("Logging demo complete!")
    print("-" * 60)


if __name__ == "__main__":
    main()
