# CoRGI Production Logging Configuration
#
# To optimize performance in production, set logging level to WARNING
# This reduces I/O overhead from debug/info statements
#
# Usage:
#   export CORGI_LOG_LEVEL=WARNING
#
# Or in Python:
#   import os
#   os.environ["CORGI_LOG_LEVEL"] = "WARNING"
#
# Expected improvement: 5-10% speedup by reducing logging I/O

import logging
import os


def configure_production_logging():
    """
    Configure logging for production use.

    Sets logging level to WARNING to reduce I/O overhead from
    excessive debug/info logging during inference.
    """
    log_level = os.getenv("CORGI_LOG_LEVEL", "INFO").upper()

    # Set root logger level
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Also set specific loggers to same level
    for logger_name in ["corgi", "transformers", "torch"]:
        logging.getLogger(logger_name).setLevel(
            getattr(logging, log_level, logging.INFO)
        )

    if log_level == "WARNING":
        print(
            "✓ Production logging enabled (WARNING level) - optimized for performance"
        )

    return logging.getLogger("corgi")


# Auto-configure on import if CORGI_LOG_LEVEL is set
if "CORGI_LOG_LEVEL" in os.environ:
    configure_production_logging()
