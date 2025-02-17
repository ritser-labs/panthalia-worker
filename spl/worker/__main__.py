# spl/worker/__main__.py
import os
from .config import args
from .logging_config import logger
from .system_check import check_system_dependencies

# Set the Docker engine URL environment variable
os.environ["DOCKER_ENGINE_URL"] = args.docker_engine_url

# Check that required system dependencies are available
check_system_dependencies(args.gui)

if args.gui:
    from .gui import run_gui
    run_gui(args)
else:
    from .main_logic import main
    import asyncio
    asyncio.run(main())
