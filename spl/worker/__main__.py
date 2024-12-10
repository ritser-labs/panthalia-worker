# spl/worker/__main__.py
import os
from .config import args
from .logging_config import logger

# Before starting anything, set the environment variable for the Docker engine URL
os.environ["DOCKER_ENGINE_URL"] = args.docker_engine_url

if args.gui:
    from .gui import run_gui
    run_gui(args)
else:
    from .main_logic import main
    import asyncio
    asyncio.run(main())
