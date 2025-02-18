python -m PyInstaller --onedir --name panthalia_worker \
  --hidden-import=PyQt5.sip \
  --hidden-import=aiohttp \
  main_worker_launcher.py
