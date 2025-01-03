# file: spl/models/__init__.py

"""
Minimal initialization for models, avoiding any references that cause circular imports.
We only import the actual ORM model classes from schema.py (or your submodules).
"""

# If your model classes are in .schema or individually in files, import them here:
from .schema import *  # or do from .schema import MyModel, AnotherModel, etc.

# Nothing else here! We do NOT import from spl.db.server.adapter or create sessions here.
