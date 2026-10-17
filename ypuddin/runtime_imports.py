"""One lock for importing the lazily loaded model libraries from concurrent threads."""

import os
import threading

# Diffusers and Transformers load their classes on first use and import overlapping modules in different
# orders. When two service threads load them for the first time together, Python's import deadlock avoidance
# can hand one thread the other's unfinished module, which reads as "cannot import name". Every import of
# Diffusers, Transformers or Accelerate in this package holds this lock. It encloses only those import
# statements, never an import of this package, so it cannot wait on a module this package is importing.
LIBRARY_IMPORTS = threading.RLock()

if hasattr(os, "register_at_fork"):
    # A forked child (a Linux DataLoader worker) keeps only the forking thread, so a lock that another thread
    # held at the fork would stay held in the child; logging resets its locks the same way.
    os.register_at_fork(after_in_child=LIBRARY_IMPORTS._at_fork_reinit)
