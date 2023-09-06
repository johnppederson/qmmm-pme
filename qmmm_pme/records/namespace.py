#! /usr/bin/env python3
"""A module to define the :class:`Namespace`, a dictionary with elements
accessible as attributes.
"""
from __future__ import annotations

from typing import Any
from typing import Optional


class Namespace(dict):
    """A class where elements are accesible as attributes, and observers
    are notified when an attribute is changed.
    """
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    # def __init__(self, observer: Any | None = None) -> None:
    #    dict.__setitem__(self, "observer", observer)

    # def __setattr__(self, attr: str, value: Any) -> None:
    #    if self.observer:
    #        self.observer.update("namespace." + attr, value)
    #    dict.__setitem__(self, attr, value)
