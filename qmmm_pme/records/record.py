#! /usr/bin/env python3
"""A module defining the :class:`Record` and :class:`Variable` classes.
"""
from __future__ import annotations

from typing import Any
from typing import Callable


class Variable:
    """A base class for wrapping a variable belonging to a
    :class:`Record`.
    """
    __slots__ = "_value", "_notifiers"
    _notifiers: list[Callable[[Any], None]] = []

    def register_notifier(self, notifier: Callable[..., None]) -> None:
        """Add a notifier to the :class:`Variable`, which will
        be called whenever the :class:`Variable` is changed.

        :param notifier: The method which will be called when the
            :class:`Variable` is changed.
        """
        self._notifiers.append(notifier)


class Record:
    """A base class for defining records.
    """
    ...
