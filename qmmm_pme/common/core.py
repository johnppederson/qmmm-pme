#! /usr/bin/env python3
"""A module defining the :class:`Core` class, which is inherited by the
:class:`Calculator` and :class:`Integrator` classes.
"""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from qmmm_pme.records import State
from qmmm_pme.records import Topology

if TYPE_CHECKING:
    from qmmm_pme.plugins import Plugin


class Core(ABC):
    """Class for defining classes that are modifiable by plugins.
    """
    _plugins = []

    def __init__(self) -> None:
        self._state = State(self)
        self._topology = Topology(self)

    def register_plugin(self, plugin: Plugin) -> None:
        """Register any plugins applied to an instance of a class
        inheriting from :class:`Core`.

        :param plugin: An instance of a class inheriting from
            :class:`Plugin`.
        """
        self._plugins.append(plugin.__class__.__name__)
        plugin.modify(self)

    def active_plugins(self) -> list[str]:
        """Return the list of active plugins.

        :return: A list of the active plugins being employed by the
            class.
        """
        return self._plugins
