#! /usr/bin/env python3
"""A module defining the :class:`Plugin` base class.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qmmm_pme.common.core import Core


class Plugin(ABC):
    """The base class for creating QM/MM/PME plugins.
    """
    _keys = []
    _modifieds = []

    @abstractmethod
    def modify(self, modifable: Core) -> None:
        """Perform pluggable modifications on object inheriting from
        :class:`Core` requested in the keys class attribute.

        :param modifiable: An instance of a class inheriting from
            :class:`Core`.
        """
