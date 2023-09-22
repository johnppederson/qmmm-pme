#! /usr/bin/env python3
"""A module for handling software interface imports.
"""
from __future__ import annotations

from configparser import ConfigParser
from importlib import import_module
from os import listdir
from typing import Callable
from typing import TYPE_CHECKING

from importlib_resources import files

from .interface import SystemTypes

if TYPE_CHECKING:
    from .interface import SoftwareSettings, SoftwareInterface
    from types import ModuleType
    from typing import Dict
    Factory = Dict[SystemTypes, Callable[[SoftwareSettings], SoftwareInterface]]

MODULE_PATH = files("qmmm_pme") / "interfaces"


def _import(module_name: str) -> ModuleType:
    module = import_module(
        ".interfaces." + module_name.split(".")[0], package="qmmm_pme",
    )
    return module


def _get_factory(module_name: str) -> Factory:
    return getattr(_import(module_name), "FACTORIES")


def get_software_factories(field: str) -> Factory:
    config = ConfigParser()
    config.read(MODULE_PATH / "interfaces.conf")
    software_name = config["DEFAULT"][field].lower()
    file_names = listdir(MODULE_PATH)
    for name in file_names:
        if software_name in name:
            factories = _get_factory(name)
    return factories
