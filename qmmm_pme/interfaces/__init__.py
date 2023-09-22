#! /usr/bin/env python3
"""A sub-package containing interfaces to external software.
"""
from __future__ import annotations

from .interface import MMSettings
from .interface import QMSettings
from .interface import SystemTypes
from .interface_manager import get_software_factories

mm_factories = get_software_factories("MMSoftware")
qm_factories = get_software_factories("QMSoftware")

del get_software_factories

__author__ = "John Pederson"
