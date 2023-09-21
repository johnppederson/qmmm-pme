#! /usr/bin/env python3
"""A sub-package containing interfaces to external software.
"""
from __future__ import annotations

from configparser import ConfigParser
from os import listdir

from .interface import SystemTypes
config = ConfigParser()
config.read("./interfaces.conf")
del ConfigParser

qm_name = config["DEFAULT"]["QMSoftware"].lower()
mm_name = config["DEFAULT"]["MMSoftware"].lower()
del config

file_names = listdir()
del listdir

for name in file_names:
    if qm_name in name:
        qm_factories = __import__(name).__dict__["FACTORIES"]
        QMSettings = __import__(name).__dict__["Settings"]
    if mm_name in name:
        mm_factories = __import__(name).__dict__["FACTORIES"]
        MMSettings = __import__(name).__dict__["Settings"]

del file_names
del name
del qm_name
del mm_name

__author__ = "John Pederson"
