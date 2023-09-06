#! /usr/bin/env python3
"""A module defining the class and metaclass for records.
"""
from __future__ import annotations

import warnings
from typing import Any
from typing import Optional


def recordclass(
        cls_name: str,
        cls_parents: tuple[Any, ...],
        cls_attrs: dict,
) -> type:
    """Metaclass for constructing record classes.

    This works by creating properties for each specified 'hidden' class
    variable.  The setter for these properties edits the class variable,
    so each instance of the record class will then have access to the
    updated record.  Observers of the record class are barred from
    updating the record, but they will be notified when the record is
    changed.

    :param cls_name: The name of the class to be constructed.
    :param cls_parents: The base classes of the class to be constructed.
    :param cls_attrs: The attributes of the class to be constructed
    :return: The modified class, now a record class.
    """
    records = [
        attr for attr, value in cls_attrs.items()
        if not callable(value)
        and not attr.startswith("__")
        and attr.startswith("_")
    ]
    prop_attrs = {}
    for prop in records:
        prop_attrs.update(_property_factory(prop))
    cls_attrs.update(prop_attrs)
    return type(cls_name, cls_parents, cls_attrs)


def _property_factory(_prop: str) -> dict[str: property]:
    """A method to create a property from a hidden class variable.

    :param _prop: The hidden class variable name.
    :return: The property name and object in dictionary format.
    """
    prop = _prop[1:]

    def getter(self):
        return getattr(self.__class__, _prop)

    def setter(self, value):
        if self.observer:
            warnings.warn(
                (
                    f"Object '{self.observer}', an observer of the "
                    + f"{self.__class__.__name__} record class, attempted "
                    + f"to change the '{prop}' record.  This behaviour "
                    + "is not permitted, so the record will remain "
                    + "unchanged."
                ),
                RuntimeWarning,
            )
        else:
            setattr(self.__class__, _prop, value)
    return {prop: property(getter, setter)}


class Record(metaclass=recordclass):
    """A base class for defining records, a borg-like class which stores
    data for all classes to see.  It is also a subject of observer
    objects, which are notified when a value is changed.

    :param observer: A class observing and accessing the record.
    """
    _registry = []

    def __init__(self, observer: Any | None = None) -> None:
        self.observer = observer
        if observer:
            self.registry.append(observer)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "observer":
            super().__setattr__(attr, value)
        if not hasattr(self, attr):
            raise AttributeError(
                (
                    f"Record '{attr}' is not a valid record of the "
                    + f"{self.__class__.__name__} record class."
                ),
            )
        if not isinstance(value, type(getattr(self, attr))):
            raise TypeError(
                (
                    f"Attempted to set record '{attr}' of the "
                    + f"{self.__class__.__name__} record class to type "
                    + f"'{type(value)}', but {attr} can only be of type "
                    + f"'{type(getattr(self, attr)).__name__}'"
                ),
            )
        for observer in self.registry:
            observer.update("record." + attr, value)
        super().__setattr__(attr, value)
