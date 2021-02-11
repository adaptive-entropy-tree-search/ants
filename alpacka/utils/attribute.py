"""Handling attributes helpers."""

import copy
import functools


def recursive_getattr(obj, path, *default):
    attrs = path.split('.')
    try:
        return functools.reduce(getattr, attrs, obj)
    except AttributeError:
        if default:
            return default[0]
        raise


def recursive_setattr(obj, path, value):
    pre, _, post = path.rpartition('.')
    return setattr(recursive_getattr(obj, pre) if pre else obj,
                   post,
                   value)


def deep_copy_without_fields(obj, fields_to_be_omitted):
    values_to_save = [getattr(obj, field_name)
                      for field_name in fields_to_be_omitted]
    for field_name in fields_to_be_omitted:
        setattr(obj, field_name, None)

    new_obj = copy.deepcopy(obj)

    for field_name, val in zip(fields_to_be_omitted, values_to_save):
        setattr(obj, field_name, val)

    return new_obj


def deep_copy_merge(obj_1, obj_2, fields_from_obj_2):
    values_to_plug = [getattr(obj_2, field_name)
                      for field_name in fields_from_obj_2]
    new_obj = copy.deepcopy(obj_1)
    for field_name, val in zip(fields_from_obj_2, values_to_plug):
        setattr(new_obj, field_name, val)

    return new_obj
