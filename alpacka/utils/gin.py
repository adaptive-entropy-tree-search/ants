"""Utilities for Gin configs."""


def extract_bindings(config_str):
    config_str = config_str.replace('\\\n', '')

    config_str = config_str.replace('\n    ', '')

    config_str = config_str.replace('\n   ', '')

    sep = ' = '

    bindings = []
    for line in config_str.split('\n'):
        line = line.strip()
        if sep in line:
            chunks = line.split(sep)
            name = chunks[0].strip()
            value = sep.join(chunks[1:]).strip()
            bindings.append((name, value))
    return bindings
