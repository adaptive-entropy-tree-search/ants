"""Operating system utilities."""

import contextlib
import os
import shutil


@contextlib.contextmanager
def _backup(path):
    (dir_path, file_name) = os.path.split(path)
    backup_path = os.path.join(dir_path, f'.{file_name}.backup')

    def remove(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    remove(backup_path)

    os.replace(path, backup_path)

    try:

        yield path

        remove(backup_path)
    except:

        remove(path)
        os.replace(backup_path, path)

        raise


@contextlib.contextmanager
def atomic_dump(paths):
    with contextlib.ExitStack() as stack:
        yield tuple(

            stack.enter_context(_backup(path)) if os.path.exists(path) else path
            for path in paths
        )
