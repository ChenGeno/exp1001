#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import sys
from Control_Exp1001.utils import get_project_absoulte_path
import traceback


class SystemPathModifier:
    """
    """
    def __init__(self, add=None, delete=None) -> None:
        self.add = add
        self.delete = delete
        self.is_delete = False

    def __enter__(self):
        import sys
        sys.path.append(self.add)
        if self.delete in sys.path:
            sys.path.remove(self.delete)
            self.is_delete = True

    def __exit__(self, exc_type, exc_value, traceback):
        import sys
        sys.path.remove(self.add)
        if self.is_delete:
            sys.path.append(self.delete)


# 因为Systemmodeling目录与Control_Exp1001包含重名文件，比如common，
# 所以在import system_modeling的时候直接修改 sys.path
with SystemPathModifier(
        add=os.path.join(get_project_absoulte_path(), 'system_modeling'),
        delete=get_project_absoulte_path()
):
    try:
        from model_render import SystemModel
    except ImportError as e:
        from Control_Exp1001.system_modeling.model_render import SystemModel
        traceback.print_exc()
