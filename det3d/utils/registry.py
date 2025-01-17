import inspect
import sys

from det3d import torchie


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + "(name={}, items={})".format(
            self._name, list(self._module_dict.keys())
        )
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        print(self._module_dict)
        print(key)
        # sys.exit()
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError(
                "module must be a class, but got {}".format(type(module_class))
            )
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(
                "{} is already registered in {}".format(module_name, self.name)
            )
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry, default_args=None, structure=None):
    print(structure)
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and "type" in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop("type")
    if torchie.is_str(obj_type):
        print("1")
        print(obj_type)
        # sys.exit()
        obj_cls = registry.get(obj_type)
        # print(self._module_dict)
        print(obj_cls)
        # sys.exit()
        if obj_cls is None:
            raise KeyError(
                "{} is not in the {} registry".format(obj_type, registry.name)
            )
    elif inspect.isclass(obj_type):
        print("2")
        obj_cls = obj_type
    else:
        print("3")
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(obj_type))
        )
    if default_args is not None:
        print("4")
        # print(default_args)
        for name, value in default_args.items():
            args.setdefault(name, value)
    print(structure)
    return obj_cls(**args)
