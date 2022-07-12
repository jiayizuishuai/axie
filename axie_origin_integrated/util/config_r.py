import os
import sys
import yaml
import collections
from copy import deepcopy


def _get_config(params, arg_name, info=None):
    config_name = None

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if "algo" in arg_name:
        path = "config/algos"
    elif "env" in arg_name:
        path = "config/envs"
    else:
        raise ValueError()

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), '..', path, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict
    else:
        raise ValueError()


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def read_config():
    params = deepcopy(sys.argv)

    with open(os.path.join(os.path.dirname(__file__), '..', "config", "default.yaml"), "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    env_config = _get_config(params, "--env_config")
    config_dict = recursive_dict_update(config_dict, env_config)

    for param in params:
        if param.startswith("--algo_config"):
            algo_name = param.split("=")[1]
            config_dict["algorithm"] = algo_name
            algo_config = _get_config(params, "--algo_config", env_config)
            config_dict = recursive_dict_update(config_dict, algo_config)

    return config_dict
