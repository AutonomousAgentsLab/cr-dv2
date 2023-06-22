"""This is in dreamerv2 directory so that it is in same directory as
standard location for config.yaml file"""

import ruamel.yaml as yaml
import pathlib
import elements

import common
import sys


def parse_flags(flags):
    """
    Load configs and update parameters according to flags.
    :param flags: List of strings, i.e. ['--name', 'value']
    :return: config object, logdir
    """
    configs = yaml.safe_load((pathlib.Path(__file__).parent / 'configs.yaml').read_text())
    parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
    config = common.Config(configs['defaults'])
    parsed, remaining = common.Flags(configs=['defaults']).parse(argv=flags, known_only=True)
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)
    return config, logdir
