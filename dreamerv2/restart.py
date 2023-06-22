from dreamerv2 import train
import ruamel.yaml as yaml
import pathlib
import common

logdir = pathlib.Path('~/logdir/crafter/dreamerv2_cr/1')
config = yaml.safe_load((pathlib.Path(f'{str(logdir)}/config.yaml')).read_text())
config = common.Config(config)
train.main(logdir, config)
print('done')
