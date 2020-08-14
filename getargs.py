# -*- coding: utf-8 -*-
# Time    : 2020/7/15 14:04
# Author  : zlich
# Filename: getargs.py
import argparse
import sys
import yaml


def loadConfig(cfgPath):
    '''
    :param cfgPath: str, path(better absolute path)
    :return: modelConfig: dict, config
    # Loader=yaml.FullLoader
    '''
    with open(cfgPath) as f:
        modelConfig = yaml.load(f, Loader=yaml.FullLoader)
    # print(modelConfig)
    return argparse.Namespace(**modelConfig)


def cfgInfo(Cfg: argparse.Namespace):
    outstr = "Config:\n"
    outstr += '--' * 30 + '\n'
    for key, value in Cfg.__dict__.items():
        outstr += "{:10s}: {}\n".format(key, value)
    return outstr + '--' * 30


def getArgs(argv):
    sys.argv = argv
    # print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/bdd100k_deeplabv3p_mobilenetv2_apex.yaml",
                        type=str, help="model config path")

    # main
    parser.add_argument("--model", default="deeplabv3+", type=str, help="model name")

    # dir
    parser.add_argument("--log_dir", default="../checkpoints", type=str, help='path to checkpoint to store')

    # train
    parser.add_argument("--batch_size", default=16, type=int, help='train batch size')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')

    parser.add_argument('--resume', default=None, type=str, help='path to checkpoint to resume from')
    parser.add_argument('--finetune', default=None, type=str, help='path to finetune')
    parser.add_argument('--freeze_bn', default=False, action='store_true', help='Freeze BN params or not')

    args = parser.parse_args()
    yamlConfig = loadConfig(args.config)
    yamlConfig.__dict__.update(args.__dict__)

    return yamlConfig


def getArgs_(argv, configPath=None):
    sys.argv = argv
    if len(argv) == 1:
        if not configPath:
            assert False, "please check, no config file!"
        else:
            pass
    else:
        if not configPath:
            if 'config' not in argv[1]:
                assert False, "please check, no config file!"
            configPath = argv[1].split('=')[-1]
            sys.argv.pop(1)
        else:
            if 'config' not in argv[1]:
                pass
            else:
                configPath = argv[1].split('=')[-1]
                sys.argv.pop(1)

    yamlConfig = loadConfig(configPath)

    parser = argparse.ArgumentParser()
    for name, value in vars(yamlConfig).items():
        parser.add_argument("--{}".format(name), default=value,
                            type=type(value), help="{}".format(name))

    args = parser.parse_args()
    if isinstance(args.size, str):
        args.size = eval(args.size)

    return args


if __name__ == '__main__':
    a = getArgs(sys.argv)
    printInfo(a)
