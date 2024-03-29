from experiments.batch.model.trainer import Trainer
from utils.parameter_parser import *


def main():
    # 示例用法
    parser = get_parser()
    args = parser.parse_args()

    # args.__setattr__("histogram", False)
    # args.__setattr__("epoch_start", 19)
    # args.__setattr__("model_train", False)
    args.__setattr__("model_name", "CGSNGin")
    args.__setattr__("dataset", "AIDS_700")

    # 如果提供了配置文件路径，从配置文件中读取参数并更新
    if args.config is not None:
        config = parse_config_file(args.config)
        update_args_with_config(args, config)

    tab_printer(args)

    trainer = Trainer(args)

    if args.epoch_start > 0:
        trainer.load(args.epoch_start)
    if args.model_train:
        for epoch in range(args.epoch_start, args.epoch_end):
            trainer.fit()
            trainer.save(epoch + 1)
            trainer.score('test')
    else:
        trainer.score('test')

if __name__ == "__main__":
    main()
