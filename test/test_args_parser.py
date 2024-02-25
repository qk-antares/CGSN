from utils.parameter_parser import *

if __name__ == "__main__":
    # 示例用法
    parser = get_parser()
    args = parser.parse_args()

    # 如果提供了配置文件路径，从配置文件中读取参数并更新
    if args.config is not None:
        config = parse_config_file(args.config)
        update_args_with_config(args, config)

    tab_printer(args)
