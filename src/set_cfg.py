import os
import sys
from pathlib import Path

from omegaconf import OmegaConf
import datetime



def setup_config():
    args = sys.argv
    print(args)

    current_dir = Path(__file__).resolve().parent
    conf_dir = current_dir / "conf"

    if len(args) > 1:
        config_file_name = args[1]
        config_file_path = conf_dir / f"{config_file_name}.yaml"
    else:
        print("aa")
        config_file_name = "test"
        config_file_path = conf_dir / "test.yaml"
    if os.path.exists(config_file_path):
        cfg = OmegaConf.load(config_file_path)
    else:
        raise "No YAML file !!!"

    # コマンドラインで受け取った引数とconfig_fileの情報をmerge
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args_list=args[2:]))
    file_name = get_filename(cfg)
    if "out_dir" not in cfg:
        output_dir_path = (
            f"{cfg.default.home_dir}"
            +f"{cfg.default.output_dir}/"
            + f"seed{cfg.default.seed}/"
            + f"{config_file_name}/"
            +f"{file_name}/"
        )
    else:
        output_dir_path = f"{cfg.out_dir}"

    if cfg.default.make_dir:
        print(f"MAKE DIR {output_dir_path}")
        os.makedirs(output_dir_path, exist_ok=True)

    out_dir_comp = {"out_dir": output_dir_path}
    cfg = OmegaConf.merge(cfg, out_dir_comp)

    config_name_comp = {"execute_config_name": config_file_name}
    cfg = OmegaConf.merge(cfg, config_name_comp)

    config_name_comp = {"override_cmd": args[2:]}
    cfg = OmegaConf.merge(cfg, config_name_comp)

    dt = datetime.datetime.today()
    datetime_comp = {"datetime": dt.strftime('%Y-%m-%d %H:%M:%S.%f')}
    cfg = OmegaConf.merge(cfg, datetime_comp)

    with open(output_dir_path + "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    return cfg


def add_config(cfg, config_name_comp: dict):
    config_file_path = cfg.out_dir + "config.yaml"
    if os.path.exists(config_file_path):
        print("### Add config")
        print(config_name_comp)
        #cfg = OmegaConf.merge(cfg, config_name_comp)
        for key, value in config_name_comp.items():
            cfg[key] = value
        with open(config_file_path, "w") as f:
            OmegaConf.save(cfg, f)
        return cfg
    else:
        raise "No YAML file !!!"
    
def override_original_config(cfg):
    config_file_path = cfg.out_dir + "config.yaml"
    if os.path.exists(config_file_path):
        original_cfg = OmegaConf.load(config_file_path)
    else:
        raise ValueError(f"cfg file {config_file_path} not found") 
    OmegaConf.merge(original_cfg, cfg)
    with open(config_file_path, "w") as f:
        OmegaConf.save(cfg, f)
    print("### Override config.")


def get_filename(cfg):
    if "my" in cfg.augment.name or "my_rand" in cfg.augment.name or cfg.optimizer.scheduler.name == "warmup":
        file_name = f"warmup{cfg.optimizer.hp.warmup_period}_"
    else:
        file_name = f"{cfg.optimizer.scheduler.name}_"
    if cfg.augment.dynamic:
        file_name = f"{file_name}Dynamic_"
    
    for aug_name in cfg.augment.name:
        print(aug_name)
        if aug_name == "rand":
            file_name = f"{file_name}Rand{cfg.augment.rand.num_op}"
            if cfg.augment.rand.weight == "random":
                file_name = f"{file_name}_Random"
            elif cfg.augment.rand.weight == "single":
                file_name = f"{file_name}_{cfg.augment.rand.single}"
            elif cfg.augment.rand.weight == "affinity":
                if cfg.augment.rand.scheduler:
                    file_name = f"{file_name}_Affinity{cfg.augment.rand.softmax_t}_{cfg.augment.rand.scheduler}Scheduled"
                else:
                    file_name = f"{file_name}_Affinity{cfg.augment.rand.softmax_t}_Fixed"
            else:
                raise ValueError(f"Invalid RandAugment weight type... {cfg.augment.rand.weight}")
            
        elif aug_name == "my":
            if cfg.augment.rand.scheduler:
                    file_name = f"{file_name}_My{cfg.augment.rand.softmax_t}_{cfg.augment.rand.scheduler}Scheduled"
            else:
                file_name = f"{file_name}_My{cfg.augment.rand.softmax_t}_Fixed"

        elif aug_name == "my_rand":
            file_name = f"{file_name}_MyRand"
        
        else:
            file_name = f"{file_name}_{aug_name.capitalize()}"
        
        
    return file_name





