import os
import sys
from pathlib import Path

from omegaconf import OmegaConf
import datetime



def setup_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    # parent_dir : ['/homes/ypark/code/segmentation/']

    args = sys.argv
    print(args)

    current_dir = Path(__file__).resolve().parent
    conf_dir = current_dir / "conf"

    if len(args) > 1:
        config_file_name = args[1]
        config_file_path = conf_dir / f"{config_file_name}.yaml"
    else:
        config_file_name = "test"
        config_file_path = conf_dir / "test.yaml"
<<<<<<< HEAD
    config_file_path = config_file_path = current_dir / "conf" / f"{config_file_name}.yaml"
=======
    config_file_path = str(current_dir) + f"/conf/{config_file_name}.yaml"
>>>>>>> 8a19810 (polynomialスケジューラーとsingleで制御できるロジックを追加)
    if os.path.exists(config_file_path):
        cfg = OmegaConf.load(config_file_path)
    else:
        raise "No YAML file !!!"
    
    # 実行のhome dirを設定
    if cfg.default.home_dir is None:
        cfg.default.home_dir = f"{parent_dir}/"
    # dataset dirを設定
    if cfg.default.dataset_dir is None:
        cfg.default.dataset_dir = os.path.join(parent_dir, "dataset")
    

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


# 実行中の結果を保存するdir nameの決定
# 命名則 : 基本的には適用したaug name，RA関連だけ区別できるように詳しい設定
# cfg.default.add_filenameでdir nameを追加することができる
def get_filename(cfg):
    if cfg.default.add_filename is not None:
        file_name = cfg.default.add_filename
    else:
        file_name = ""
    
    
    for aug_name in cfg.augment.name:
        print(aug_name)
        if aug_name == "ra":
            file_name = f"{file_name}RA{cfg.augment.ra.num_op}"
            if cfg.augment.ra.weight == "random":
                file_name = f"{file_name}_Random"
            elif cfg.augment.ra.weight == "single":
                file_name = f"{file_name}_{cfg.augment.ra.single}"
            elif cfg.augment.ra.weight == "affinity":
                file_name = f"{file_name}_Affinity{cfg.augment.ra.softmax_t}"
            else:
                raise ValueError(f"Invalid RandAugment weight type... {cfg.augment.ra.weight}")
            
            if cfg.augment.ra.random_magnitude:
                file_name = f"{file_name}_Randmag"

            
        elif aug_name == "single":
            file_name = f"{file_name}SinglePass{cfg.augment.ra.softmax_t}_{cfg.augment.ra.init_epoch}"
            if cfg.augment.ra.random_magnitude:
                file_name = f"{file_name}_Randmag"

        elif aug_name == "w_ra":
            file_name = f"{file_name}WarmupRA{cfg.augment.ra.init_epoch}"
            if cfg.augment.ra.random_magnitude:
                file_name = f"{file_name}_Randmag"
        
        else:
            file_name = f"{file_name}{aug_name.capitalize()}"

        
    return file_name







