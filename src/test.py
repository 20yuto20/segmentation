import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from set_cfg import setup_config
from dataloader import get_dataloader
from utils.common import setup_device
from utils.suggest import suggest_network

def test(cfg, device, model, test_loader):
    model.eval()
    
    output_dir = os.path.join(cfg.test.result_dir, "final_result_segmentation")
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader, desc="Generating Predicitons!!")):
            image = sample['image'].to(device)
            
            output = model(image)
            pred = torch.argmax(output, dim=1)
            
            for j, pred_single in enumerate(pred):
                pred_img = Image.fromarray(pred_single.byte().cpu().numpy().astype(np.uint8))
                pred_img.save(os.path.join(output_dir, f"pred_{i*cfg.learn.batch_size+j}.png"))
    
    print(f"Predicitons saved in {output_dir}")

def main(cfg):
    device = setup_device(cfg)
    
    model = suggest_network(cfg)
    model.to(device)
    
    # Load best model in the all experiments
    # This is because we cannnot send the each experiment results according to the PASCAL VOC home page.
    ###############
    # Best Practice
    # The VOC challenge encourages two types of participation: (i) methods which are trained using only the provided "trainval" (training + validation) data; (ii) methods built or trained using any data except the provided test data, for example commercial systems. In both cases the test data must be used strictly for reporting of results alone - it must not be used in any way to train or tune systems, for example by runing multiple parameter choices and reporting the best results obtained.
    # If using the training data we provide as part of the challenge development kit, all development, e.g. feature selection and parameter tuning, must use the "trainval" (training + validation) set alone. One way is to divide the set into training and validation sets (as suggested in the development kit). Other schemes e.g. n-fold cross-validation are equally valid. The tuned algorithms should then be run only once on the test data.
    # In VOC2007 we made all annotations available (i.e. for training, validation and test data) but since then we have not made the test annotations available. Instead, results on the test data are submitted to an evaluation server.
    # Since algorithms should only be run once on the test data we strongly discourage multiple submissions to the server (and indeed the number of submissions for the same algorithm is strictly controlled), as the evaluation server should not be used for parameter tuning.
    # We encourage you to publish test results always on the latest release of the challenge, using the output of the evaluation server. If you wish to compare methods or design choices e.g. subsets of features, then there are two options: (i) use the entire VOC2007 data, where all annotations are available; (ii) report cross-validation results using the latest "trainval" set alone.
    ############### 
    
    best_model_path = cfg.test.best_model
    model.load_state_dict(torch.load(best_model_path))
    
    _, _, test_loader = get_dataloader(cfg)
    
    test(cfg, device, model, test_loader)
    

if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)