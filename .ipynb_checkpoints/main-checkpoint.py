import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from enum import Enum, auto
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time

from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider, train_collate
from src.utils import save_checkpoint, load_checkpoint  # Import the checkpoint functions

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    gt_flow = gt_flow.permute(0, 3, 1, 2)  # Change shape from [B, 480, 640, 2] to [B, 2, 480, 640]
    print(f"pred_flow shape: {pred_flow.shape}, gt_flow shape: {gt_flow.shape}")  # Shape check

    pred_flow = F.interpolate(pred_flow, size=gt_flow.shape[2:], mode='bilinear', align_corners=True)
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataloader setup
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    train_data = DataLoader(train_set,
                            batch_size=args.data_loader.train.batch_size // 2,  # Reduced batch size
                            shuffle=args.data_loader.train.shuffle,
                            collate_fn=collate_fn,
                            drop_last=False)
    test_data = DataLoader(test_set,
                           batch_size=args.data_loader.test.batch_size,
                           shuffle=args.data_loader.test.shuffle,
                           collate_fn=collate_fn,
                           drop_last=False)

    # Model setup
    model = EVFlowNet(args.train).to(device)

    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    accumulation_steps = 2  # Accumulate gradients over this many steps

    # Checkpoint path
    checkpoint_path = 'checkpoint.pth'
    start_epoch = 0
    best_loss = float('inf')

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer)

    # Training loop
    model.train()
    for epoch in range(start_epoch, args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch + 1))
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)  # [B, 8, 480, 640] (2 frames concatenated)
            ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]
            
            # Forward pass
            flow_dict, _ = model(event_image)  # Unpack model output
            flow = flow_dict['flow0']  # Use the appropriate flow output for error computation
            
            # Compute EPE error
            loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
            loss = loss / accumulation_steps  # Normalize loss for gradient accumulation
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            print(f"batch {i} loss: {loss.item() * accumulation_steps}")  # Multiply back to original loss
            total_loss += loss.item() * accumulation_steps
        avg_loss = total_loss / len(train_data)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

        if avg_loss <= 2.8:  # Stop training condition
            print(f"Loss has reached the threshold of 2.8. Stopping training.")
            break

        # Save checkpoint if the current loss is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)

    # Save final model
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"checkpoints/model_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluation
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        first_batch_event_image = None
        for i, batch in enumerate(tqdm(test_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            if i == 0:
                first_batch_event_image = event_image.clone()
            
            batch_flow_dict, _ = model(event_image)  # Unpack model output
            batch_flow = batch_flow_dict['flow0']  # Use the appropriate key for the flow

            resized_batch_flow = F.interpolate(batch_flow, size=(480, 640), mode='bilinear', align_corners=False)
            flow = torch.cat([flow, resized_batch_flow], 0)
        
        # Add the first batch prediction as the first frame
        first_batch_flow_dict, _ = model(first_batch_event_image)
        first_batch_flow = first_batch_flow_dict['flow0']
        resized_first_batch_flow = F.interpolate(first_batch_flow, size=(480, 640), mode='bilinear', align_corners=False)
        flow = torch.cat([resized_first_batch_flow, flow], 0)

    # Debugging output before saving
    print(f"flow shape: {flow.shape}")

    # Save submission
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
