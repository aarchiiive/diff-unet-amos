
import os 
import glob 
import torch 
import torch.nn as nn

def delete_last_model(model_dir, symbol):
    last_model = glob.glob(f"{model_dir}/{symbol}*.pt")
    if len(last_model) != 0:
        os.remove(last_model[0])


def save_new_model_and_delete_last(model, optimizer, scheduler, epoch, best_mean_dice, id, save_path, delete_symbol=None):
    save_dir = os.path.dirname(save_path)

    os.makedirs(save_dir, exist_ok=True)
    # if delete_last_model is not None:
    #     delete_last_model(save_dir, delete_symbol)
    
    if isinstance(model, nn.DataParallel):
        model = model.module
        
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch' : epoch,
        'best_mean_dice' : best_mean_dice,
        'id' : id,
    }
    
    torch.save(state, save_path)

    print(f"model is saved in {save_path}")
