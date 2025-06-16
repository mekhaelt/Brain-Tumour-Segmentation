""" 
author Cecilia Diana-Albelda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from tqdm import tqdm
import cfg
from utils import *

def print_gpu_memory_usage():
    """Print current GPU memory usage"""
    print("\nGPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Max cached: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

def print_tensor_memory_usage(tensor, name=""):
    """Print memory usage of a specific tensor"""
    if tensor is None:
        return
    size_bytes = tensor.element_size() * tensor.nelement()
    print(f"\nTensor {name} Memory Usage:")
    print(f"Shape: {tensor.shape}")
    print(f"Size: {size_bytes / 1024**3:.2f} GB")
    print(f"Device: {tensor.device}")

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True


def train_sam(args, net: nn.Module, optimizer,  train_loader,
          epoch, writer, vis = 50):
    print("Starting train_sam function...")
    # print_gpu_memory_usage()  # Initial memory usage
    
    hard = 0
    epoch_loss_values = 0.0
    epoch_loss = 0.0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    print(f"Using device: {GPUdevice}")

    if args.thd:
        sigmoid = nn.Sigmoid()
        lossfunc = nn.BCELoss() 
    else:
        lossfunc = criterion_G

    print("Starting training loop...")
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            #print("Processing batch...")
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            #print(f"Image shape: {imgs.shape}, Mask shape: {masks.shape}")
            # print_tensor_memory_usage(imgs, "Input images")
            # print_tensor_memory_usage(masks, "Input masks")
            
            # If not enough GPU, uncomment the following 3 lines
            i_slices = SelectEquiSlices(4, masks)
            imgs = imgs[:,:,:,:,i_slices] 
            masks = masks[:,:,:,:,i_slices]
            #print("After selecting slices:")
            # print_tensor_memory_usage(imgs, "Sliced images")
            # print_tensor_memory_usage(masks, "Sliced masks")

            if 'pt' not in pack:
                #print("Generating click prompt...")
                a = masks
                imgs, pt, masks = generate_click_prompt(imgs, masks)
                #print("After click prompt:")
                # print_tensor_memory_usage(imgs, "Images after click prompt")
                # print_tensor_memory_usage(masks, "Masks after click prompt")
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']
            #print(f"Processing file: {name}")

            if args.thd:
                #print("Applying threshold transformations...")
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                point_labels = torch.ones(imgs.size(0))
                # Project generated points to the new image size (next line) 
                pt = torch.Tensor(numpy.array([((pt[i].detach().cpu().numpy()*(args.out_size,args.out_size))/masks.shape[2:]) for i in range (pt.shape[0])]))
                imgs = torchvision.transforms.Resize((args.image_size,args.image_size), antialias=None)(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size), antialias=None)(masks)
                #print("After threshold transformations:")
                # print_tensor_memory_usage(imgs, "Images after threshold")
                # print_tensor_memory_usage(masks, "Masks after threshold")
            
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels[0] != -1:
                #print("Processing point labels...")
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice) # shape: (b_size, 2) 
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice) # shape: (b_size) 
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch) # shape: (1, b_size, 2) 

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            # imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            #print("Setting up model parameters...")
            '''Train'''
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters(): 
                    if ("Adapter" in n): 
                        value.requires_grad = True
                    else:
                        value.requires_grad = False
            if args.mod == 'sam':
                for n, value in net.image_encoder.named_parameters():
                    value.requires_grad = False 
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder) 
                if args.mod == 'sam_adalora':
                    # Initialize the RankAllocator 
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10, 
                        total_step=3000, beta1=0.85, beta2=0.85, 
                    )
            else:
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = True

            if args.four_chan == True:
                for n, value in net.image_encoder.named_parameters(): 
                    if ('patch_embed' in n): # 1st layer 
                        value.requires_grad = True 
            
            #print("Running image encoder...")
            # print_gpu_memory_usage()  # Memory before image encoder
            
            # Enable gradient checkpointing for attention blocks
            if hasattr(net.image_encoder, 'blocks'):
                for block in net.image_encoder.blocks:
                    if hasattr(block, 'attn'):
                        block.attn.use_checkpoint = True
            
            imge = net.image_encoder(imgs)
            #print("After image encoder:")
            # print_tensor_memory_usage(imge, "Image embeddings")
            # print_gpu_memory_usage()  # Memory after image encoder
            
            #print("Running prompt encoder...")
            with torch.no_grad():
                if args.net == 'sam':
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                    )
                    #print("After prompt encoder:")
                    # print_tensor_memory_usage(se, "Sparse embeddings")
                    # print_tensor_memory_usage(de, "Dense embeddings")
                    
            #print("Running mask decoder...")
            if args.net == 'sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                )
                #print("After mask decoder:")
                # print_tensor_memory_usage(pred, "Predictions")
 
            # Resize to the ordered output size
            pred = F.interpolate(pred, size=(args.out_size, args.out_size))

            #print("Computing loss...")
            loss = lossfunc(sigmoid(pred), masks)
            # print_gpu_memory_usage()  # Memory after loss computation

            epoch_loss += loss.item()
            epoch_loss_values += 1
            pbar.set_postfix(**{'loss (batch)': loss})

            #print("Backpropagating...")
            if args.mod == 'sam_adalora':
                (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()

            current_lr = args.lr 
            
            optimizer.zero_grad()
            # print_gpu_memory_usage()  # Memory after optimization step

            '''vis images'''
            #FOR VISUALIZATION
            # if args.vis > 0 and ind % args.vis == 0:
            #     print('VIS: ',  vis)
            #     namecat = 'Train'
            #     for na in name:
            #         namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
            #     vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()
            # print_gpu_memory_usage()  # Memory at end of batch

    return epoch_loss/epoch_loss_values, current_lr

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()
    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    

    if args.thd:
        sigmoid = nn.Sigmoid()
        lossfunc = nn.BCELoss() 
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)# [0,:,:,:,:,:] 
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)# [0,:,:,:,:,:] 

            # If not enough GPU, uncomment the following 4 lines
            num_slices = 4 
            i_slices = [random.randint(0,masksw.shape[-1]-1) for i in range(num_slices)] 
            imgsw = imgsw[:,:,:,:,i_slices] 
            masksw = masksw[:,:,:,:,i_slices] 

            if 'pt' not in pack:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)

            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    point_labels = torch.ones(imgs.size(0))
                    # Project generated points to the new image size (next line) 
                    pt = torch.Tensor(numpy.array([((pt[i].detach().cpu().numpy()*(args.out_size,args.out_size))/masks.shape[2:]) for i in range (pt.shape[0])]))
                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size), antialias=None)(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size), antialias=None)(masks)
                    
                    
                showp = pt

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)
                    
                    if args.net == 'sam':
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                        

                    if args.net == 'sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=False,
                        )
                   
                    # Resize to the ordered output size
                    pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                    tot += lossfunc(sigmoid(pred), masks)

                    # '''vis images'''
                    # if args.vis and ind % args.vis == 0:
                    #     namecat = 'Test'
                    #     for na in name:
                    #         img_name = na.split('/')[-1].split('.')[0]
                    #         namecat = namecat + img_name + '+'
                    #     vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)
                    

                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val ,  tuple([a/n_val for a in mix_res]) 

def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )

# if __name__ == "__main__":
#     # Import necessary modules
#     from torch.utils.tensorboard import SummaryWriter
#     from torch.utils.data import DataLoader
#     from models.sam.build_sam import build_sam_vit_b
#     from dataset import Brats
    
#     # Initialize args with required parameters
#     args = cfg.parse_args()
#     args.four_chan = True  # Override default since we're using 4-channel BRATS data
#     args.thd = True  # Enable 3D processing for medical imaging data
#     args.data_path = 'data'

#     # Initialize model using the build function
#     net = build_sam_vit_b(args, checkpoint="sam_vit_b_01ec64.pth")
#     net = net.to(GPUdevice)
    
#     # Create optimizer
#     optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
#     # Create data loader with BRATS dataset using the existing dataset class
#     train_dataset = Brats(args, data_path=args.data_path, mode='Training', prompt='click', plane=False)
#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
#     # Create tensorboard writer
#     writer = SummaryWriter('runs/test_train_sam')
    
#     # Test training for one epoch
#     epoch = 1
#     loss, lr = train_sam(args, net, optimizer, train_loader, epoch, writer, vis=args.vis)
#     print(f"Training completed. Loss: {loss:.4f}, Learning rate: {lr}")
    
#     # Validate after training
#     val_loss, val_metrics = validation_sam(args, train_loader, epoch, net)
#     print(f"[VAL] Epoch {epoch} completed. Loss: {val_loss:.4f}, Metrics: {val_metrics}")