import argparse
import math
import json
import os
import random
from copy import deepcopy
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from utils import *
import models_adapter
from video_dataset import VideoDataset
from configs import DATASETS, CLIP_VIT_B16_PATH
import numpy as np
import shutil

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--warmup_epochs', type=int, default=3) 
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--auto_remove', action='store_true')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--resume', type=str)
    
    # ===== 【文本与泛化控制参数】 =====
    parser.add_argument('--label_csv', type=str, help='long-form (LLM-augmented) prompts CSV')
    parser.add_argument('--short_label_csv', type=str, default=None, help='short prompts CSV')
    parser.add_argument('--text_mix_ratio', type=float, default=0.5, help='0.5: 50%长文本 + 50%短文本')
    parser.add_argument('--mlm_label', type=str)
    
    parser.add_argument('--pretrain', type=str)
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys())
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--spatial_size', type=int, default=224)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--sampling_rate', type=int, default=0)
    parser.add_argument('--num_spatial_views', type=int, default=1)
    parser.add_argument('--num_temporal_views', type=int, default=1)
    parser.add_argument('--auto_augment', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize_type', type=str, default='random_resized_crop', choices=['random_resized_crop', 'random_short_side_scale_jitter'])
    parser.add_argument('--scale_range', type=float, nargs=2, default=[0.08, 1.0])
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=1)
    
    # ===== 【核心调优杠杆参数】 =====
    parser.add_argument('--grad_scale', type=float, default=0.3, help='GradientScale for decoupled FC branch')
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--head_lr_mult', type=float, default=2.0)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--use_fine_grained_contrastive', action='store_true')
    parser.add_argument('--distill_weight', type=float, default=2.0, help='冻结 CLIP 蒸馏权重')
    parser.add_argument('--mlm_weight', type=float, default=0.5, help='MLM 视觉文本细粒度对齐权重')
    parser.add_argument('--fc_grad_scale', type=float, default=None, help='FC 空间头梯度缩放(默认=grad_scale)')
    parser.add_argument('--oad_pe_type', type=str, default='sincos', choices=['sincos', 'learnable'], help='OAD 时空位置编码类型')
    parser.add_argument('--use_motion', action='store_true', help='启用运动差分分支(替代 cluster，喂闭集时序头；SSv2 用)')
    parser.add_argument('--disable_cluster', action='store_true', help='禁用 cluster 模块(该 layer 仍用于 OAD/motion 定位)')
    parser.add_argument('--use_text_cond_query', action='store_true',
                        help='启用 OAD 文本条件化 query。注意：要对零样本生效，必须同时开 --use_fine_grained_contrastive，否则 OAD 不进迁移路径，文本条件化对零样本无效')

    parser.add_argument('--data_root_train', type=str, default='/media/zhang/data2/zly/k400_train')
    parser.add_argument('--data_root_val', type=str, default='/media/zhang/data2/zly/k400_test')
    parser.add_argument('--train_list', type=str, default=None, help='覆盖 DATASETS[dataset][TRAIN_LIST]；few-shot 时指向 k-shot list')
    parser.add_argument('--val_list', type=str, default=None, help='覆盖 DATASETS[dataset][VAL_LIST]；一般 few-shot 仍用完整测试集，不用填')

    args = parser.parse_args()
    args.distributed = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("{}".format(args).replace(', ', ',\n'))

    print('creating model')
    classes_long, num_aug_long, text_dict_long, classes_dict = text_prompt(args.label_csv, args.mlm_label)
    mlm_label_dict = load_word_index_mapping(args.mlm_label)
    mlm_head_len = len(mlm_label_dict)

    target_classes = [""] * len(classes_dict)
    for k, v in classes_dict.items():
        # k 是整数ID，v 是字符串名称，直接反过来用即可
        target_classes[int(k)] = str(v).replace('_', ' ')

    text_dict_short = None
    num_aug_short = 0
    if args.short_label_csv is not None and args.text_mix_ratio > 0:
        _, num_aug_short, text_dict_short, _ = text_prompt(args.short_label_csv, args.mlm_label)
        print(f'[TEXT MIX] short prompts loaded: num_aug={num_aug_short}, mix_ratio={args.text_mix_ratio}')

    cluster_config = {
        'layers': [11], 'current_tokens': 197, 'target_tokens': 49,
        'algorithm': 'kmediods++', 'distance': 'euclidean', 'iter_limit': 10,
        'enable': (not args.disable_cluster),
    }
    
    # ViT-B/16: 224 -> 14x14 = 196 patches。换 backbone(如 B/32 用 //32，L/14 用 //14)时改这里
    num_spatial_tokens = (args.spatial_size // 16) ** 2
    oad_config = {
        'num_queries': 8, 'num_layers': 2, 'num_frames': args.num_frames,
        'grad_scale': args.grad_scale, 'use_fine_grained_contrastive': args.use_fine_grained_contrastive,
        'fc_grad_scale': (args.fc_grad_scale if args.fc_grad_scale is not None else args.grad_scale),
        'num_spatial_tokens': num_spatial_tokens,
        'pe_type': args.oad_pe_type,
        'use_text_cond': args.use_text_cond_query,
        'text_dim': 512,
        'use_motion': args.use_motion,
    }

    model = models_adapter.__dict__[args.model](
        num_classes=DATASETS[args.dataset]['NUM_CLASSES'],
        num_frames=args.num_frames, mlm_head_len=mlm_head_len,
        cluster_config=cluster_config, oad_config=oad_config,
    ).cuda().train()

    # ====================================================================
    # 【加载冻结锚定模型，保护零样本底盘】
    # ====================================================================
    frozen_clip = None
    if args.distill_weight > 0:
        print("Loading Frozen Original CLIP for feature distillation anchor...")
        frozen_clip = models_adapter.__dict__[args.model](
            num_classes=DATASETS[args.dataset]['NUM_CLASSES'],
            num_frames=args.num_frames, mlm_head_len=mlm_head_len,
            cluster_config=None, oad_config=None 
        ).cuda().eval()

        frozen_ckpt = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
        frozen_clip.load_state_dict(frozen_ckpt.state_dict(), strict=False)
        for p in frozen_clip.parameters(): p.requires_grad = False
    else:
        print("distill_weight<=0: 跳过冻结 CLIP teacher 的加载与前向(省显存/算力)")

    for k, v in model.named_parameters():
        if ('adapter' in k or 'cluster' in k or 'oad' in k or 'gate' in k
                or 'text_prefix' in k or 'spatial_t' in k or 'motion' in k):
            v.requires_grad = True; v.data = v.data.float()
        else:
            v.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_trainable} ({n_trainable/1e6:.2f} M)')

    model = model.to(device)
    model_without_ddp = model
    
    print("Initializing EMA model...")
    ema_model = deepcopy(model_without_ddp).eval()
    for p in ema_model.parameters(): p.requires_grad = False
    ema_decay = 0.9995

    print('creating dataset')
    _train_list = args.train_list if args.train_list else DATASETS[args.dataset]['TRAIN_LIST']
    _val_list   = args.val_list   if args.val_list   else DATASETS[args.dataset]['VAL_LIST']
    if not args.eval_only:
        dataset_train = VideoDataset(
            list_path=_train_list, data_root=args.data_root_train,
            random_sample=True, mirror=args.mirror, spatial_size=args.spatial_size, 
            auto_augment=args.auto_augment, num_frames=args.num_frames, 
            sampling_rate=args.sampling_rate, resize_type=args.resize_type, scale_range=args.scale_range,
        )
    dataset_val = VideoDataset(
        list_path=_val_list, data_root=args.data_root_val,
        random_sample=False, spatial_size=args.spatial_size, num_frames=args.num_frames, 
        sampling_rate=args.sampling_rate, num_spatial_views=args.num_spatial_views,
        num_temporal_views=args.num_temporal_views,
    )

    if not args.eval_only:
        dataloader_train = DataLoaderX(
            dataset_train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=False,
        )
    dataloader_val = DataLoaderX(
        dataset_val, batch_size=2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if args.save_dir is not None and not args.eval_only:
        os.makedirs(args.save_dir, exist_ok=True)
        shutil.copy('models_adapter.py', args.save_dir)
        shutil.copy('utils.py', args.save_dir)
        shutil.copy('main.py', args.save_dir)

    if args.eval_only:
        optimizer, loss_scaler, lr_sched = None, None, None
    else:
        if args.lr is None:
            effective_batch_size = args.batch_size * args.accumulation_steps
            args.lr = args.blr * effective_batch_size * 1 / 256
            print(f'effective lr: {args.lr} (base_lr={args.blr}, effective_batch={effective_batch_size})')

        # ====================================================================
        # 【核心修正】：精准分流，建立阶梯式文本学习率 (0.1x / 0.25x / 1.0x)
        # ====================================================================
        params_base_decay, params_base_no_decay = [], []
        params_new_decay, params_new_no_decay = [], []
        params_head_decay, params_head_no_decay = [], []
        params_text_prefix = []
        params_text_adapter = [] 

        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            no_decay = ('.bias' in n) or ('LayerNorm' in n) or ('ln_' in n)
            
            is_text_prefix = 'text_prefix' in n
            is_head = any(t in n for t in ['adapter_fc_spatial', 'adapter_fc_temporal'])
            is_new = any(t in n for t in ['cluster', 'oad', 'gate', 'spatial_t', 'motion']) \
                     and not is_head and not is_text_prefix
            
            # 【精准修复】：严格限定必须是以 transformer.resblocks 开头，排除了 visual.transformer，防止误伤视觉分支！
            is_pure_text_adapter = False
            if n.startswith('transformer.resblocks.') and 'adapter' in n:
                parts = n.split('.')
                try:
                    layer_idx = int(parts[2]) # 直接取 transformer.resblocks 后面的层号
                    if layer_idx < 12:  
                        is_pure_text_adapter = True
                except ValueError:
                    pass

            if is_text_prefix:
                params_text_prefix.append(p)
            elif is_pure_text_adapter:
                params_text_adapter.append(p) # 只放入 0-11 层的纯文本参数
            elif is_head:
                (params_head_no_decay if no_decay else params_head_decay).append(p)
            elif is_new:
                (params_new_no_decay if no_decay else params_new_decay).append(p)
            else:
                # 包含 12 层 CMLM 融合层，全速 1.0x lr
                (params_base_no_decay if no_decay else params_base_decay).append(p)

        optimizer = torch.optim.AdamW([
            # 基础模块（包含视觉、MLM头以及 CMLM 第12层的Adapter）使用完整的 1倍 LR
            {'params': params_base_decay, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': params_base_no_decay, 'lr': args.lr, 'weight_decay': 0.0},
            
            {'params': params_new_decay, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': params_new_no_decay, 'lr': args.lr, 'weight_decay': 0.0},
            {'params': params_head_decay, 'lr': args.lr * args.head_lr_mult, 'weight_decay': args.weight_decay},
            {'params': params_head_no_decay, 'lr': args.lr * args.head_lr_mult, 'weight_decay': 0.0},
            
            # 阶梯学习率：底盘 Prompt 0.1x，纯文本 Adapter(0-11层) 0.25x
            {'params': params_text_prefix, 'lr': args.lr * 0.1, 'weight_decay': 0.0},
            {'params': params_text_adapter, 'lr': args.lr * 0.25, 'weight_decay': args.weight_decay},
        ])

        loss_scaler = torch.amp.GradScaler('cuda')

        def lr_func(step):
            steps_per_epoch = len(dataloader_train) / args.accumulation_steps
            epoch = step / steps_per_epoch
            if epoch < args.warmup_epochs: return (epoch + 1e-5) / args.warmup_epochs
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * math.pi
            return 0.5 * (1 + math.cos(progress))
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    def evaluate(eval_model, log_stats=None, zero_shot_mode=False):
        if zero_shot_mode:
            print('[ZERO-SHOT CHECK] disabling all gates (prefix handled in forward)')
            prev_gate_cluster = eval_model.visual.gate_cluster.data.clone()
            prev_gate_oad = eval_model.visual.gate_oad.data.clone()
            eval_model.visual.gate_cluster.data.zero_()
            eval_model.visual.gate_oad.data.zero_()
            
            if getattr(eval_model.visual, 'use_fine_grained_contrastive', False):
                prev_gate_cc = eval_model.visual.gate_cluster_contrast.data.clone()
                prev_gate_oc = eval_model.visual.gate_oad_contrast.data.clone()
                eval_model.visual.gate_cluster_contrast.data.zero_()
                eval_model.visual.gate_oad_contrast.data.zero_()

        metric_logger = MetricLogger(delimiter="    ")
        header = 'Test:'
        eval_model.eval()
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                feats = []
                for aug_idx in range(num_aug_long):
                    text_inputs = text_dict_long[aug_idx].cuda()
                    feat, _ = eval_model.encode_text(text_inputs, force_no_prefix=True) # 绝对纯净模式
                    feats.append(feat / feat.norm(dim=-1, keepdim=True))
                text_features = torch.stack(feats).mean(dim=0)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # 文本条件化 query：把全类别文本特征作为 text_bank 下发（训练/评测同分布）
                if args.use_text_cond_query:
                    eval_model.visual.text_bank = text_features.float().detach()

        for data, labels in metric_logger.log_every(dataloader_val, 10, header):
            data, labels = data.cuda(), labels.cuda()
            B, V = data.size(0), data.size(1)
            data = data.flatten(0, 1)
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    out = eval_model.encode_image(data)
                    image_features, _, logits = out[0], out[1], out[2]
                scores = logits.softmax(dim=-1).view(B, V, -1).mean(dim=1)
                acc1_fc = (scores.topk(1, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                acc5_fc = (scores.topk(5, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).view(B, V, -1).mean(dim=1)
                acc1 = (similarity.topk(1, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                acc5 = (similarity.topk(5, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
            
            if hasattr(eval_model.visual, 'last_gate_oad'): metric_logger.meters['gate_oad'].update(eval_model.visual.last_gate_oad, n=B)
            if hasattr(eval_model.visual, 'last_gate_cluster'): metric_logger.meters['gate_cluster'].update(eval_model.visual.last_gate_cluster, n=B)
            
            metric_logger.meters['acc1'].update(acc1, n=similarity.size(0))
            metric_logger.meters['acc5'].update(acc5, n=similarity.size(0))
            metric_logger.meters['acc1_fc'].update(acc1_fc, n=similarity.size(0))
            metric_logger.meters['acc5_fc'].update(acc5_fc, n=similarity.size(0))
        
        gate_info = ""
        if 'gate_oad' in metric_logger.meters: gate_info += f" || GateOAD {metric_logger.meters['gate_oad'].global_avg:.3f}"
        if 'gate_cluster' in metric_logger.meters: gate_info += f" GateCluster {metric_logger.meters['gate_cluster'].global_avg:.3f}"

        print('* Acc@1 {t1.global_avg:.3f} Acc@5 {t5.global_avg:.3f} || AccFC@1 {t1fc.global_avg:.3f} AccFC@5 {t5fc.global_avg:.3f}{gate}'
              .format(t1=metric_logger.acc1, t5=metric_logger.acc5, t1fc=metric_logger.acc1_fc, t5fc=metric_logger.acc5_fc, gate=gate_info))
        if log_stats is not None: log_stats.update({'val_' + k: m.global_avg for k, m in metric_logger.meters.items()})

        if zero_shot_mode:
            eval_model.visual.gate_cluster.data.copy_(prev_gate_cluster)
            eval_model.visual.gate_oad.data.copy_(prev_gate_oad)
            if getattr(eval_model.visual, 'use_fine_grained_contrastive', False):
                eval_model.visual.gate_cluster_contrast.data.copy_(prev_gate_cc)
                eval_model.visual.gate_oad_contrast.data.copy_(prev_gate_oc)

    # ====================================================================
    # 【最强防线：WiSE-FT 同分布稳定性评估】
    # 注意：真实跨数据集 Zero-Shot 请单独写脚本加载 UCF101/HMDB51 跑评测
    # ====================================================================
    def evaluate_wise_ft(eval_model, target_cls_list, frozen_mdl):
        print("\n===== Running ZERO-SHOT Evaluation (WiSE-FT Interpolation & Ensemble) =====")
        try:
            import clip
        except ImportError:
            print("Warning: 'clip' module not found. Skipping Advanced Zero-Shot check.")
            return

        alpha = 0.5 
        original_sd = {k: v.clone() for k, v in eval_model.state_dict().items()}
        
        eval_sd = eval_model.state_dict()
        frozen_sd = frozen_mdl.state_dict()
        for k in eval_sd.keys():
            if k in frozen_sd: eval_sd[k].copy_(alpha * eval_sd[k] + (1 - alpha) * frozen_sd[k])
            else: eval_sd[k].copy_(alpha * eval_sd[k])
                
        eval_model.eval()
        templates = ["a video of a person {}.", "a video of a person doing {}.", "a video showing {}.", "a tutorial on how to {}.", "a person is performing {}.", "the action of {} in a video."]
        
        print("Extracting multi-template ensembled text features...")
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                all_feats = []
                for t in templates:
                    texts = [t.format(c) for c in target_cls_list]
                    txt_tokens = clip.tokenize(texts, truncate=True).cuda()
                    feat, _ = eval_model.encode_text(txt_tokens, force_no_prefix=True)
                    all_feats.append(feat / feat.norm(dim=-1, keepdim=True))
                text_features = torch.stack(all_feats).mean(dim=0)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                if args.use_text_cond_query:
                    eval_model.visual.text_bank = text_features.float().detach()

        metric_logger = MetricLogger(delimiter="    ")
        header = 'Zero-Shot (WiSE-FT):'
        
        for data, labels in metric_logger.log_every(dataloader_val, 10, header):
            data, labels = data.cuda(), labels.cuda()
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    out = eval_model.encode_image(data.flatten(0, 1))
                    image_features = out[0] / out[0].norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).view(data.size(0), data.size(1), -1).mean(dim=1)
                    acc1 = (similarity.topk(1, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
            metric_logger.meters['zs_acc1'].update(acc1, n=similarity.size(0))
            
        print(f'* Zero-Shot (WiSE-FT) Acc@1 {metric_logger.zs_acc1.global_avg:.3f}')
        eval_model.load_state_dict(original_sd)

    @torch.no_grad()
    def refresh_text_bank(mdl):
        # 用当前权重重算"全类别文本库"(无 prefix)，每个 epoch 刷新一次，detach 后作为 OAD 条件化先验
        if not args.use_text_cond_query:
            return
        was_training = mdl.training
        mdl.eval()
        feats = []
        with torch.amp.autocast('cuda'):
            for aug_idx in range(num_aug_long):
                ti = text_dict_long[aug_idx].cuda()
                f, _ = mdl.encode_text(ti, force_no_prefix=True)
                feats.append(f / f.norm(dim=-1, keepdim=True))
        bank = torch.stack(feats).mean(dim=0)
        bank = (bank / bank.norm(dim=-1, keepdim=True)).float().detach()
        mdl.visual.text_bank = bank
        if was_training:
            mdl.train()

    start_epoch = load_model(args, model_without_ddp, optimizer, lr_sched, loss_scaler)
    
    if loss_scaler is not None:
        if hasattr(loss_scaler, '_scale') and loss_scaler._scale is not None: loss_scaler._scale = loss_scaler._scale.cuda()
        if hasattr(loss_scaler, '_grow_tracker') and loss_scaler._grow_tracker is not None: loss_scaler._grow_tracker = loss_scaler._grow_tracker.cuda()
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor): state[k] = v.cuda()

    print("Syncing EMA model weights from the loaded model...")
    ema_model.load_state_dict(model_without_ddp.state_dict())

    if args.eval_only:
        evaluate(ema_model)
        if frozen_clip is not None:
            evaluate_wise_ft(ema_model, target_classes, frozen_clip)
        return

    loss_img = KLLoss(); loss_txt = KLLoss()

    max_accuracy = 0.0
    if args.save_dir is not None:
        log_path = os.path.join(args.save_dir, 'log.txt')
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    for line in f:
                        log_dict = json.loads(line.strip())
                        acc = log_dict.get('val_acc1_fc', log_dict.get('val_acc1', 0.0))
                        if acc > max_accuracy: max_accuracy = acc
                print(f"====> [Secure Load] Recovered TRUE historic best accuracy from log.txt: {max_accuracy:.3f}%")
            except Exception as e:
                pass

    for epoch in range(start_epoch, args.epochs):
        metric_logger = MetricLogger(delimiter="    ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        model.train()
        optimizer.zero_grad()
        refresh_text_bank(model)  # 文本条件化 query 的 text_bank（关闭该功能时为 no-op）

        for step, (data, labels) in enumerate(metric_logger.log_every(dataloader_train, args.print_freq, header)):
            data, labels = data.cuda(), labels.cuda()

            # 动态长短文本混合喂入 (Mix Strategy)
            use_short = (text_dict_short is not None) and (random.random() < args.text_mix_ratio)
            if use_short:
                text_id = np.random.randint(num_aug_short, size=len(labels))
                batch_texts = torch.stack([text_dict_short[j][i, :] for i, j in zip(labels, text_id)]).long().cuda()
                text_inputs_aux = text_dict_short[random.randint(0, num_aug_short - 1)].cuda()
            else:
                text_id = np.random.randint(num_aug_long, size=len(labels))
                batch_texts = torch.stack([text_dict_long[j][i, :] for i, j in zip(labels, text_id)]).long().cuda()
                text_inputs_aux = text_dict_long[random.randint(0, num_aug_long - 1)].cuda()

            with torch.amp.autocast('cuda'):
                # 1. 【主干对齐】：使用 50/50 混合的长短文本做全局对比学习 (无 Prefix 污染)
                out = model(data, batch_texts, force_no_prefix=True)
                
                logits_per_image, logits_per_text = out[0], out[1]
                image_embedding = out[2]
                text_mlm_inputs_unmasked, image_mlm_inputs = out[3], out[4]
                logits_fc = out[5]

                acc1_fc = (logits_fc.topk(1, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                acc5_fc = (logits_fc.topk(5, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                loss_fc = F.cross_entropy(logits_fc, labels, label_smoothing=args.label_smoothing)

                ground_truth = torch.tensor(gen_label(labels), dtype=data.dtype, device=device)
                loss_nce = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) * 0.5

                text_features, _ = model.encode_text(text_inputs_aux, force_no_prefix=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                image_norm = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                
                similarity = 100.0 * image_norm @ text_features.T
                loss_sim = F.cross_entropy(similarity, labels)
                acc1 = (similarity.softmax(dim=-1).topk(1, dim=-1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
                acc5 = (similarity.softmax(dim=-1).topk(5, dim=-1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100

                # 2. 【核心防线：原生 CLIP 特征锚定蒸馏】(distill_weight<=0 时跳过 teacher 前向)
                if frozen_clip is not None:
                    with torch.no_grad():
                        frozen_feat = frozen_clip.encode_image(data)[0]
                        frozen_norm = frozen_feat / frozen_feat.norm(dim=-1, keepdim=True)
                    loss_distill = (1.0 - (image_norm * frozen_norm).sum(dim=-1)).mean()
                else:
                    loss_distill = torch.zeros((), device=image_norm.device)

                # 3. 【终极双轨修正：专属短类名的细粒度 MLM 掩码分支】
                loss_mlm = torch.tensor(0.0, device=device)
                if args.mlm_weight > 0:
                    mlm_input_tokens_ids = []
                    mlm_labels_list = []
                    
                    for lbl in labels.tolist():
                        # 修复: classes_dict 是 {name: index}，用 int 标签直接索引会 KeyError；
                        # target_classes 才是 index->name（且已把 '_' 换成空格）。
                        class_name = target_classes[lbl]
                        masked_tokens, masked_label = get_masked_sample(class_name, mlm_label_dict, masked_rate=0.2)
                        mlm_input_tokens_ids.append(masked_tokens)
                        mlm_labels_list.append(masked_label)
                        
                    mlm_input_tokens_ids = torch.tensor(mlm_input_tokens_ids, dtype=torch.long).cuda(non_blocking=True)
                    mlm_labels_list = torch.tensor(mlm_labels_list, dtype=torch.long).cuda(non_blocking=True)
                    
                    _, text_mlm_inputs_masked = model.encode_text(mlm_input_tokens_ids, force_no_prefix=True)
                    loss_mlm = model.compute_mlm(text_mlm_inputs_masked, mlm_labels_list, image_mlm_inputs)["mlm_loss"]

                # 4. 总 Loss 汇总
                total_loss = (loss_nce + 0.5 * loss_sim + loss_fc + args.distill_weight * loss_distill + args.mlm_weight * loss_mlm) / args.accumulation_steps

            loss_scaler.scale(total_loss).backward()
            
            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(dataloader_train):
                loss_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                loss_scaler.step(optimizer)
                loss_scaler.update()
                lr_sched.step()
                
                with torch.no_grad():
                    ema_decay_eff = 1 - (1 - ema_decay) * args.accumulation_steps
                    for ep, p in zip(ema_model.parameters(), model_without_ddp.parameters()):
                        if p.requires_grad: ep.data.mul_(ema_decay_eff).add_(p.data, alpha=1 - ema_decay_eff)
                optimizer.zero_grad()

            metric_logger.update(
                loss=total_loss.item() * args.accumulation_steps, loss_fc=loss_fc.item(),
                loss_nce=loss_nce.item(), loss_sim=loss_sim.item(), loss_dist=loss_distill.item(), loss_mlm=loss_mlm.item(),
                lr=optimizer.param_groups[0]['lr'], acc1=acc1, acc5=acc5, acc1_fc=acc1_fc, acc5_fc=acc5_fc,
                gate_oad=getattr(model_without_ddp.visual, 'last_gate_oad', 0.0),
                gate_cluster=getattr(model_without_ddp.visual, 'last_gate_cluster', 0.0),
                gate_motion=getattr(model_without_ddp.visual, 'last_gate_motion', 0.0),
            )

        print('Averaged stats:', metric_logger)
        log_stats = {'train_' + k: m.global_avg for k, m in metric_logger.meters.items()}
        save_model(args, epoch, model_without_ddp, optimizer, lr_sched, loss_scaler)

        if epoch >= 20: args.eval_freq = 2
        if 'k400' in args.dataset and epoch >= 9: args.eval_freq = 1
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            evaluate(ema_model, log_stats)
            
            if args.save_dir is not None:
                current_acc = log_stats.get('val_acc1_fc', log_stats.get('val_acc1', 0.0))
                if current_acc > max_accuracy:
                    max_accuracy = current_acc
                    print(f"\n⭐ New Best Accuracy: {max_accuracy:.3f}%! Saving to checkpoint_best.pth...")
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'ema': ema_model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'lr_sched': lr_sched.state_dict(), 'scaler': loss_scaler.state_dict(),
                        'epoch': epoch, 'max_accuracy': max_accuracy, 'args': args,
                    }, os.path.join(args.save_dir, 'checkpoint_best.pth'))

        if args.save_dir is not None:
            log_stats['epoch'] = epoch
            with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    print('\n===== Zero-shot capability check (final epoch) =====')
    evaluate(ema_model, zero_shot_mode=True)
    if frozen_clip is not None:
        evaluate_wise_ft(ema_model, target_classes, frozen_clip)

if __name__ == '__main__':
    main()

