# yapf: disable
import argparse
import contextlib
import json
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from omegaconf import OmegaConf
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm

from datasets.dataloader import get_dataloader
from models.transformer import get_transformer3d
from models.transformer.utils import (
    count_trainable_parameters,
    randn_tensor,
)
from parallel.parallel import fsdp_transformer_ulysses
from parallel.utils import get_device_mesh
from trainer import (
    TrainerBase,
    linear_lr_warmpup,
    update_ema,
)
from trainer.scheduler import TimestepSamplerMP, FlowMatchScheduler
from utils import maybe_corrupt_ref_motion, smooth_motion_rep
from sampling.flow_sampler import FlowSampler, FlowSampleResult
from sampling.differentiable_flow_sampler import (
    DifferentiableSamplerConfig,
    SourceNoiseGateConfig,
    run_source_noise_subspace_probe,
    run_source_noise_reproduction_gate,
)
from sampling.relative_root_forward_guidance_v2 import (
    PROTOCOL_NAME as RELATIVE_ROOT_FORWARD_V2_PROTOCOL,
    MinimalSourceNoiseConfig,
    select_source_noise_output,
    run_minimal_source_noise_optimization,
)
from sampling.relative_root_trunk_guidance_v2_1 import (
    PROTOCOL_NAME as RELATIVE_ROOT_TRUNK_V2_1_PROTOCOL,
    RelativeRootTrunkConfig,
    run_minimal_source_noise_relative_root_trunk_optimization,
)
from sampling.noise_protocol import SampleNoiseProtocol
from sampling.m1_guidance import M1Config, M1Guidance
from sampling.relative_root_forward_guidance import (
    PROTOCOL_NAME as RELATIVE_ROOT_FORWARD_PROTOCOL,
    RelativeRootForwardConfig,
    RelativeRootForwardGuidance,
)
from sampling.relative_root_forward_guidance_v1_1 import (
    PROTOCOL_NAME as RELATIVE_ROOT_FORWARD_V1_1_PROTOCOL,
    ResidualAdaptiveRootForwardConfig,
    ResidualAdaptiveRootForwardGuidance,
)
from sampling.relative_root_forward_guidance_v1_2 import (
    PROTOCOL_NAME as RELATIVE_ROOT_FORWARD_V1_2_PROTOCOL,
    TrunkStabilizedRootForwardConfig,
    TrunkStabilizedRootForwardGuidance,
)
from sampling.relative_root_forward_guidance_v1_3 import (
    PROTOCOL_NAME as RELATIVE_ROOT_FORWARD_V1_3_PROTOCOL,
    ShadowPoseHierarchicalConfig,
    ShadowPoseHierarchicalRootForwardGuidance,
)
from sampling.absolute_mean_pelvis_guidance import (
    AbsoluteMeanPelvisConfig,
    AbsoluteMeanPelvisGuidance,
)
from sampling.absolute_mean_pelvis_guidance_v2 import (
    PROTOCOL_NAME as ABSOLUTE_MEAN_V2_PROTOCOL,
    AbsoluteMeanPelvisConfigV2,
    AbsoluteMeanPelvisGuidanceV2,
)
from sampling.absolute_mean_pelvis_guidance_v3 import (
    PROTOCOL_NAME as ABSOLUTE_MEAN_V3_PROTOCOL,
    AbsoluteMeanPelvisConfigV3,
    AbsoluteMeanPelvisGuidanceV3,
)
from sampling.absolute_mean_pelvis_guidance_v4 import (
    PROTOCOL_NAME as ABSOLUTE_MEAN_V4_PROTOCOL,
    AbsoluteMeanPelvisConfigV4,
    AbsoluteMeanPelvisGuidanceV4,
)
from motion_rep.baselines import build_b0

# yapf: enable

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def count_model_parameters(model: nn.Module):
    params = sum([np.prod(p.size()) for p in model.parameters()])
    return params


def resolve_initial_noise(
    logger,
    latents: torch.Tensor,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve legacy batch noise or replay an externally captured tensor."""
    if initial_noise is None:
        generator = torch.Generator(device).manual_seed(seed)
        return randn_tensor(
            logger,
            latents.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    if tuple(initial_noise.shape) != tuple(latents.shape):
        raise ValueError(
            "initial_noise must match latents shape, "
            f"got {tuple(initial_noise.shape)} vs {tuple(latents.shape)}"
        )
    if initial_noise.dtype != dtype:
        raise ValueError(
            f"initial_noise dtype must be {dtype}, got {initial_noise.dtype}"
        )
    return initial_noise.to(device=device)


def resolve_validation_noise(
    protocol: SampleNoiseProtocol | None,
    sample_ids,
    seed: int,
    latents: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor | None:
    """Resolve opt-in sample-level validation noise for one data-loader batch.

    The historical batch generator remains in :func:`resolve_initial_noise`.
    This adapter is only called when ``m0.noise_protocol: sample_v1`` is
    explicitly configured, and uses the loader's sample IDs before any text/
    motion condition regrouping.  That makes the resulting tensor independent
    of batch size and condition order while preserving caller order.
    """
    if protocol is None:
        return None
    return resolve_validation_noise_batch(
        protocol=protocol,
        sample_ids=sample_ids,
        seed=seed,
        latents=latents,
        dtype=dtype,
        device=device,
    ).noise


def resolve_validation_noise_batch(
    protocol: SampleNoiseProtocol,
    sample_ids,
    seed: int,
    latents: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device | str,
):
    """Return the protocol batch and audit records for one validation batch."""
    if latents.ndim < 2:
        raise ValueError(
            f"latents must have batch and sample dimensions, got {tuple(latents.shape)}"
        )
    ids = [str(sample_id) for sample_id in sample_ids]
    if len(ids) != latents.shape[0]:
        raise ValueError(
            "sample_ids length must match latents batch size, "
            f"got {len(ids)} vs {latents.shape[0]}"
        )
    batch = protocol.generate_batch(
        sample_ids=ids,
        seed=seed,
        sample_shape=tuple(latents.shape[1:]),
        dtype=dtype,
        device=device,
    )
    if tuple(batch.noise.shape) != tuple(latents.shape):
        raise RuntimeError(
            "sample-level noise shape mismatch, "
            f"got {tuple(batch.noise.shape)} vs {tuple(latents.shape)}"
        )
    return batch


def canonicalize_m0_batch(
    motion_norm: torch.Tensor,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
) -> torch.Tensor:
    """Run the explicit B0 finalization on a batch of M0 physical outputs."""

    if motion_norm.ndim != 3 or motion_norm.shape[-1] != 276:
        raise ValueError("motion_norm must have shape [B,T,276]")
    physical = motion_norm.float() * motion_std[:, None, :].float() + motion_mean[:, None, :].float()
    canonical = torch.stack([build_b0(sample).motion for sample in physical], dim=0)
    return (canonical - motion_mean[:, None, :].float()) / motion_std[:, None, :].float()


def sample_data(loader, sampler, start_epoch, start_iter):
    epoch = start_epoch
    while True:
        sampler.set_epoch(epoch)
        begin_iter = start_iter if epoch == start_epoch else 0
        epoch += 1
        for _, batch in enumerate(loader, start=begin_iter):
            yield batch


def main(args):
    is_training = args.mode == 'train'

    train_target = args.experiment.get('train_target', ['transformer'])
    train_transformer = 'transformer' in train_target
    dist.init_process_group('nccl')
    global_rank = dist.get_rank()
    is_main_process = global_rank == 0
    device = torch.device(global_rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    device_mesh_dp_hybrid = get_device_mesh(use_hybrid=True, tp_size=None)# ?
    device_mesh_dp_tp = get_device_mesh(
        use_hybrid=False, tp_size=args.parallel.tp_size)
    global_device_mesh = get_device_mesh(use_hybrid=False, tp_size=None)
    dp_mesh = device_mesh_dp_tp['dp']
    world_size = dist.get_world_size()

    dp_rank = dp_mesh.get_local_rank()
    dropout_generator = torch.Generator(device)
    dropout_generator.manual_seed(dp_rank + int(args.experiment.global_seed))
    global_dp_rank = global_device_mesh.get_local_rank()
    loglevel = args.experiment.get('loglevel', 'INFO').upper()
    trainer = TrainerBase(
        args.experiment.result_dir, log_level=loglevel, rank=global_rank, mode=args.mode)
    if global_rank == 0:
        trainer.save_config(args)
    dist.barrier()
    logger, tb_tracker, timer = trainer.logger, trainer.tb_tracker, trainer.timer
    
    result_folder = os.path.join(trainer.vis_dir, args.mbench_name)
    os.makedirs(result_folder, exist_ok=True)
    logger.info(f'result_folder: {result_folder}')
    logger.info(f'text_key: {args.dataset.text_key}')
    
    dtype_mapping = dict(
        bf16=torch.bfloat16, fp32=torch.float32, fp16=torch.float16)
    dtype = dtype_mapping[args.precision.mixed_precision]
    grad_dtype = dtype_mapping[args.precision.grad_precision]

    logger.info(
        f'dtype {dtype}, grad_dtype {grad_dtype}'
    )
    dist.barrier()

    ref_corruption_cfg = args.get('ref_motion_corruption', {})
    train_ref_corruption_cfg = None
    if ref_corruption_cfg.get('enable', False):
        train_ref_corruption_cfg = ref_corruption_cfg

    base_repo_path = args.model_path[args.experiment.model_name]

    resume_path, resume_step = trainer.get_resume_path_and_step(
        auto_resume=args.experiment.auto_resume,
        resume_path=args.experiment.resume_path)

    patch_size = 2
    in_channel = args.model.get('in_channels', 16)
    model = get_transformer3d(
        model_name=args.experiment.model_name,
        load_pretrain=args.experiment.load_pretrain,
        patch_size=patch_size,
        in_channel=in_channel,
        base_repo=base_repo_path,
        strict=False,
        model_kwargs=args.get(
            'model',
            dict(
                force_no_sincos_embed=True, rope_mode='naive',
                load_path=None)))
    model = model.to(device=device, dtype=dtype)

    if train_transformer:
        ema = deepcopy(model)
    
    logger.debug(
        f'rank {global_rank:02d} original transformer parameters: {count_model_parameters(model)}',
        main_process_only=False,
    )
    load_save_dict = {}
    model_for_opt = []
    if train_transformer:
        load_save_dict['model'] = model
        load_save_dict['ema'] = ema
        model_for_opt.append('model')

    if resume_path is not None:
        trainer.load_ckpt(
            global_dp_rank,
            load_save_dict,
            model_for_opt=None,
            optimizer=None,
            global_step=resume_step)  # load optimizer after sharding
        logger.info(
            f'resume from {resume_path}, resume_step {resume_step}',
            main_process_only=True,
        )
    if train_transformer:
        dp_strategy = 'op_grad'
        transformer_device_mesh = global_device_mesh
    else:
        dp_strategy = 'hybrid'
        transformer_device_mesh = device_mesh_dp_hybrid

    transformer_fsdp_func = partial(
        fsdp_transformer_ulysses,
        device_mesh=device_mesh_dp_tp,
        global_device_mesh=transformer_device_mesh,
        dtype=dtype,
        grad_dtype=grad_dtype,
        strategy=dp_strategy,
    )
    model = transformer_fsdp_func(model=model)
    if train_transformer:
        ema = transformer_fsdp_func(model=ema)
        ema.requires_grad_(False)
        ema.eval()
        load_save_dict['ema'] = ema
        load_save_dict['model'] = model

    logger.debug(
        f'rank {global_rank:02d} transformer parameters after sharding: {count_model_parameters(model)}',
        main_process_only=False,
    )

    def get_opt_params():
        
        trainable_modules = args.experiment.get('trainable_modules', None)

        # First, set all parameters to not require gradients
        for param in model.parameters():
            param.requires_grad = False
        
        # If specific modules are provided, enable gradients for those
        if trainable_modules:
            # Enable gradients for specified modules
            for name, module in model.named_parameters():
                if any(m in name for m in trainable_modules):
                    logger.info(f'Enabling gradients for {name}')
                    module.requires_grad = True
        else:
            # Enable gradients for all parameters
            for param in model.parameters():
                param.requires_grad = True
        
        # Filter and return only trainable parameters
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        logger.info(f'Trainable parameters: {sum([p.numel() for p in params_to_optimize])}')
        return params_to_optimize
    
    if is_training:
        opt = torch.optim.AdamW(
            get_opt_params(),
            lr=args.solver.lr,
            betas=tuple(args.solver.betas),
            weight_decay=args.solver.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=linear_lr_warmpup(args.solver.warmup_steps))
        scaler = ShardedGradScaler()

        if resume_path is not None:
            trainer.load_ckpt(
                global_dp_rank,
                load_save_dict=load_save_dict,
                model_for_opt=model_for_opt,
                optimizer=None,
                global_step=resume_step,
                load_optimizer_only=True,
            )
            logger.info(
                f'Optimizer Loaded resume from {resume_path}, resume_step {resume_step}',
                main_process_only=True,
            )
    else:
        opt, lr_scheduler = None, None    
    
    
    def to_train_mode():
        model.train(
        )  # NOTE even when train_target == ['controlnet'] the transformer should be on train mode for training

    def to_eval_mode():
        model.eval()
    
    wan_scheduler = FlowMatchScheduler()

    m0_cfg = args.get('m0', {})
    m0_noise_protocol_name = m0_cfg.get('noise_protocol', 'legacy_batch')
    if m0_noise_protocol_name not in ('legacy_batch', 'sample_v1'):
        raise ValueError(
            'm0.noise_protocol must be legacy_batch or sample_v1, '
            f'got {m0_noise_protocol_name!r}'
        )
    m0_initial_noise = None
    m0_initial_noise_path = m0_cfg.get('initial_noise_path', None)
    if m0_initial_noise_path:
        if m0_noise_protocol_name != 'legacy_batch':
            raise ValueError(
                'm0.initial_noise_path cannot be combined with '
                'm0.noise_protocol: sample_v1'
            )
        m0_initial_noise = torch.load(
            m0_initial_noise_path, map_location='cpu', weights_only=True
        ).to(device=device)
    m0_sample_noise_protocol = None
    if m0_noise_protocol_name == 'sample_v1':
        m0_sample_noise_protocol = SampleNoiseProtocol(
            cache_dir=m0_cfg.get('sample_noise_cache_dir', None)
        )
    m0_artifact_dir = m0_cfg.get('artifact_dir', None)
    if m0_artifact_dir is not None and (
        m0_initial_noise is None and m0_sample_noise_protocol is None
    ):
        raise ValueError(
            'm0.artifact_dir requires m0.initial_noise_path or '
            'm0.noise_protocol: sample_v1'
        )

    # M1 is opt-in and independent of the frozen M0 configuration.  The
    # default config has no ``m1`` section, so the historical single-pass
    # generation path remains untouched when it is absent.
    m1_cfg = args.get('m1', {})
    m1_strategy_config = M1Config.from_mapping(m1_cfg)
    m1_enabled = bool(m1_strategy_config.enabled)
    m1_trace_enabled = bool(m1_cfg.get('trace_enabled', False)) if m1_enabled else False
    m1_artifact_dir = m1_cfg.get('artifact_dir', None) if m1_enabled else None
    m1_target_delta_deg = float(m1_cfg.get('target_delta_deg', 5.0))
    if (
        m1_enabled
        and m1_artifact_dir is None
        and args.get('save_motion_visualizations', True)
    ):
        raise ValueError(
            'm1.enabled requires m1.artifact_dir when visualizations are enabled'
        )

    # Absolute-mean pelvis guidance is a separate opt-in protocol.  v1 stays
    # available for historical reproduction; v2 selects the complete FK and
    # heading-removed local-sagittal boundary explicitly.
    # It cannot be
    # combined with the historical relative-angle M1 hook in one run.
    absolute_cfg = args.get('absolute_mean_pelvis', {})
    absolute_protocol_requested = str(absolute_cfg.get('protocol', 'v1'))
    if absolute_protocol_requested in {'v1', 'vimogen_absolute_mean_pelvis_v1'}:
        absolute_strategy_config = AbsoluteMeanPelvisConfig.from_mapping(absolute_cfg)
        absolute_guidance_class = AbsoluteMeanPelvisGuidance
        absolute_protocol_name = 'vimogen_absolute_mean_pelvis_v1'
    elif absolute_protocol_requested in {
        'v2',
        ABSOLUTE_MEAN_V2_PROTOCOL,
    }:
        absolute_strategy_config = AbsoluteMeanPelvisConfigV2.from_mapping(absolute_cfg)
        absolute_guidance_class = AbsoluteMeanPelvisGuidanceV2
        absolute_protocol_name = ABSOLUTE_MEAN_V2_PROTOCOL
    elif absolute_protocol_requested in {
        'v3',
        ABSOLUTE_MEAN_V3_PROTOCOL,
    }:
        absolute_strategy_config = AbsoluteMeanPelvisConfigV3.from_mapping(absolute_cfg)
        absolute_guidance_class = AbsoluteMeanPelvisGuidanceV3
        absolute_protocol_name = ABSOLUTE_MEAN_V3_PROTOCOL
    elif absolute_protocol_requested in {
        'v4',
        ABSOLUTE_MEAN_V4_PROTOCOL,
    }:
        absolute_strategy_config = AbsoluteMeanPelvisConfigV4.from_mapping(absolute_cfg)
        absolute_guidance_class = AbsoluteMeanPelvisGuidanceV4
        absolute_protocol_name = ABSOLUTE_MEAN_V4_PROTOCOL
    else:
        raise ValueError(
            'absolute_mean_pelvis.protocol must be v1, '
            f'{ABSOLUTE_MEAN_V2_PROTOCOL}, {ABSOLUTE_MEAN_V3_PROTOCOL}, or {ABSOLUTE_MEAN_V4_PROTOCOL}'
        )
    absolute_enabled = bool(absolute_strategy_config.enabled)
    absolute_artifact_dir = (
        absolute_cfg.get('artifact_dir', None) if absolute_enabled else None
    )
    absolute_target_mean_deg = float(absolute_cfg.get('target_mean_deg', 5.0))
    if m1_enabled and absolute_enabled:
        raise ValueError('m1 and absolute_mean_pelvis are mutually exclusive')
    if absolute_enabled and absolute_target_mean_deg not in {5.0, 10.0}:
        raise ValueError('absolute_mean_pelvis target_mean_deg must be +5 or +10')
    if absolute_enabled and absolute_artifact_dir is None:
        raise ValueError('absolute_mean_pelvis.enabled requires artifact_dir')

    # Root-forward v1 is deliberately independent of the historical M1 and
    # absolute pelvis protocols.  Its M0 baseline is projected once from the
    # official FP32 endpoint, then frozen inside the guidance object.
    relative_cfg = args.get('relative_root_forward', {})
    relative_protocol_requested = str(relative_cfg.get('protocol', RELATIVE_ROOT_FORWARD_PROTOCOL))
    if relative_protocol_requested not in {
        RELATIVE_ROOT_FORWARD_PROTOCOL,
        RELATIVE_ROOT_FORWARD_V1_1_PROTOCOL,
        RELATIVE_ROOT_FORWARD_V1_2_PROTOCOL,
        RELATIVE_ROOT_FORWARD_V1_3_PROTOCOL,
        RELATIVE_ROOT_FORWARD_V2_PROTOCOL,
        RELATIVE_ROOT_TRUNK_V2_1_PROTOCOL,
        'v2',
        'v1',
    }:
        raise ValueError(
            'relative_root_forward.protocol must be '
            f'{RELATIVE_ROOT_FORWARD_PROTOCOL}, {RELATIVE_ROOT_FORWARD_V1_1_PROTOCOL}, '
            f'{RELATIVE_ROOT_FORWARD_V1_2_PROTOCOL}, {RELATIVE_ROOT_FORWARD_V1_3_PROTOCOL}, '
            f'or {RELATIVE_ROOT_FORWARD_V2_PROTOCOL} or {RELATIVE_ROOT_TRUNK_V2_1_PROTOCOL}'
        )
    source_noise_protocols = {
        RELATIVE_ROOT_FORWARD_V2_PROTOCOL,
        RELATIVE_ROOT_TRUNK_V2_1_PROTOCOL,
        'v2',
    }
    if relative_protocol_requested in source_noise_protocols:
        relative_strategy_config = None
        relative_guidance_class = None
    elif relative_protocol_requested == RELATIVE_ROOT_FORWARD_V1_3_PROTOCOL:
        relative_strategy_config = ShadowPoseHierarchicalConfig.from_mapping(relative_cfg)
        relative_guidance_class = ShadowPoseHierarchicalRootForwardGuidance
    elif relative_protocol_requested == RELATIVE_ROOT_FORWARD_V1_2_PROTOCOL:
        relative_strategy_config = TrunkStabilizedRootForwardConfig.from_mapping(relative_cfg)
        relative_guidance_class = TrunkStabilizedRootForwardGuidance
    elif relative_protocol_requested == RELATIVE_ROOT_FORWARD_V1_1_PROTOCOL:
        relative_strategy_config = ResidualAdaptiveRootForwardConfig.from_mapping(relative_cfg)
        relative_guidance_class = ResidualAdaptiveRootForwardGuidance
    else:
        relative_strategy_config = RelativeRootForwardConfig.from_mapping(relative_cfg)
        relative_guidance_class = RelativeRootForwardGuidance
    relative_enabled = (
        bool(relative_strategy_config.enabled)
        if relative_strategy_config is not None else False
    )
    relative_artifact_dir = (
        relative_cfg.get(
            'artifact_dir',
            os.path.join(
                'results', 'phase7',
                'relative_root_forward_v1_2'
                if relative_protocol_requested == RELATIVE_ROOT_FORWARD_V1_2_PROTOCOL
                else 'relative_root_forward_v1_3'
                if relative_protocol_requested == RELATIVE_ROOT_FORWARD_V1_3_PROTOCOL
                else 'relative_root_forward_v1',
            ),
        )
        if relative_enabled else None
    )
    relative_target_delta_deg = float(relative_cfg.get('target_delta_deg', 5.0))
    if relative_enabled and not -10.0 <= relative_target_delta_deg <= 10.0:
        raise ValueError('relative_root_forward target_delta_deg must lie in [-10,10]')
    if sum((m1_enabled, absolute_enabled, relative_enabled)) > 1:
        raise ValueError(
            'm1, absolute_mean_pelvis, and relative_root_forward are mutually exclusive'
        )
    source_noise_cfg = (
        relative_cfg
        if relative_protocol_requested in source_noise_protocols
        else args.get('source_noise', {})
    )
    source_noise_enabled = (
        relative_protocol_requested in source_noise_protocols
        or bool(source_noise_cfg.get('enabled', False))
    )
    source_noise_artifact_dir = (
        source_noise_cfg.get(
            'artifact_dir',
            os.path.join(
                'results', 'phase7',
                'relative_root_trunk_v2_1'
                if relative_protocol_requested == RELATIVE_ROOT_TRUNK_V2_1_PROTOCOL
                else 'relative_root_forward_v2',
            ),
        )
        if source_noise_enabled else None
    )
    source_noise_target_delta_deg = float(
        source_noise_cfg.get('target_delta_deg', relative_target_delta_deg)
    )
    if source_noise_enabled and not -10.0 <= source_noise_target_delta_deg <= 10.0:
        raise ValueError('source noise target_delta_deg must lie in [-10,10]')
    if source_noise_enabled and (m1_enabled or absolute_enabled or relative_enabled):
        raise ValueError(
            'source-noise v2 is mutually exclusive with m1, absolute_mean_pelvis, '
            'and relative_root_forward v1.x'
        )
    control_enabled = m1_enabled or absolute_enabled or relative_enabled or source_noise_enabled

    # Source-noise v2 begins with a strict, opt-in reproduction stop gate.
    # The gate leaves all historical samplers untouched and is deliberately
    # incompatible with endpoint guidance or representation reconciliation.
    source_noise_gate_cfg = args.get('source_noise_gate', {})
    source_noise_gate_enabled = bool(source_noise_gate_cfg.get('enabled', False))
    source_noise_gate_artifact_dir = (
        source_noise_gate_cfg.get('artifact_dir', None)
        if source_noise_gate_enabled else None
    )
    if source_noise_gate_enabled and control_enabled:
        raise ValueError('source_noise_gate cannot be combined with control guidance')
    if source_noise_gate_enabled and source_noise_gate_artifact_dir is None:
        raise ValueError('source_noise_gate.enabled requires artifact_dir')
    if source_noise_gate_enabled and int(args.experiment.get('validation_steps', 50)) != 50:
        raise ValueError('source_noise_gate requires experiment.validation_steps=50')
    source_noise_probe_cfg = args.get('source_noise_probe', {})
    source_noise_probe_enabled = bool(source_noise_probe_cfg.get('enabled', False))
    if source_noise_probe_enabled and int(args.experiment.get('validation_steps', 50)) != 50:
        raise ValueError('source_noise_probe requires experiment.validation_steps=50')
    source_noise_probe_artifact_dir = (
        source_noise_probe_cfg.get('artifact_dir', None)
        if source_noise_probe_enabled else None
    )
    if source_noise_probe_enabled and source_noise_probe_artifact_dir is None:
        raise ValueError('source_noise_probe.enabled requires artifact_dir')

    # Explicitly opt-in control-aware 276D reconciliation.  The default
    # configuration has no ``representation`` section, so historical M0/M1
    # outputs and their audit artifacts remain unchanged.
    representation_cfg = args.get('representation', {})
    if (
        (absolute_enabled or relative_enabled or source_noise_enabled)
        and bool(representation_cfg.get('reconciliation', {}).get('enabled', False))
    ):
        raise ValueError(
            'guided protocol owns the final reconciliation boundary; '
            'disable representation.reconciliation'
        )

    # NOTE to keep the same data within SP
    seed = (args.experiment.global_seed * world_size + dp_rank)

    logger.info(f'seed is {seed}')
    torch.manual_seed(seed)

    bucket_config_type = args.experiment.get('bucket_config_type', None)
    if bucket_config_type is not None:
        data_seed = args.experiment.global_seed
    else:
        data_seed = seed

    if is_training:
        dataloader, sampler = get_dataloader(
            local_batch=args.dataloader.local_batch,
            dp_mesh=dp_mesh,
            dataset_args=args.dataset,
            seed=data_seed,
            num_workers=args.dataloader.num_workers,
            bucket_config_type=bucket_config_type,
            dataset_name=args.experiment.dataset_name, 
            is_test=False)
    test_dataloader, test_sampler = get_dataloader(
        local_batch=args.dataloader.test_local_batch,
        dp_mesh=dp_mesh,
        dataset_args=args.dataset,
        seed=data_seed,
        num_workers=args.dataloader.num_workers,
        bucket_config_type=bucket_config_type,
        dataset_name='MBenchWiRefMotion', 
        is_test=True)

    if args.dataloader.global_batch is not None:
        accumulate_times = args.dataloader.global_batch // (
            args.dataloader.local_batch * dp_mesh.size())
    else:
        accumulate_times = 1
    torch.cuda.empty_cache()

    def generate_pipe(
        model,
        prompt_emb,
        prompt_emb_null,
        latents,
        latents_mask,
        ref_latents,
        ref_latents_mask,
        num_inference_steps: int = 50,
        cfg_scale: float = 5.0,
        use_ema: bool = False,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.bfloat16,
        scheduler: FlowMatchScheduler = None,
        seed: int = None,
        logger=None,
        condition_on_text: bool = False,
        attend_to_text_mask: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        return_artifacts: bool = False,
        batch_invariant: bool = False,
        m1_guidance=None,
        absolute_mean_guidance=None,
        relative_root_forward_guidance=None,
        trace_enabled: bool = False,
        motion_mean: torch.Tensor | None = None,
        motion_std: torch.Tensor | None = None,
        representation_config: dict | None = None,
    ):
        """Generate predictions with the frozen M0 sampler."""
        to_eval_mode()
        inf_model = ema if use_ema else model
        if seed is None:
            seed = torch.randint(0, 1000000, (1,)).item()
        noise = resolve_initial_noise(
            logger, latents, seed, device, dtype, initial_noise=initial_noise
        )
        result = FlowSampler(
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            denoising_strength=0.7,
            cfg_scale=cfg_scale,
        ).generate(
            model=inf_model,
            prompt_emb=prompt_emb,
            prompt_emb_null=prompt_emb_null,
            initial_noise=noise,
            valid_mask=latents_mask,
            ref_motion=ref_latents,
            ref_motion_mask=ref_latents_mask,
            condition_on_text=condition_on_text,
            attend_to_text_mask=attend_to_text_mask,
            dtype=dtype,
            show_progress=logger is not None,
            batch_invariant=batch_invariant,
            m1_guidance=m1_guidance,
            absolute_mean_guidance=absolute_mean_guidance,
            relative_root_forward_guidance=relative_root_forward_guidance,
            trace_enabled=trace_enabled,
            reconciliation_config=(
                None
                if representation_config is None
                else representation_config.get('reconciliation')
            ),
            motion_mean=motion_mean,
            motion_std=motion_std,
        )

        latents_pred = result.reconciled if result.reconciled is not None else result.official

        # Compute inference loss
        loss = torch.nn.functional.mse_loss(latents_pred.float(), latents.float(), reduction='none').mean(dim=-1)
        loss = loss * latents_mask
        loss = loss.sum(-1) / latents_mask.sum(-1)  # [B]
        loss = loss.mean()
        logger.info(f'Validation loss: {loss.item()}', main_process_only=True)

        to_train_mode()
        return result if return_artifacts else latents_pred

    avg_loss_dict = {'loss': 0, 'loss_text': 0, 'loss_ref_motion': 0}

    if is_training:
        dataloader = sample_data(
            dataloader,
            sampler,
            start_epoch=resume_step // len(dataloader),
            start_iter=resume_step % len(dataloader),
        )

    test_dataloader_len = len(test_dataloader)
    test_dataloader = sample_data(
        test_dataloader,
        test_sampler,
        start_epoch=resume_step // len(test_dataloader),
        start_iter=resume_step % len(test_dataloader),
    )

    if not is_training:
        eval_steps = int(args.experiment.get('eval_steps', 1000))
        if eval_steps < 1:
            raise ValueError('eval_steps must be positive')
        args.experiment.max_steps = resume_step + eval_steps
        
    pbar = tqdm(
        range(resume_step, args.experiment.max_steps),
        disable=not is_main_process,
        initial=resume_step,
    )

    to_train_mode()
    total_trainable, total_untrainable = count_trainable_parameters(
                model.named_parameters())
    logger.info(
        f'Total trainable parameters {total_trainable} \n Total untrainable parameters {total_untrainable}',
        main_process_only=True,
    )

    for global_step in pbar:
        step_plus = global_step + 1
        if is_training:
            with timer.data:
                batch = next(dataloader)
                latents = batch.pop('motion').to(device=device, dtype=dtype)  # [B, T, C]
                latents_mask = batch.pop('motion_mask').to(device=device, dtype=dtype)    # [B, T]
                prompt_emb = batch.pop('prompt_emb').to(device=device, dtype=dtype)   # [B, L, C]
                motion_mean = batch.pop('motion_mean').to(device=device)
                motion_std = batch.pop('motion_std').to(device=device)   # []       
                ref_latents = batch.pop('ref_motion').to(device=device, dtype=dtype)  # [B, T, C]
                ref_latents_mask = batch.pop('ref_motion_mask').to(device=device, dtype=dtype)  # [B, T]
                motion_dim_mask = batch.pop('motion_dim_mask').to(device=device)    # [B, C]
                attend_to_text_mask = batch.pop('attend_to_text_mask').to(device=device)  # [B]

            if train_ref_corruption_cfg is not None:
                attend_to_ref = ~attend_to_text_mask.bool()
                if attend_to_ref.any():
                    corrupted_latents, corrupted_mask = maybe_corrupt_ref_motion(
                        ref_latents, ref_latents_mask, train_ref_corruption_cfg, is_test=False)
                    corrupted_latents = corrupted_latents.to(device=device, dtype=dtype)
                    corrupted_mask = corrupted_mask.to(device=device, dtype=dtype)
                    ref_latents[attend_to_ref] = corrupted_latents[attend_to_ref]
                    ref_latents_mask[attend_to_ref] = corrupted_mask[attend_to_ref]

            wan_scheduler.set_timesteps(1000, training=True)
            noise = torch.randn_like(latents)
            timestep_ids = torch.randint(0, wan_scheduler.num_train_timesteps, (latents.shape[0],))
            timesteps = wan_scheduler.timesteps[timestep_ids].to(device=device, dtype=dtype)
            noisy_latents = wan_scheduler.add_noise(latents, noise, timesteps).to(dtype)
            training_target = wan_scheduler.training_target(latents, noise, timesteps)

            with timer.forward:
                with torch.amp.autocast(dtype=torch.bfloat16, device_type=device.type):
                    noise_pred = model(x=noisy_latents, timestep=timesteps, context=prompt_emb, x_mask=latents_mask, ref_motion=ref_latents, 
                    ref_motion_mask=ref_latents_mask, use_gradient_checkpointing=True)

                    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction='none') # [B, T, C]

                    # only compute loss for the unmasked channels and unmasked latents
                    motion_dim_mask = motion_dim_mask.unsqueeze(1).repeat(1, latents.shape[1], 1)  # [B, T, C]
                    latents_mask_expand = latents_mask.unsqueeze(-1).expand(-1, -1, latents.shape[-1])  # [B, T, C]
                    scheduler_weight = wan_scheduler.training_weight(timesteps) # [B]
                    scheduler_weight_expand = scheduler_weight.unsqueeze(-1).unsqueeze(-1).expand(-1, latents.shape[1], latents.shape[-1])  # [B, T, C] 
                    loss_mask = motion_dim_mask * latents_mask_expand   # [B, T, C]
                    channel_weights = torch.ones(loss.shape[-1], device=latents.device, dtype=latents.dtype).view(1, 1, -1)
                    channel_weights[:, :, 258:] = 3.0  # upweight the global motion channels
                    loss = loss * loss_mask * scheduler_weight_expand * channel_weights   # [B, T, C]
                    
                    # get loss_text and loss_ref_motion based on the attend_to_text_mask
                    loss_text = loss[attend_to_text_mask==1]
                    loss_ref_motion = loss[attend_to_text_mask==0]
                    loss_mask_text = loss_mask[attend_to_text_mask==1]
                    loss_mask_ref_motion = loss_mask[attend_to_text_mask==0]

                    # compute the mean loss for the non-zero values
                    loss_text_mean = loss_text.reshape(-1).sum() / loss_mask_text.reshape(-1).sum()
                    loss_ref_motion_mean = loss_ref_motion.reshape(-1).sum() / loss_mask_ref_motion.reshape(-1).sum()
                    loss_mean = loss.reshape(-1).sum() / loss_mask.reshape(-1).sum()
                    avg_loss_dict['loss'] += loss_mean.item()
                    if loss_mask_text.reshape(-1).sum() != 0:
                        avg_loss_dict['loss_text'] += loss_text_mean.item()
                    if loss_mask_ref_motion.reshape(-1).sum() != 0:
                        avg_loss_dict['loss_ref_motion'] += loss_ref_motion_mean.item()

                    loss = loss_mean
                    
            no_sync = step_plus % accumulate_times != 0 and dp_strategy == 'op_grad'
            with model.no_sync() if no_sync else contextlib.nullcontext():
                with timer.backward:
                    if dtype == torch.float16:
                        scaler.scale(loss).backward()
                    elif dtype == torch.bfloat16 or dtype == torch.float32:
                        loss.backward()
            if step_plus % accumulate_times == 0:
                if dtype == torch.float16:
                    scaler.unscale_(opt)
                model.clip_grad_norm_(args.solver.grad_clip)
                if dtype == torch.float16:
                    scaler.step(opt)
                    scaler.update()
                elif dtype == torch.bfloat16 or dtype == torch.float32:
                    opt.step()
                lr_scheduler.step()
                opt.zero_grad()

            if step_plus % args.experiment.log_every == 0:
                with timer.log:
                    loss_str = ''
                    for avg_loss_key, avg_loss in avg_loss_dict.items():
                        avg_loss = torch.tensor(
                            [avg_loss],
                            device=device) / args.experiment.log_every
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                        avg_loss = avg_loss.item() / world_size
                        loss_str += f'step: {step_plus}, {avg_loss_key}: {avg_loss} '
                        tb_tracker.add_scalar(
                            tag=f'train/{avg_loss_key}',
                            scalar_value=avg_loss,
                            global_step=step_plus)
                        avg_loss_dict[avg_loss_key] = 0
                    tb_tracker.add_scalar(
                        tag='train/lr',
                        scalar_value=opt.param_groups[0]['lr'],
                        global_step=step_plus,
                    )
                    logger.info(
                        loss_str + f'rank {global_rank:02d} '
                        f'Peak Mem: {torch.cuda.max_memory_allocated() / 1024 / 1024:,.2f} MiB ',
                        main_process_only=True,
                    )

            if step_plus % args.experiment.checkpoint_every == 0:
                with timer.ckpt:
                    trainer.save_ckpt(
                        global_dp_rank,
                        load_save_dict,
                        model_for_opt=model_for_opt,
                        optimizer=None,
                        global_step=step_plus,
                        dcp=False)
            if step_plus % args.experiment.ema_every == 0:
                with timer.ema:
                    if train_transformer:
                        update_ema(ema, model, decay=args.experiment.ema_decay)

        # Add validation after logging and EMA updates
        if (not is_training) or ((step_plus % args.experiment.visualize_every)== 0):
            for test_batch_idx in range(test_dataloader_len):
                with timer.data:
                    batch = next(test_dataloader)
                    latents = batch.pop('motion').to(device=device, dtype=dtype)  # [B, T, C]
                    latents_mask = batch.pop('motion_mask').to(device=device, dtype=dtype)    # [B, T]
                    prompt_emb = batch.pop('prompt_emb').to(device=device, dtype=dtype)   # [B, L, C]
                    prompt_emb_null = batch.pop('prompt_emb_null').to(device=device, dtype=dtype)   # [B, L, C]
                    text = batch.pop('text')
                    motion_mean = batch.pop('motion_mean').to(device=device)
                    motion_std = batch.pop('motion_std').to(device=device)
                    motion_dim_mask = batch.pop('motion_dim_mask').to(device=device)    # [B, C]
                    attend_to_text_mask = batch.pop('attend_to_text_mask').to(device=device)
                    ref_latents_original = batch.pop('ref_motion_original').to(device=device, dtype=dtype)  # [B, T, C]
                    ref_latents = batch.pop('ref_motion').to(device=device, dtype=dtype)  # [B, T, C]
                    ref_latents_mask = batch.pop('ref_motion_mask').to(device=device, dtype=dtype)  # [B, T]
                    test_sample_ids = batch.get('test_sample_id')
                    if test_sample_ids is None:
                        raise ValueError(
                            'sample-level M0 noise requires test_sample_id in the validation batch'
                        )

                torch.cuda.empty_cache()
                logger.info(
                    f'step: {step_plus}, generating validation samples',
                    main_process_only=True)

                ref_latents_visual = ref_latents.clone()
                ref_latents_visual_mask = ref_latents_mask.clone()

                m0_noise_records = None
                if m0_sample_noise_protocol is not None:
                    m0_noise_result = resolve_validation_noise_batch(
                        protocol=m0_sample_noise_protocol,
                        sample_ids=test_sample_ids,
                        seed=seed,
                        latents=latents,
                        dtype=dtype,
                        device=device,
                    )
                    m0_initial_noise_batch = m0_noise_result.noise
                    m0_noise_records = m0_noise_result.records
                elif m0_initial_noise is not None:
                    if tuple(m0_initial_noise.shape) != tuple(latents.shape):
                        raise ValueError(
                            'M0 initial noise must match the validation batch: '
                            f'{tuple(m0_initial_noise.shape)} vs {tuple(latents.shape)}'
                        )
                    m0_initial_noise_batch = m0_initial_noise
                else:
                    m0_initial_noise_batch = None

                m0_artifact_dir_current = m0_artifact_dir
                if (
                    m0_artifact_dir_current is not None
                    and m0_sample_noise_protocol is not None
                ):
                    m0_artifact_dir_current = os.path.join(
                        m0_artifact_dir_current,
                        f'batch_{test_batch_idx:03d}',
                    )

                m1_artifact_dir_current = m1_artifact_dir
                if m1_artifact_dir_current is not None and m0_sample_noise_protocol is not None:
                    m1_artifact_dir_current = os.path.join(
                        m1_artifact_dir_current,
                        f'batch_{test_batch_idx:03d}',
                    )
                absolute_artifact_dir_current = absolute_artifact_dir
                if (
                    absolute_artifact_dir_current is not None
                    and m0_sample_noise_protocol is not None
                ):
                    absolute_artifact_dir_current = os.path.join(
                        absolute_artifact_dir_current,
                        f'batch_{test_batch_idx:03d}',
                    )
                relative_artifact_dir_current = relative_artifact_dir
                if (
                    relative_artifact_dir_current is not None
                    and m0_sample_noise_protocol is not None
                ):
                    relative_artifact_dir_current = os.path.join(
                        relative_artifact_dir_current,
                        f'batch_{test_batch_idx:03d}',
                    )
                source_noise_artifact_dir_current = source_noise_artifact_dir
                if (
                    source_noise_artifact_dir_current is not None
                    and m0_sample_noise_protocol is not None
                ):
                    source_noise_artifact_dir_current = os.path.join(
                        source_noise_artifact_dir_current,
                        f'batch_{test_batch_idx:03d}',
                    )

                raw_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if m0_artifact_dir is not None
                    else None
                )
                official_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if m0_artifact_dir is not None
                    else None
                )
                m1_raw_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if m1_enabled
                    else None
                )
                m1_official_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if m1_enabled
                    else None
                )
                absolute_raw_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if absolute_enabled else None
                )
                absolute_official_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if absolute_enabled else None
                )
                absolute_g0_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if absolute_enabled else None
                )
                absolute_g1_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if absolute_enabled else None
                )
                absolute_summary_records = []
                relative_raw_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if relative_enabled else None
                )
                relative_official_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if relative_enabled else None
                )
                relative_g0_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if relative_enabled else None
                )
                relative_m0_consistent_latents_full = (
                    torch.zeros_like(latents, dtype=torch.float32)
                    if relative_enabled else None
                )
                relative_summary_records = []

                attend_to_text_mask_bool = attend_to_text_mask.bool()
                text_mask = attend_to_text_mask_bool
                motion_mask = ~attend_to_text_mask_bool
                condition_names = ['text' if flag else 'motion' for flag in attend_to_text_mask_bool.tolist()]

                gen_latents_full = torch.zeros_like(
                    latents,
                    dtype=torch.float32
                    if (absolute_enabled or relative_enabled or source_noise_enabled)
                    else latents.dtype,
                )
                for condition_name, sample_mask in (('text', text_mask), ('motion', motion_mask)):
                    if not sample_mask.any().item():
                        continue
                    condition_ref_latents = (torch.zeros_like(ref_latents_visual[sample_mask])
                                             if condition_name == 'text' else ref_latents_visual[sample_mask])
                    condition_initial_noise = (
                        None if m0_initial_noise_batch is None
                        else m0_initial_noise_batch[sample_mask]
                    )
                    m0_result = generate_pipe(
                        model=model,
                        prompt_emb=prompt_emb[sample_mask],
                        prompt_emb_null=prompt_emb_null[sample_mask],
                        latents=latents[sample_mask],
                        latents_mask=latents_mask[sample_mask],
                        ref_latents=condition_ref_latents,
                        ref_latents_mask=ref_latents_visual_mask[sample_mask],
                        num_inference_steps=args.experiment.get('validation_steps', 50),
                        cfg_scale=args.experiment.get('cfg_scale', 5.0),
                        use_ema=False,
                        device=device,
                        dtype=dtype,
                        scheduler=wan_scheduler,
                        seed=seed,
                        logger=logger,
                        condition_on_text=(condition_name == 'text'),
                        attend_to_text_mask=attend_to_text_mask_bool[sample_mask],
                        initial_noise=condition_initial_noise,
                        motion_mean=motion_mean[sample_mask],
                        motion_std=motion_std[sample_mask],
                        representation_config=representation_cfg,
                        # M1 needs the raw/official fields even when the
                        # no-video sweep deliberately does not persist the
                        # intermediate artifact files.
                        return_artifacts=(
                            m0_artifact_dir is not None
                            or control_enabled
                            or source_noise_gate_enabled
                        ),
                        batch_invariant=(
                            m0_sample_noise_protocol is not None
                            and bool(m0_cfg.get('batch_invariant', False))
                        ),
                    )
                    if (
                        source_noise_enabled
                        and isinstance(m0_result, FlowSampleResult)
                        and raw_latents_full is not None
                    ):
                        raw_latents_full[sample_mask] = m0_result.raw
                        official_latents_full[sample_mask] = m0_result.official_pre_cast
                    if source_noise_gate_enabled:
                        if not isinstance(m0_result, FlowSampleResult):
                            raise RuntimeError('source_noise_gate requires FlowSampleResult')
                        if int(sample_mask.sum().item()) != 1:
                            raise ValueError('source_noise_gate requires one sample per condition')
                        gate_record = run_source_noise_reproduction_gate(
                            model=model,
                            scheduler=wan_scheduler,
                            official_result=m0_result,
                            prompt_emb=prompt_emb[sample_mask],
                            prompt_emb_null=prompt_emb_null[sample_mask],
                            valid_mask=latents_mask[sample_mask],
                            ref_motion=condition_ref_latents,
                            ref_motion_mask=ref_latents_visual_mask[sample_mask],
                            condition_on_text=(condition_name == 'text'),
                            attend_to_text_mask=attend_to_text_mask_bool[sample_mask],
                            motion_mean=motion_mean[sample_mask],
                            motion_std=motion_std[sample_mask],
                            dtype=dtype,
                            sampler_config=DifferentiableSamplerConfig(
                                num_inference_steps=50,
                                denoising_strength=0.7,
                                cfg_scale=float(args.experiment.get('cfg_scale', 5.0)),
                                use_gradient_checkpointing=bool(
                                    source_noise_gate_cfg.get(
                                        'use_gradient_checkpointing', True
                                    )
                                ),
                            ),
                            gate_config=SourceNoiseGateConfig(
                                target_delta_deg=float(
                                    source_noise_gate_cfg.get('target_delta_deg', 10.0)
                                ),
                                max_reserved_mib=float(
                                    source_noise_gate_cfg.get(
                                        'max_reserved_mib', 28672.0
                                    )
                                ),
                            ),
                        )
                        if global_rank == 0:
                            gate_dir = os.path.join(
                                source_noise_gate_artifact_dir,
                                f'batch_{test_batch_idx:03d}',
                                condition_name,
                            )
                            os.makedirs(gate_dir, exist_ok=True)
                            gate_path = os.path.join(
                                gate_dir, 'differentiable_50step_gate.json'
                            )
                            with open(gate_path, 'w', encoding='utf-8') as gate_file:
                                json.dump(gate_record, gate_file, indent=2, sort_keys=True)
                    if source_noise_probe_enabled:
                        if not isinstance(m0_result, FlowSampleResult):
                            raise RuntimeError('source_noise_probe requires FlowSampleResult')
                        if int(sample_mask.sum().item()) != 1:
                            raise ValueError('source_noise_probe requires one sample per condition')
                        historical_path = source_noise_probe_cfg.get('historical_delta_path', None)
                        historical_delta = None
                        if historical_path is not None:
                            historical_delta = torch.load(
                                historical_path, map_location=device, weights_only=True
                            ).to(device=device, dtype=torch.float32)
                        probe_record = run_source_noise_subspace_probe(
                            model=model,
                            scheduler=wan_scheduler,
                            official_result=m0_result,
                            prompt_emb=prompt_emb[sample_mask],
                            prompt_emb_null=prompt_emb_null[sample_mask],
                            valid_mask=latents_mask[sample_mask].bool(),
                            ref_motion=condition_ref_latents,
                            ref_motion_mask=ref_latents_visual_mask[sample_mask],
                            condition_on_text=(condition_name == 'text'),
                            attend_to_text_mask=attend_to_text_mask_bool[sample_mask],
                            motion_mean=motion_mean[sample_mask],
                            motion_std=motion_std[sample_mask],
                            dtype=dtype,
                            sampler_config=DifferentiableSamplerConfig(
                                num_inference_steps=50,
                                denoising_strength=0.7,
                                cfg_scale=float(args.experiment.get('cfg_scale', 5.0)),
                                use_gradient_checkpointing=bool(
                                    source_noise_probe_cfg.get('use_gradient_checkpointing', True)
                                ),
                            ),
                            historical_delta=historical_delta,
                            direction_seed=int(source_noise_probe_cfg.get('direction_seed', 314159)),
                            rms_values=tuple(float(value) for value in source_noise_probe_cfg.get('rms_values', [0.005, 0.01])),
                            target_delta_deg=float(source_noise_probe_cfg.get('target_delta_deg', 10.0)),
                        )
                        if global_rank == 0:
                            probe_dir = os.path.join(
                                source_noise_probe_artifact_dir,
                                f'batch_{test_batch_idx:03d}',
                                condition_name,
                            )
                            os.makedirs(probe_dir, exist_ok=True)
                            response = probe_record.pop('response_matrices').numpy()
                            baseline_features = probe_record.pop('baseline_features').numpy()
                            np.savez_compressed(
                                os.path.join(probe_dir, 'subspace.npz'),
                                response_matrices=response,
                                baseline_features=baseline_features,
                            )
                            with open(os.path.join(probe_dir, 'subspace_probe.json'), 'w', encoding='utf-8') as probe_file:
                                json.dump(probe_record, probe_file, indent=2, sort_keys=True)
                    source_noise_result = None
                    if source_noise_enabled:
                        if not isinstance(m0_result, FlowSampleResult):
                            raise RuntimeError(
                                'source-noise v2 requires FlowSampleResult from M0'
                            )
                        source_config_class = (
                            RelativeRootTrunkConfig
                            if relative_protocol_requested == RELATIVE_ROOT_TRUNK_V2_1_PROTOCOL
                            else MinimalSourceNoiseConfig
                        )
                        source_config = source_config_class(
                            iterations=int(source_noise_cfg.get('iterations', 120)),
                            step_rms=float(source_noise_cfg.get('step_rms', 0.01)),
                            max_delta_rms=float(
                                source_noise_cfg.get('max_delta_rms', 1.0)
                            ),
                            line_search_steps=int(
                                source_noise_cfg.get('line_search_steps', 8)
                            ),
                            feasible_pitch_mae_deg=float(
                                source_noise_cfg.get('feasible_pitch_mae_deg', 1.0)
                            ),
                            feasible_forward_p95_deg=float(
                                source_noise_cfg.get('feasible_forward_p95_deg', 2.0)
                            ),
                            forward_loss_temperature=float(
                                source_noise_cfg.get('forward_loss_temperature', 5.0)
                            ),
                            use_gradient_checkpointing=bool(
                                source_noise_cfg.get('use_gradient_checkpointing', True)
                            ),
                            max_runtime_seconds=float(
                                source_noise_cfg.get('max_runtime_seconds', 0.0)
                            ),
                            **({
                                'feasible_relative_mae_deg': float(
                                    source_noise_cfg.get('feasible_relative_mae_deg', 1.0)
                                ),
                                'feasible_relative_p95_deg': float(
                                    source_noise_cfg.get('feasible_relative_p95_deg', 2.0)
                                ),
                            } if relative_protocol_requested == RELATIVE_ROOT_TRUNK_V2_1_PROTOCOL else {}),
                        )
                        source_noise_runner = (
                            run_minimal_source_noise_relative_root_trunk_optimization
                            if relative_protocol_requested == RELATIVE_ROOT_TRUNK_V2_1_PROTOCOL
                            else run_minimal_source_noise_optimization
                        )
                        source_noise_result = source_noise_runner(
                            model=model,
                            scheduler=wan_scheduler,
                            official_result=m0_result,
                            prompt_emb=prompt_emb[sample_mask],
                            prompt_emb_null=prompt_emb_null[sample_mask],
                            valid_mask=latents_mask[sample_mask],
                            ref_motion=condition_ref_latents,
                            ref_motion_mask=ref_latents_visual_mask[sample_mask],
                            condition_on_text=(condition_name == 'text'),
                            attend_to_text_mask=attend_to_text_mask_bool[sample_mask],
                            motion_mean=motion_mean[sample_mask],
                            motion_std=motion_std[sample_mask],
                            dtype=dtype,
                            target_delta_deg=source_noise_target_delta_deg,
                            config=source_config,
                            sampler_config=DifferentiableSamplerConfig(
                                num_inference_steps=50,
                                denoising_strength=0.7,
                                cfg_scale=float(args.experiment.get('cfg_scale', 5.0)),
                                use_gradient_checkpointing=source_config.use_gradient_checkpointing,
                            ),
                        )
                        # Keep the selected source-noise result as the active
                        # condition.  The M0 result is only the fallback when
                        # no source-noise candidate was selected.
                        condition_result = select_source_noise_output(
                            m0_result, source_noise_result
                        )
                        if global_rank == 0:
                            source_dir = os.path.join(
                                source_noise_artifact_dir_current,
                                condition_name,
                            )
                            os.makedirs(source_dir, exist_ok=True)
                            torch.save(
                                source_noise_result.optimized_norm.detach().cpu(),
                                os.path.join(source_dir, 'optimized_norm.pt'),
                            )
                            torch.save(
                                source_noise_result.source_delta.detach().cpu(),
                                os.path.join(source_dir, 'source_delta.pt'),
                            )
                            with open(
                                os.path.join(source_dir, 'guidance_summary.json'),
                                'w',
                                encoding='utf-8',
                            ) as source_file:
                                json.dump(
                                    source_noise_result.summary,
                                    source_file,
                                    indent=2,
                                    sort_keys=True,
                                )
                    # M1 uses the same initial noise and text/ref conditions:
                    # first obtain M0, then run a second pass with an endpoint
                    # strategy built from its explicit B0-canonical endpoint.
                    if not source_noise_enabled:
                        condition_result = m0_result
                    m1_result = None
                    if m1_enabled:
                        if not isinstance(m0_result, FlowSampleResult):
                            raise RuntimeError('M1 requires FlowSampleResult from the M0 pass')
                        condition_mean = motion_mean[sample_mask]
                        condition_std = motion_std[sample_mask]
                        baseline_canonical_norm = canonicalize_m0_batch(
                            m0_result.raw,
                            condition_mean,
                            condition_std,
                        ).to(device=device, dtype=dtype)
                        m1_strategy = M1Guidance(
                            baseline_motion_norm=baseline_canonical_norm,
                            valid_mask=latents_mask[sample_mask].bool(),
                            mean=condition_mean,
                            std=condition_std,
                            target_delta_deg=m1_target_delta_deg,
                            config=m1_strategy_config,
                        )
                        m1_result = generate_pipe(
                            model=model,
                            prompt_emb=prompt_emb[sample_mask],
                            prompt_emb_null=prompt_emb_null[sample_mask],
                            latents=latents[sample_mask],
                            latents_mask=latents_mask[sample_mask],
                            ref_latents=condition_ref_latents,
                            ref_latents_mask=ref_latents_visual_mask[sample_mask],
                            num_inference_steps=args.experiment.get('validation_steps', 50),
                            cfg_scale=args.experiment.get('cfg_scale', 5.0),
                            use_ema=False,
                            device=device,
                            dtype=dtype,
                            scheduler=wan_scheduler,
                            seed=seed,
                            logger=logger,
                            condition_on_text=(condition_name == 'text'),
                            attend_to_text_mask=attend_to_text_mask_bool[sample_mask],
                            initial_noise=condition_initial_noise,
                            motion_mean=condition_mean,
                            motion_std=condition_std,
                            representation_config=representation_cfg,
                            # Keep the M1 raw/official pair auditable and make
                            # the downstream assignment use FlowSampleResult.
                            return_artifacts=True,
                            batch_invariant=(
                                m0_sample_noise_protocol is not None
                                and bool(m0_cfg.get('batch_invariant', False))
                            ),
                            m1_guidance=m1_strategy,
                            trace_enabled=m1_trace_enabled,
                        )
                        condition_result = m1_result
                    absolute_result = None
                    relative_result = None
                    if absolute_enabled:
                        if not isinstance(m0_result, FlowSampleResult):
                            raise RuntimeError(
                                'absolute mean guidance requires FlowSampleResult from M0'
                            )
                        condition_mean = motion_mean[sample_mask]
                        condition_std = motion_std[sample_mask]
                        absolute_kwargs = dict(
                            baseline_motion_norm=m0_result.raw.float(),
                            valid_mask=latents_mask[sample_mask].bool(),
                            mean=condition_mean,
                            std=condition_std,
                            target_mean_deg=absolute_target_mean_deg,
                            config=absolute_strategy_config,
                        )
                        if absolute_protocol_name == ABSOLUTE_MEAN_V4_PROTOCOL:
                            absolute_kwargs['calibration'] = None
                        absolute_strategy = absolute_guidance_class(**absolute_kwargs)
                        absolute_result = generate_pipe(
                            model=model,
                            prompt_emb=prompt_emb[sample_mask],
                            prompt_emb_null=prompt_emb_null[sample_mask],
                            latents=latents[sample_mask],
                            latents_mask=latents_mask[sample_mask],
                            ref_latents=condition_ref_latents,
                            ref_latents_mask=ref_latents_visual_mask[sample_mask],
                            num_inference_steps=args.experiment.get('validation_steps', 50),
                            cfg_scale=args.experiment.get('cfg_scale', 5.0),
                            use_ema=False,
                            device=device,
                            dtype=dtype,
                            scheduler=wan_scheduler,
                            seed=seed,
                            logger=logger,
                            condition_on_text=(condition_name == 'text'),
                            attend_to_text_mask=attend_to_text_mask_bool[sample_mask],
                            initial_noise=condition_initial_noise,
                            motion_mean=condition_mean,
                            motion_std=condition_std,
                            return_artifacts=True,
                            batch_invariant=(
                                m0_sample_noise_protocol is not None
                                and bool(m0_cfg.get('batch_invariant', False))
                            ),
                            absolute_mean_guidance=absolute_strategy,
                        )
                        condition_result = absolute_result
                    if relative_enabled:
                        if not isinstance(m0_result, FlowSampleResult):
                            raise RuntimeError(
                                'relative root-forward guidance requires FlowSampleResult from the M0 pass'
                            )
                        condition_mean = motion_mean[sample_mask]
                        condition_std = motion_std[sample_mask]
                        relative_strategy = relative_guidance_class(
                            baseline_motion_norm=m0_result.official_pre_cast.float(),
                            valid_mask=latents_mask[sample_mask].bool(),
                            mean=condition_mean,
                            std=condition_std,
                            target_delta_deg=relative_target_delta_deg,
                            config=relative_strategy_config,
                        )
                        relative_result = generate_pipe(
                            model=model,
                            prompt_emb=prompt_emb[sample_mask],
                            prompt_emb_null=prompt_emb_null[sample_mask],
                            latents=latents[sample_mask],
                            latents_mask=latents_mask[sample_mask],
                            ref_latents=condition_ref_latents,
                            ref_latents_mask=ref_latents_visual_mask[sample_mask],
                            num_inference_steps=args.experiment.get('validation_steps', 50),
                            cfg_scale=args.experiment.get('cfg_scale', 5.0),
                            use_ema=False,
                            device=device,
                            dtype=dtype,
                            scheduler=wan_scheduler,
                            seed=seed,
                            logger=logger,
                            condition_on_text=(condition_name == 'text'),
                            attend_to_text_mask=attend_to_text_mask_bool[sample_mask],
                            initial_noise=condition_initial_noise,
                            motion_mean=condition_mean,
                            motion_std=condition_std,
                            return_artifacts=True,
                            batch_invariant=(
                                m0_sample_noise_protocol is not None
                                and bool(m0_cfg.get('batch_invariant', False))
                            ),
                            relative_root_forward_guidance=relative_strategy,
                            trace_enabled=relative_strategy_config.trace_enabled,
                        )
                        relative_m0_consistent_latents_full[sample_mask] = relative_strategy.baseline_motion_norm
                        condition_result = relative_result
                    if isinstance(condition_result, FlowSampleResult):
                        condition_gen_latents = (
                            condition_result.g0
                            if absolute_result is not None or relative_result is not None
                            else condition_result.reconciled
                            if condition_result.reconciled is not None
                            else condition_result.official
                        )
                        if raw_latents_full is not None and isinstance(m0_result, FlowSampleResult):
                            raw_latents_full[sample_mask] = m0_result.raw
                            official_latents_full[sample_mask] = m0_result.official_pre_cast
                        if m1_result is not None and m1_raw_latents_full is not None:
                            m1_raw_latents_full[sample_mask] = m1_result.raw
                            m1_official_latents_full[sample_mask] = m1_result.official_pre_cast
                        if absolute_result is not None:
                            absolute_raw_latents_full[sample_mask] = absolute_result.raw
                            absolute_official_latents_full[sample_mask] = absolute_result.official_pre_cast
                            absolute_g0_latents_full[sample_mask] = absolute_result.g0
                            absolute_g1_latents_full[sample_mask] = absolute_result.g1
                            absolute_summary_records.append(
                                absolute_result.guidance_summary
                            )
                        if relative_result is not None:
                            relative_raw_latents_full[sample_mask] = relative_result.raw
                            relative_official_latents_full[sample_mask] = relative_result.official_pre_cast
                            relative_g0_latents_full[sample_mask] = relative_result.g0
                            relative_summary_records.append(relative_result.guidance_summary)
                    else:
                        condition_gen_latents = condition_result
                    gen_latents_full[sample_mask] = condition_gen_latents.to(gen_latents_full.dtype)

                if m0_artifact_dir_current is not None and global_rank == 0:
                    os.makedirs(m0_artifact_dir_current, exist_ok=True)
                    torch.save(
                        m0_initial_noise_batch.detach().cpu(),
                        os.path.join(m0_artifact_dir_current, 'z0_replayed.pt'),
                    )
                    torch.save(
                        raw_latents_full.detach().cpu(),
                        os.path.join(m0_artifact_dir_current, 'm0_raw_norm_batch.pt'),
                    )
                    torch.save(
                        official_latents_full.detach().cpu(),
                        os.path.join(m0_artifact_dir_current, 'm0_official_norm_batch.pt'),
                    )
                    if m0_noise_records is not None:
                        with open(
                            os.path.join(m0_artifact_dir_current, 'sample_noise_manifest.json'),
                            'w',
                            encoding='utf-8',
                        ) as manifest_file:
                            json.dump(
                                {
                                    'protocol_version': 'vimogen-sample-noise-v1',
                                    'seed': seed,
                                    'sample_shape': list(latents.shape[1:]),
                                    'dtype': str(dtype).removeprefix('torch.'),
                                    'records': [asdict(record) for record in m0_noise_records],
                                },
                                manifest_file,
                                indent=2,
                                sort_keys=True,
                            )

                if m1_artifact_dir_current is not None and global_rank == 0:
                    os.makedirs(m1_artifact_dir_current, exist_ok=True)
                    torch.save(
                        m1_raw_latents_full.detach().cpu(),
                        os.path.join(m1_artifact_dir_current, 'm1_raw_norm_batch.pt'),
                    )
                    torch.save(
                        m1_official_latents_full.detach().cpu(),
                        os.path.join(m1_artifact_dir_current, 'm1_official_norm_batch.pt'),
                    )
                    with open(
                        os.path.join(m1_artifact_dir_current, 'm1_config.json'),
                        'w',
                        encoding='utf-8',
                    ) as m1_config_file:
                        json.dump(
                            {
                                'target_delta_deg': m1_target_delta_deg,
                                'lambda_scale': m1_strategy_config.lambda_scale,
                                'sigma_window': [m1_strategy_config.sigma_min, m1_strategy_config.sigma_max],
                                'max_correction_rms': m1_strategy_config.max_correction_rms,
                                'angle_weight': m1_strategy_config.angle_weight,
                                'hold_weight': m1_strategy_config.hold_weight,
                                'heading_mode': m1_strategy_config.heading_mode,
                                'consistency_mode': m1_strategy_config.consistency_mode,
                                'trace_enabled': m1_trace_enabled,
                            },
                            m1_config_file,
                            indent=2,
                            sort_keys=True,
                        )
                    if m1_trace_enabled and isinstance(m1_result, FlowSampleResult) and m1_result.trace is not None:
                        torch.save(
                            {key: value.detach().cpu() for key, value in m1_result.trace.items()},
                            os.path.join(m1_artifact_dir_current, 'm1_trace.pt'),
                        )

                if absolute_artifact_dir_current is not None and global_rank == 0:
                    os.makedirs(absolute_artifact_dir_current, exist_ok=True)
                    for filename, tensor in (
                        ('guided_raw_norm_batch.pt', absolute_raw_latents_full),
                        ('guided_official_norm_batch.pt', absolute_official_latents_full),
                        ('g0_norm_batch.pt', absolute_g0_latents_full),
                        ('g1_norm_batch.pt', absolute_g1_latents_full),
                    ):
                        torch.save(
                            tensor.detach().cpu(),
                            os.path.join(absolute_artifact_dir_current, filename),
                        )
                    with open(
                        os.path.join(
                            absolute_artifact_dir_current,
                            'guidance_summary.json',
                        ),
                        'w',
                        encoding='utf-8',
                    ) as guidance_file:
                        json.dump(
                            {
                                'protocol': absolute_protocol_name,
                                'target_mean_deg': absolute_target_mean_deg,
                                'records': absolute_summary_records,
                            },
                            guidance_file,
                            indent=2,
                            sort_keys=True,
                        )

                if relative_artifact_dir_current is not None and global_rank == 0:
                    os.makedirs(relative_artifact_dir_current, exist_ok=True)
                    for filename, tensor in (
                        ('guided_raw_norm_batch.pt', relative_raw_latents_full),
                        ('guided_official_norm_batch.pt', relative_official_latents_full),
                        ('g0_norm_batch.pt', relative_g0_latents_full),
                        ('m0_consistent_norm_batch.pt', relative_m0_consistent_latents_full),
                    ):
                        torch.save(
                            tensor.detach().cpu(),
                            os.path.join(relative_artifact_dir_current, filename),
                        )
                    with open(
                        os.path.join(relative_artifact_dir_current, 'guidance_summary.json'),
                        'w',
                        encoding='utf-8',
                    ) as guidance_file:
                        json.dump(
                            {
                                'protocol': relative_strategy.PROTOCOL,
                                'target_delta_deg': relative_target_delta_deg,
                                'records': relative_summary_records,
                            },
                            guidance_file,
                            indent=2,
                            sort_keys=True,
                        )
                    if (
                        relative_strategy_config.trace_enabled
                        and isinstance(relative_result, FlowSampleResult)
                        and relative_result.trace is not None
                    ):
                        torch.save(
                            {key: value.detach().cpu() for key, value in relative_result.trace.items()},
                            os.path.join(relative_artifact_dir_current, 'relative_root_forward_trace.pt'),
                        )
                
                # Visualization
                save_motion_visualizations = args.get('save_motion_visualizations', True)
                if not save_motion_visualizations:
                    # The formal MBench sweep only needs the generated
                    # normalized batch.  Persist one auditable batch archive
                    # and materialize per-prompt physical 276D files offline;
                    # this avoids 450 GPU/CPU synchronizations and thousands
                    # of small serialization operations in the generation job.
                    archive_folder = os.path.join(
                        result_folder,
                        f'batch_{test_batch_idx:03d}',
                    )
                    os.makedirs(archive_folder, exist_ok=True)
                    batch_archive = {
                        'motion_norm': gen_latents_full.detach().cpu(),
                        'motion_mask': latents_mask.bool().detach().cpu(),
                        'motion_mean': motion_mean[0:1].detach().cpu(),
                        'motion_std': motion_std[0:1].detach().cpu(),
                        'sample_ids': [str(value) for value in test_sample_ids],
                        'condition_names': condition_names,
                    }
                    torch.save(
                        batch_archive,
                        os.path.join(archive_folder, 'mbench_raw_norm_batch.pt'),
                    )
                    with open(
                        os.path.join(archive_folder, 'mbench_batch_manifest.json'),
                        'w',
                        encoding='utf-8',
                    ) as manifest_file:
                        json.dump(
                            {
                                'protocol': 'vimogen_publication_mbench_raw_batch_v1',
                                'sample_count': len(test_sample_ids),
                                'sample_ids': [str(value) for value in test_sample_ids],
                                'condition_names': condition_names,
                                'normalized_archive': 'mbench_raw_norm_batch.pt',
                            },
                            manifest_file,
                            indent=2,
                            sort_keys=True,
                        )
                    logger.info(
                        f'Saved normalized MBench batch archive for {len(test_sample_ids)} samples',
                        main_process_only=True,
                    )
                    torch.cuda.empty_cache()
                    if not is_training:
                        continue
                    continue

                motion_dict, txt_dict = {}, {}
                # for batch_idx in tqdm(range(1), desc="Saving Visualization Data", disable=logger is None):
                # vis_num = 5 if is_training else gen_latents.shape[0]
                gen_latents = gen_latents_full
                # The formal MBench sweep does not need videos or rendered
                # visualizations.  Move the complete generated batch to CPU
                # once before slicing, instead of issuing one GPU-to-CPU
                # transfer per sample in TrainerBase.save_motion_dict.
                generated_motion_cpu = None
                motion_mean_cpu = motion_mean[0:1].detach().cpu() if not save_motion_visualizations else None
                motion_std_cpu = motion_std[0:1].detach().cpu() if not save_motion_visualizations else None
                if not save_motion_visualizations:
                    # Denormalize on GPU before the single CPU transfer.  A
                    # large CPU-side broadcast over all 450 prompts is much
                    # slower and can occupy the server for several minutes.
                    generated_motion_cpu = (
                        gen_latents_full
                        * motion_std[0:1]
                        + motion_mean[0:1]
                    ).detach().cpu()
                latents_mask_cpu_full = (
                    latents_mask.bool().cpu()
                    if generated_motion_cpu is not None else None
                )
                vis_num = gen_latents.shape[0]
                if vis_num < gen_latents.shape[0]:
                    vis_idx = torch.randint(0, gen_latents.shape[0], (vis_num,))
                else:
                    vis_idx = torch.arange(gen_latents.shape[0])
                for batch_idx in tqdm(vis_idx.tolist(), desc="Saving Visualization Data", disable=logger is None):
                    test_sample_id = batch.get('test_sample_id')[batch_idx]
                    txt_dict[f'step{step_plus:08d}/{test_sample_id}/prompt.txt'] = text[batch_idx]
                    latents_mask_ = (
                        latents_mask_cpu_full[batch_idx]
                        if latents_mask_cpu_full is not None
                        else latents_mask[batch_idx].bool()
                    )  # [T]
                    ref_latents_mask_ = ref_latents_mask[batch_idx].bool()  # [T]
                    ref_latents_visual_mask_ = ref_latents_visual_mask[batch_idx].bool()
                    condition_name = condition_names[batch_idx]
                    if torch.any(latents_mask_):
                        source_latents = generated_motion_cpu if generated_motion_cpu is not None else gen_latents
                        mask_for_index = latents_mask_.cpu() if generated_motion_cpu is not None else latents_mask_
                        motion_dict[f'step{step_plus:08d}/{test_sample_id}/motion_gen_condition_on_{condition_name}.pt'] = source_latents[batch_idx:batch_idx+1, mask_for_index]
                        if condition_name == 'motion':
                            reference_motion = ref_latents_original[batch_idx:batch_idx+1, ref_latents_mask_]
                            if generated_motion_cpu is not None:
                                reference_motion = (
                                    reference_motion
                                    * motion_std[0:1]
                                    + motion_mean[0:1]
                                ).detach().cpu()
                            motion_dict[f'step{step_plus:08d}/{test_sample_id}/motion_ref.pt'] = reference_motion

                trainer.save_motion_dict(
                    motion_dict,
                    mean=motion_mean[0:1],
                    std=motion_std[0:1],
                    device=device,
                    vis=save_motion_visualizations,
                    pre_denormalized=generated_motion_cpu is not None,
                    result_folder=result_folder,
                )
                trainer.save_txt_dict(txt_dict, result_folder=result_folder)
                logger.info(f'Saved visualization data for step {step_plus}', main_process_only=True)

                torch.cuda.empty_cache()    
                
            if not is_training:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViMoGen Training and Evaluation Script')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tm2m_train',
        help='config file',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='choose is training or evaluating')
    parser.add_argument(
        '--mbench_name',
        type=str,
        default='mbench')
    args = parser.parse_args()
    main_args = OmegaConf.load(args.config)
    main_args.mode = args.mode
    main_args.mbench_name = args.mbench_name
    main(main_args)
