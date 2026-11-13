"""Train a fresh pinned Aya Expanse adapter on the exact D cohort.

The independent longest-four smoke run never supplies weights to the formal run.
The real TRL token IDs and completion labels are checked before any optimizer step.
No evaluation labels, checkpoint selection, truncation, or remote logging are used.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'CohereLabs/aya-expanse-8b'
REVISION = '5062468bf9bc0c6035fd64e06274333ec127d980'
D_DATA_SHA = 'b87644f64bdab7a179a059fc63a708fe9d84cae776a908455497163ba8bb744a'
D_MANIFEST_SHA = '7c66783becfc958c215847040d4b600414d17396c219bfbbea20011b4e669cdb'
D_ADAPTER_SHA = '576eac6622ce5bc332912998e42bed51f5c24099888c543856ecca20c009d437'
ROW_KEYS = {'id', 'paper_id', 'answerable', 'prompt', 'completion', 'input_tokens',
            'completion_tokens', 'total_tokens', 'reference_index', 'human_review_status'}
QUANT = {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
         'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': 'bfloat16'}
LORA = {'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.05, 'bias': 'none',
        'task_type': 'CAUSAL_LM', 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']}
TRAIN = {'max_sequence_tokens': 4096, 'max_steps': 64, 'epochs': 1, 'batch_size': 1,
         'gradient_accumulation_steps': 2, 'learning_rate': 1e-4, 'warmup_ratio': 0.05,
         'lr_scheduler': 'linear', 'optimizer': 'adamw_torch', 'weight_decay': 0.01,
         'max_grad_norm': 1.0, 'seed': 42, 'data_seed': 42, 'bf16': True,
         'gradient_checkpointing': True, 'gradient_checkpointing_kwargs': {'use_reentrant': False},
         'completion_only_loss': True, 'loss_type': 'chunked_nll', 'packing': False,
         'padding_free': False, 'dataset_num_proc': 1, 'evaluation': False,
         'save_intermediate_checkpoints': False, 'truncation': False, 'offload': False}
BASE = {'model_id': MODEL_ID, 'revision': REVISION, 'dtype': 'bfloat16',
        'attention_implementation': 'sdpa', 'max_total_tokens': 8192,
        'max_new_tokens': 256, 'do_sample': False, 'seed': 42}
SMOKE = {'max_steps': 2, 'examples': 4,
         'selection': 'Longest four full sequences; ties preserve original D order',
         'fresh_base_and_adapter_for_formal_training': True}
EXPECTED_CONFIG = {'schema_version': 1, 'experiment': 'expanse8b-v7-sft',
                   'base_model': BASE, 'quantization': QUANT, 'lora': LORA,
                   'training': TRAIN, 'smoke': SMOKE}
CODE_FILES = ['scripts/train_expanse_sft.py', 'scripts/prepare_expanse_sft.py',
              'scripts/run_expanse_trial.py']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'Duplicate JSON key: ' + key)
        value[key] = item
    return value


def decode(data):
    return json.loads(data, object_pairs_hook=unique_object)


def serializable(value):
    return json.loads(json.dumps(value, default=lambda x: sorted(x) if isinstance(x, set) else str(x)))


def save(path, value):
    Path(path).write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2) + '\n')


def validate_config(config):
    # Canonical JSON also rejects bool/int substitution accepted by Python equality.
    require(canonical(config) == canonical(EXPECTED_CONFIG), 'Training configuration differs from the fixed D schedule')
    return config


def read_rows(data):
    rows = [decode(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    require(len(rows) == 128 and len({r['id'] for r in rows}) == 128, 'Require the complete distinct 128-row D cohort')
    for row in rows:
        require(set(row) == ROW_KEYS, 'Unexpected training fields; use only the frozen prompt/completion rows')
        require(all(isinstance(row[k], str) and row[k] for k in ['id', 'paper_id', 'prompt', 'completion']), 'Invalid source identity or training text')
        require(type(row['answerable']) is bool and row['reference_index'] == 0
                and row['human_review_status'] == 'pending', 'Source labels/reference/human status changed')
        require(all(type(row[k]) is int and row[k] > 0 for k in ['input_tokens', 'completion_tokens', 'total_tokens']), 'Invalid token counts')
        require(row['input_tokens'] + row['completion_tokens'] == row['total_tokens'] <= 4096,
                'Full sequence is over budget or token counts disagree; never truncate or drop')
    require(sum(r['answerable'] for r in rows) == 109, 'D must retain 109 positive and 19 negative rows')
    return rows


def resolve_repo_path(root, value):
    path = Path(value)
    require(not path.is_absolute() and '..' not in path.parts, 'Provenance path must remain inside the project')
    return Path(root) / path


def validate_protocol(protocol_path, config_bytes, cfg, data, rows, root=ROOT):
    raw = Path(protocol_path).read_bytes()
    protocol = decode(raw)
    require(protocol['schema_version'] == 1 and protocol['experiment'] == 'expanse8b-v7-sft', 'Unexpected training protocol')
    require(protocol['model'] == {'model_id': MODEL_ID, 'revision': REVISION}, 'Training model identity changed')
    require(protocol['data_sha256'] == sha(data) and protocol['selected_ids'] == [r['id'] for r in rows], 'Training bytes or D row order changed')
    require(protocol['counts'] == {'examples': 128, 'positive': 109, 'negative': 19}, 'Training cohort counts changed')
    require(protocol['max_sequence_tokens'] == 4096 and protocol['truncation'] is False, 'Full-context training budget changed')
    require(protocol['resampling'] is False and protocol['target_relabeling'] is False
            and protocol['overlong_count'] == 0, 'The same complete D cohort is required')
    require(file_sha(Path(root)/'scripts/prepare_expanse_sft.py') == protocol['preparation_code_sha256'],
            'Frozen preparation code changed')
    for section in ['training', 'lora', 'quantization']:
        require(canonical(protocol[section]) == canonical(cfg[section]), 'Protocol/config mismatch: ' + section)
    require(protocol['training_config']['sha256'] == sha(config_bytes), 'Training config bytes differ from protocol')
    require(file_sha(resolve_repo_path(root, protocol['training_config']['path'])) == sha(config_bytes), 'Repository training config changed')
    source = protocol['source_D']
    require(source['data_sha256'] == D_DATA_SHA and source['training_manifest_sha256'] == D_MANIFEST_SHA
            and source['adapter_sha256'] == D_ADAPTER_SHA, 'Source D lineage differs')
    d_bytes = resolve_repo_path(root, source['data_file']).read_bytes()
    d_manifest_bytes = resolve_repo_path(root, source['training_manifest_file']).read_bytes()
    require(sha(d_bytes) == D_DATA_SHA and sha(d_manifest_bytes) == D_MANIFEST_SHA, 'Source D files changed')
    d_rows = [decode(line) for line in d_bytes.decode().splitlines() if line.strip()]
    d_manifest = decode(d_manifest_bytes)
    require(d_manifest['training_complete'] is True and d_manifest['selected_ids'] == protocol['selected_ids'], 'D training was not completed on these ordered IDs')
    require([(r['id'], r['paper_id'], r['answerable']) for r in d_rows]
            == [(r['id'], r['paper_id'], r['answerable']) for r in rows], 'D source identity/labels changed')
    for original, rebuilt in zip(d_rows, rows):
        _, old_end = json.JSONDecoder().raw_decode(original['completion'])
        _, new_end = json.JSONDecoder().raw_decode(rebuilt['completion'])
        require(original['completion'][:old_end] == rebuilt['completion'][:new_end],
                'The target JSON must remain byte-identical to D: ' + rebuilt['id'])
    require(protocol['evaluation_input']['paper_overlap'] == 0, 'Training/evaluation paper overlap')
    for name, expected in protocol['source_files_sha256'].items():
        require('evaluation-label' not in name and 'predictions' not in name, 'Training verification must not read evaluation labels or predictions')
        require(file_sha(resolve_repo_path(root, name)) == expected, 'Preparation source changed: ' + name)
    return {'sha256': sha(raw), 'source_D_data_sha256': D_DATA_SHA,
            'same_ordered_D_ids_verified': 128, 'evaluation_labels_read': False}


def select_rows(rows, smoke):
    return sorted(rows, key=lambda r: -r['total_tokens'])[:4] if smoke else list(rows)


def verify_tokenizer_rows(rows, tokenizer):
    tokenizer.padding_side = 'right'
    tokenizer.add_bos_token = False
    encoded, accounting = {}, []
    for row in rows:
        prompt_ids = tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
        ids = tokenizer(row['prompt'] + row['completion'], add_special_tokens=False)['input_ids']
        default_ids = tokenizer(row['prompt'] + row['completion'])['input_ids']
        require(default_ids == ids, 'Default training tokenization adds special tokens: ' + row['id'])
        require(ids[:len(prompt_ids)] == prompt_ids, 'Prompt/completion tokenizer boundary changed: ' + row['id'])
        require(len(prompt_ids) == row['input_tokens'] and len(ids) == row['total_tokens'] <= 4096
                and len(ids) - len(prompt_ids) == row['completion_tokens'], 'Full token counts changed: ' + row['id'])
        require(ids[-1] == tokenizer.eos_token_id, 'Completion must end in the native end-of-turn token')
        encoded[row['id']] = ids
        accounting.append({'id': row['id'], 'input_tokens': len(prompt_ids), 'total_tokens': len(ids),
                           'completion_tokens': len(ids) - len(prompt_ids), 'input_ids_sha256': sha(canonical(ids)),
                           'default_special_tokens_unchanged': True, 'truncation': False})
    return encoded, accounting


def verify_collated(actual_ids, labels, expected_ids, prompt_tokens):
    require(actual_ids == expected_ids, 'Trainer changed or truncated the complete sequence')
    require(len(labels) == len(expected_ids), 'Unexpected label padding or truncation')
    require(labels[:prompt_tokens] == [-100] * prompt_tokens, 'Prompt tokens received supervision')
    require(labels[prompt_tokens:] == expected_ids[prompt_tokens:], 'Completion tokens or EOS were masked/changed')


def verify_resolved_args(args, steps):
    expected = {'max_steps': steps, 'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 2,
                'num_train_epochs': 1, 'learning_rate': 1e-4, 'weight_decay': 0.01, 'max_grad_norm': 1.0,
                'seed': 42, 'data_seed': 42, 'bf16': True, 'fp16': False, 'max_length': 4096,
                'completion_only_loss': True, 'loss_type': 'chunked_nll', 'packing': False,
                'padding_free': False, 'gradient_checkpointing': True,
                'gradient_checkpointing_kwargs': {'use_reentrant': False}, 'optim': 'adamw_torch',
                'lr_scheduler_type': 'linear', 'eval_strategy': 'no', 'save_strategy': 'no',
                'load_best_model_at_end': False, 'dataloader_drop_last': False}
    for key, value in expected.items():
        actual = getattr(args, key)
        require(getattr(actual, 'value', actual) == value, 'Resolved training argument differs: ' + key)
    require(args.get_warmup_steps(steps) == math.ceil(steps * .05), 'Resolved warmup differs from D')


def parameter_fingerprint(model, torch):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require('lora_' in name, 'Unexpected non-adapter trainable parameter')
            value = parameter.detach().cpu().float().contiguous()
            require(bool(torch.isfinite(value).all()), 'Non-finite adapter weights')
            digest.update(canonical([name, str(parameter.dtype), list(parameter.shape)]))
            digest.update(value.numpy().tobytes()); count += parameter.numel()
    require(count > 0, 'No trainable LoRA weights')
    return digest.hexdigest(), count


def audit_model(model, bnb):
    devices = Counter(str(p.device) for p in model.parameters())
    require(set(devices) == {'cuda:0'}, 'All model parameters must be on CUDA 0; no offload')
    quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    require(quantized, 'No quantized Linear4bit modules loaded')
    for module in quantized:
        state = module.weight.quant_state
        require(state is not None and state.quant_type == 'nf4' and state.nested
                and str(module.compute_dtype) == 'torch.bfloat16', 'Effective quantization differs')
    trainables = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    require(trainables and all('lora_' in n for n, _ in trainables), 'Only new LoRA weights may train')
    return {'parameter_device_counts': dict(devices), 'linear4bit_module_count': len(quantized),
            'trainable_parameters': sum(p.numel() for _, p in trainables),
            'trainable_parameter_names': [n for n, _ in trainables],
            'trainable_dtypes': dict(Counter(str(p.dtype) for _, p in trainables)),
            'all_parameter_storage_dtypes': dict(Counter(str(p.dtype) for p in model.parameters()))}


def require_finite_metrics(logs):
    for key in ['loss', 'grad_norm', 'learning_rate', 'train_loss']:
        if key in logs:
            require(isinstance(logs[key], (int, float)) and math.isfinite(logs[key]), 'Non-finite training metric: ' + key)


def inventory(directory):
    return {str(p.relative_to(directory)): {'bytes': p.stat().st_size, 'sha256': file_sha(p)}
            for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def run(args):
    config_bytes, data = args.config.read_bytes(), args.data.read_bytes()
    cfg = validate_config(decode(config_bytes)); rows = read_rows(data)
    binding = validate_protocol(args.protocol, config_bytes, cfg, data, rows)
    require(not args.output.exists(), 'Output directory exists; never overwrite or resume a prior run')
    selected = select_rows(rows, args.smoke); steps = 2 if args.smoke else 64
    args.output.mkdir(parents=True); (args.output/'code').mkdir()
    for name in CODE_FILES:
        shutil.copyfile(ROOT/name, args.output/'code'/Path(name).name)
    (args.output/'training-data.jsonl').write_bytes(data)
    (args.output/'training-config.json').write_bytes(config_bytes)
    (args.output/'training-protocol.json').write_bytes(args.protocol.read_bytes())
    manifest = {'schema_version': 1, 'status': 'preparing', 'started_at_utc': now(),
                'training_complete': False, 'smoke': args.smoke, 'preflight_only': args.preflight_only,
                'base_model': cfg['base_model'], 'data_sha256': sha(data), 'config_source_sha256': sha(config_bytes),
                'training_protocol_sha256': binding['sha256'], 'protocol_verification': binding,
                'selected_ids': [r['id'] for r in selected], 'full_data_ids': [r['id'] for r in rows],
                'selected_examples': len(selected), 'selected_total_tokens': sum(r['total_tokens'] for r in selected),
                'selected_supervised_tokens': sum(r['completion_tokens'] for r in selected),
                'training_steps_requested': steps, 'nominal_example_presentations': steps * 2,
                'quantization': QUANT, 'fresh_base_model': True, 'fresh_adapter': True,
                'resume_checkpoint': None, 'training_seed': 42, 'data_seed': 42,
                'code_sha256': {name: file_sha(ROOT/name) for name in CODE_FILES},
                'frameworks': decode((ROOT/'configs/frameworks.lock.json').read_bytes()),
                'backend': 'AlignTune TRLSFTTrainer', 'evaluation_labels_read': False,
                'independent_human_review': False, 'human_review_status': 'pending',
                'tokenizer_adjustment': 'Disable automatic BOS; native chat template is already rendered, then verify exact default token IDs.'}
    save(args.output/'manifest.json', manifest)
    stage = 'tokenizer_preflight'; sampler = None
    try:
        os.environ['WANDB_MODE'] = 'disabled'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_OFFLINE'] = '1'
        from transformers import AutoConfig, AutoTokenizer
        load_kwargs = {'revision': REVISION, 'trust_remote_code': False, 'local_files_only': True}
        tok = AutoTokenizer.from_pretrained(MODEL_ID, **load_kwargs)
        model_config = AutoConfig.from_pretrained(MODEL_ID, **load_kwargs)
        require(model_config.model_type == 'cohere' and model_config._commit_hash == REVISION, 'Unexpected cached architecture or revision')
        require(model_config.max_position_embeddings >= 4096, 'Model context does not support full training inputs')
        encoded, accounting = verify_tokenizer_rows(rows, tok)
        tok.save_pretrained(args.output/'tokenizer')
        model_config.save_pretrained(args.output/'model-config')
        save(args.output/'token-accounting.json', accounting)
        manifest.update(preflight_complete=True, tokenizer_files=inventory(args.output/'tokenizer'),
                        model_config_files=inventory(args.output/'model-config'),
                        token_accounting_sha256=file_sha(args.output/'token-accounting.json'),
                        maximum_full_sequence_tokens=max(r['total_tokens'] for r in rows),
                        complete_training_rows_verified=128, status='prepared')
        save(args.output/'manifest.json', manifest)
        print(json.dumps({'stage': 'prepared', 'complete_rows': 128, 'selected_rows': len(selected),
                          'maximum_tokens': manifest['maximum_full_sequence_tokens'], 'steps': steps}), flush=True)
        if args.preflight_only:
            manifest['completed_at_utc'] = now(); save(args.output/'manifest.json', manifest); return
        stage = 'load_training_runtime'
        sys.path.insert(0, str(ROOT/'vendor/aligntune/src'))
        import torch
        import bitsandbytes as bnb
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback, set_seed
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from safetensors.torch import load_file
        from aligntune.backends.trl.sft.sft_generation import TRLSFTTrainer
        from aligntune.core.sft.config import SFTConfig, ModelConfig, DatasetConfig, TrainingConfig, LoggingConfig
        from run_expanse_trial import DeviceMemorySampler
        import trl.trainer.sft_trainer as trl_sft
        require(torch.cuda.is_available(), 'CUDA is required; never fall back to CPU')
        torch.cuda.set_device(0); set_seed(42)
        manifest['packages'] = {n: importlib.metadata.version(n) for n in
                               ['torch', 'transformers', 'trl', 'peft', 'bitsandbytes', 'datasets', 'accelerate', 'huggingface-hub']}
        manifest.update(gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda,
                        gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                        trl_sft_source_sha256=file_sha(inspect.getfile(trl_sft)),
                        align_tune_trainer_source_sha256=file_sha(inspect.getfile(TRLSFTTrainer)))
        stage = 'load_fresh_base_and_adapter'
        sampler = DeviceMemorySampler(torch); sampler.start(); torch.cuda.reset_peak_memory_stats(0)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs, dtype=torch.bfloat16,
                    quantization_config=quant, device_map={'': 0}, attn_implementation='sdpa')
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={'use_reentrant': False})
        model = get_peft_model(model, LoraConfig(**LORA))
        initial, count = parameter_fingerprint(model, torch)
        manifest.update(adapter_initial_trainable_sha256=initial, trainable_parameters=count,
                        loaded_model=audit_model(model, bnb), lora=serializable(model.peft_config['default'].to_dict()),
                        initial_allocated_gib=torch.cuda.memory_allocated(0)/2**30)
        config = SFTConfig(
            model=ModelConfig(name_or_path=MODEL_ID, precision='bf16', max_seq_length=4096,
                              trust_remote_code=False, use_unsloth=False),
            dataset=DatasetConfig(name=str(args.data.resolve())),
            train=TrainingConfig(max_steps=steps, epochs=1, per_device_batch_size=1,
                gradient_accumulation_steps=2, learning_rate=1e-4, warmup_ratio=.05,
                lr_scheduler='linear', optimizer='adamw_torch', weight_decay=.01, max_grad_norm=1.0,
                bf16=True, seed=42, data_seed=42, load_best_model_at_end=False,
                packing=False, padding_free=False, completion_only_loss=True,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                loss_type='chunked_nll', dataset_num_proc=1,
                extra_params={'save_strategy': 'no', 'logging_strategy': 'steps', 'eval_strategy': 'no', 'report_to': 'none'}),
            logging=LoggingConfig(output_dir=str(args.output/'adapter'), loggers=['none'],
                                  report_to='none', log_interval=1, save_strategy='no', eval_strategy='no'))
        manifest['requested_config'] = serializable(asdict(config))
        backend = TRLSFTTrainer(config); backend.model, backend.tokenizer = model, tok
        backend.dataset = Dataset.from_list([{'prompt': r['prompt'], 'completion': r['completion']} for r in selected])
        backend.eval_dataset = None
        gradient_checks = []; step_logs = []

        class VerifyAndRecord(TrainerCallback):
            def on_train_begin(self, hfargs, state, control, **kwargs):
                verify_resolved_args(hfargs, steps)
                actual = backend.trainer.train_dataset
                require(len(actual) == len(selected), 'Trainer dropped or added rows')
                for i, row in enumerate(selected):
                    batch = backend.trainer.data_collator([actual[i]])
                    verify_collated(actual[i]['input_ids'], batch['labels'][0].tolist(),
                                    encoded[row['id']], row['input_tokens'])
                manifest.update(status='training', token_and_label_mask_check='passed_all_selected_rows',
                                actual_trainer_rows_verified=len(selected), resolved_training_args=hfargs.to_dict(),
                                resolved_warmup_steps=hfargs.get_warmup_steps(steps))
                save(args.output/'manifest.json', manifest)

            def on_pre_optimizer_step(self, hfargs, state, control, **kwargs):
                grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                require(bool(grads), 'No adapter gradients before optimizer step')
                finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
                nonzero = bool(torch.stack([(g != 0).any() for g in grads]).any())
                require(finite and nonzero, 'Non-finite or all-zero adapter gradients')
                gradient_checks.append({'optimizer_step': state.global_step + 1,
                                        'finite': finite, 'any_nonzero': nonzero, 'gradient_tensors': len(grads)})

            def on_log(self, hfargs, state, control, logs=None, **kwargs):
                if logs:
                    require_finite_metrics(logs)
                    record = {'step': state.global_step, **logs,
                              'peak_allocated_gib': torch.cuda.max_memory_allocated(0)/2**30,
                              'peak_reserved_gib': torch.cuda.max_memory_reserved(0)/2**30}
                    step_logs.append(record)
                    with (args.output/'training_log.jsonl').open('a') as handle:
                        handle.write(json.dumps(serializable(record)) + '\n')
                    print(json.dumps(serializable(record)), flush=True)

        backend.get_hf_callbacks = lambda: [VerifyAndRecord()]
        save(args.output/'manifest.json', manifest)
        stage = 'training'; torch.cuda.synchronize(0); started = time.perf_counter()
        result = backend.train(); torch.cuda.synchronize(0)
        manifest.update(result=result, wall_seconds=time.perf_counter() - started,
                        actual_final_global_step=backend.trainer.state.global_step,
                        actual_final_epoch=backend.trainer.state.epoch)
        stage = 'verify_saved_adapter'
        require(result['steps'] == backend.trainer.state.global_step == steps, 'Training did not complete the fixed steps')
        require(abs(backend.trainer.state.epoch - 1.0) < 1e-6, 'Fixed cohort must complete exactly one epoch')
        require([x['optimizer_step'] for x in gradient_checks] == list(range(1, steps + 1)), 'Missing optimizer gradient checks')
        require([r['step'] for r in step_logs if 'loss' in r] == list(range(1, steps + 1)), 'Missing per-step training losses')
        require_finite_metrics(result['metrics']); require(math.isfinite(result['final_loss']), 'Non-finite final loss')
        final, final_count = parameter_fingerprint(model, torch)
        require(final != initial and final_count == count, 'Adapter weights did not change')
        tok.save_pretrained(args.output/'adapter')
        weights, adapter_config = args.output/'adapter/adapter_model.safetensors', args.output/'adapter/adapter_config.json'
        require(weights.is_file() and adapter_config.is_file(), 'Saved adapter is missing')
        tensors = load_file(str(weights), device='cpu')
        require(tensors and all(bool(torch.isfinite(v).all()) for v in tensors.values()), 'Saved adapter contains non-finite tensors')
        saved = decode(adapter_config.read_bytes())
        require(saved['base_model_name_or_path'] == MODEL_ID and saved['peft_type'] == 'LORA'
                and set(saved['target_modules']) == set(LORA['target_modules'])
                and all(saved[k] == LORA[k] for k in ['r', 'lora_alpha', 'lora_dropout', 'bias']), 'Saved adapter configuration changed')
        save(args.output/'gradient-checks.json', gradient_checks)
        device_memory = sampler.finish(); sampler = None
        manifest.update(training_complete=True, status='completed', completed_at_utc=now(),
                        adapter_final_trainable_sha256=final, finite_adapter_weights=True,
                        adapter_sha256=file_sha(weights), adapter_config_sha256=file_sha(adapter_config),
                        gradient_checks_sha256=file_sha(args.output/'gradient-checks.json'),
                        gradient_checks_passed=len(gradient_checks), training_log_sha256=file_sha(args.output/'training_log.jsonl'),
                        peak_allocated_gib=torch.cuda.max_memory_allocated(0)/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(0)/2**30, device_memory=device_memory)
        save(args.output/'manifest.json', manifest)
        print(json.dumps({'status': 'training_complete', 'smoke': args.smoke, 'steps': steps,
                          'peak_allocated_gib': manifest['peak_allocated_gib'], 'wall_seconds': manifest['wall_seconds'],
                          'adapter_sha256': manifest['adapter_sha256']}), flush=True)
    except Exception as exc:
        manifest.update(status='failed', training_complete=False, failed_stage=stage,
                        failed_at_utc=now(), error_type=type(exc).__name__, error=str(exc))
        if 'torch' in locals():
            try:
                manifest.update(peak_allocated_gib=torch.cuda.max_memory_allocated(0)/2**30,
                                peak_reserved_gib=torch.cuda.max_memory_reserved(0)/2**30)
            except Exception:
                pass
        if sampler is not None:
            try:
                manifest['device_memory'] = sampler.finish(); sampler = None
            except Exception as cleanup:
                manifest['sampler_cleanup_error'] = str(cleanup)
        save(args.output/'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/expanse8b-v7-training.json')
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--smoke', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
