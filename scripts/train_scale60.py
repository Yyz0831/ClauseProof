"""Fresh QLoRA training of two pinned models on the frozen 60% QASPER cohort.

No old adapter, checkpoint resume, evaluation labels, row dropping, or truncation.
Actual TRL token IDs and completion-only labels are verified before optimizer steps.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from train_expanse_sft import (
    require, now, canonical, sha, file_sha, decode, serializable, save,
    resolve_repo_path, verify_collated, parameter_fingerprint, audit_model,
    require_finite_metrics, inventory, QUANT, LORA, TRAIN, D_DATA_SHA, D_MANIFEST_SHA,
)
from prepare_expanse_sft import render_pair
from paperproof_eval import parse_prediction

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'scale60-v8-sft'
CUDA_ALLOC_CONF = 'expandable_segments:True'
MODELS = {
    'tiny_d60_4bit': ('CohereLabs/tiny-aya-global', 'af89d219b53ed9b13b8a4645f9c8028973510324', 'cohere2'),
    'expanse8b_60_4bit': ('CohereLabs/aya-expanse-8b', '5062468bf9bc0c6035fd64e06274333ec127d980', 'cohere'),
}
ROW_KEYS = {'id', 'paper_id', 'answerable', 'prompt', 'completion', 'input_tokens',
            'completion_tokens', 'total_tokens', 'reference_index', 'human_review_status'}
CODE_FILES = ['scripts/train_scale60.py', 'scripts/train_expanse_sft.py',
              'scripts/prepare_expanse_sft.py', 'scripts/prepare_scale60.py',
              'scripts/run_expanse_trial.py', 'scripts/run_baseline.py',
              'scripts/paperproof_eval.py']
SMOKE = {'max_steps': 2, 'examples': 4,
         'selection': 'Longest four full sequences; ties preserve frozen shared order',
         'fresh_base_and_adapter_for_formal_training': True}


def build_config(arm, examples, max_sequence_tokens=8192):
    require(arm in MODELS, 'Unknown pinned model arm')
    require(type(examples) is int and examples >= 4, 'Require at least four distinct examples')
    require(type(max_sequence_tokens) is int and max_sequence_tokens in (4096, 8192), 'Unsupported sequence budget')
    model, revision, _ = MODELS[arm]
    training = deepcopy(TRAIN)
    training.update(max_sequence_tokens=max_sequence_tokens, max_steps=math.ceil(examples / 2))
    return {'schema_version': 1, 'experiment': EXPERIMENT, 'arm': arm, 'examples': examples,
            'base_model': {'model_id': model, 'revision': revision, 'dtype': 'bfloat16',
                'attention_implementation': 'sdpa', 'max_total_tokens': 8192,
                'max_new_tokens': 256, 'do_sample': False, 'seed': 42},
            'quantization': deepcopy(QUANT), 'lora': deepcopy(LORA), 'training': training,
            'smoke': deepcopy(SMOKE)}


def validate_config(config):
    expected = build_config(config['arm'], config['examples'], config['training']['max_sequence_tokens'])
    require(canonical(config) == canonical(expected), 'Configuration differs from frozen one-epoch QLoRA schedule')
    return config


def read_rows(data, examples, max_length):
    rows = [decode(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    require(len(rows) == examples and len({r['id'] for r in rows}) == examples, 'Missing or duplicated training rows')
    for row in rows:
        require(set(row) == ROW_KEYS, 'Unexpected rendered training fields')
        require(all(isinstance(row[k], str) and row[k] for k in ['id', 'paper_id', 'prompt', 'completion']), 'Invalid training identity or text')
        require(type(row['answerable']) is bool and type(row['reference_index']) is int
                and row['reference_index'] >= 0 and row['human_review_status'] == 'pending', 'Invalid reference metadata')
        require(all(type(row[k]) is int and row[k] > 0 for k in ['input_tokens', 'completion_tokens', 'total_tokens']), 'Invalid token counts')
        require(row['input_tokens'] + row['completion_tokens'] == row['total_tokens'] <= max_length,
                'Complete sequence over budget or count mismatch; never drop or truncate')
    return rows


def validate_protocol(protocol_path, config_bytes, cfg, data, rows, root=None):
    root = ROOT if root is None else Path(root)
    raw = Path(protocol_path).read_bytes(); protocol = decode(raw)
    require(protocol['schema_version'] == 1 and protocol['experiment'] == EXPERIMENT, 'Unexpected shared protocol')
    require(protocol['selected_ids'] == [r['id'] for r in rows], 'Frozen shared membership/order changed')
    counts = {'examples': len(rows), 'positive': sum(r['answerable'] for r in rows),
              'negative': sum(not r['answerable'] for r in rows)}
    require(all(protocol['counts'].get(k) == v for k, v in counts.items())
            and cfg['examples'] == len(rows), 'Shared cohort count mismatch')
    require(protocol['counts']['source_questions'] == 2593
            and protocol['source_fraction_requested'] == .60
            and len(rows) == round(2593 * .60)
            and abs(protocol['source_fraction_realized'] - len(rows)/2593) < 1e-12,
            'This experiment must contain 60 percent of the official 2593 training questions')
    require(protocol['no_truncation'] is True and protocol['no_target_relabeling'] is True,
            'Only full-source original targets are permitted')
    require(set(protocol['per_arm']) == set(MODELS), 'Both frozen training arms are required')
    paired_by_arm = {}
    for arm, spec in protocol['per_arm'].items():
        model, revision, _ = MODELS[arm]
        require(spec['model_id'] == model and spec['revision'] == revision, 'Model identity changed')
        require(spec['max_sequence_tokens'] == cfg['training']['max_sequence_tokens'], 'Arms use different length limits')
        cb = resolve_repo_path(root, spec['config_file']).read_bytes()
        other_cfg = validate_config(decode(cb))
        require(sha(cb) == spec['config_sha256'] and other_cfg['arm'] == arm
                and other_cfg['examples'] == len(rows), 'Paired config provenance mismatch')
        if arm == cfg['arm']:
            require(cb == config_bytes and spec['data_sha256'] == sha(data), 'Selected config/data binding changed')
        paired = resolve_repo_path(Path(protocol_path).parent, spec['data_file']).read_bytes()
        require(sha(paired) == spec['data_sha256'], 'Paired data bytes changed')
        paired_rows = read_rows(paired, len(rows), spec['max_sequence_tokens'])
        paired_by_arm[arm] = {r['id']: r for r in paired_rows}
        require([(r['id'], r['paper_id'], r['answerable'], r['reference_index']) for r in paired_rows]
                == [(r['id'], r['paper_id'], r['answerable'], r['reference_index']) for r in rows], 'Arms do not share identical logical rows')
    logical_spec = protocol['logical_data']
    logical_bytes = resolve_repo_path(root, logical_spec['path']).read_bytes()
    require(sha(logical_bytes) == logical_spec['sha256'], 'Shared logical source changed')
    logical = [decode(line) for line in logical_bytes.decode().splitlines() if line.strip()]
    require([r['id'] for r in logical] == protocol['selected_ids'], 'Logical data order changed')
    for original, rendered in zip(logical, rows):
        require(all(original[k] == rendered[k] for k in ['id', 'paper_id', 'answerable', 'reference_index', 'human_review_status']), 'Logical/rendered metadata differs')
        target, end = json.JSONDecoder().raw_decode(rendered['completion'])
        require(rendered['completion'][:end] == original['target_json'], 'Original target JSON changed')
        require(target['answerable'] is original['answerable'], 'Target label changed')
        parsed, error, _ = parse_prediction(original['target_json'], {p['id'] for p in original['paragraphs']})
        require(error is None and parsed == target, 'Invalid target or unknown evidence ID')
        require(isinstance(original['messages'], list) and original['messages'], 'Missing logical messages')
        for arm in MODELS:
            other = paired_by_arm[arm][original['id']]
            _, other_end = json.JSONDecoder().raw_decode(other['completion'])
            require(other['completion'][:other_end] == original['target_json'], 'Paired arm target differs')
    d = protocol['source_D']
    d_bytes = resolve_repo_path(root, d['data_file']).read_bytes()
    dm_bytes = resolve_repo_path(root, d['training_manifest_file']).read_bytes()
    require(d['data_sha256'] == sha(d_bytes) == D_DATA_SHA
            and d['training_manifest_sha256'] == sha(dm_bytes) == D_MANIFEST_SHA,
            'Original D lineage changed')
    d_rows = [decode(line) for line in d_bytes.decode().splitlines() if line.strip()]
    d_manifest = decode(dm_bytes)
    require(len(d_rows) == 128 and d['selected_count'] == 128
            and d_manifest['training_complete'] is True
            and d_manifest['selected_ids'] == [r['id'] for r in d_rows], 'Original D cohort is incomplete')
    require(all(paired_by_arm['tiny_d60_4bit'].get(r['id']) == r for r in d_rows),
            'The expanded Tiny cohort must retain every complete original D row unchanged')
    for name, expected in protocol['source_files_sha256'].items():
        require('evaluation-label' not in name and 'predictions' not in name and 'qasper-test' not in name,
                'Training verification must not read evaluation labels, outputs, or test')
        require(file_sha(resolve_repo_path(root, name)) == expected, 'Source provenance changed: ' + name)
    ev = protocol['evaluation_input']
    ev_bytes = resolve_repo_path(root, ev['path']).read_bytes()
    require(sha(ev_bytes) == ev['sha256'] and ev['paper_overlap'] == 0, 'Evaluation-input provenance changed')
    eval_papers = {decode(line)['paper_id'] for line in ev_bytes.decode().splitlines() if line.strip()}
    require(not eval_papers & {r['paper_id'] for r in rows}, 'Training/evaluation paper overlap')
    require(file_sha(root/'scripts/prepare_scale60.py') == protocol['preparation_code_sha256'], 'Data preparer code changed')
    return {'sha256': sha(raw), 'arm': cfg['arm'], 'shared_logical_data_sha256': sha(logical_bytes),
            'same_ordered_shared_ids_verified': len(rows), 'evaluation_labels_read': False}, logical


def select_rows(rows, smoke):
    return sorted(rows, key=lambda r: -r['total_tokens'])[:4] if smoke else list(rows)


def verify_tokenizer_rows(rows, tokenizer, max_length, logical):
    tokenizer.padding_side = 'right'; tokenizer.add_bos_token = False
    require(len(rows) == len(logical), 'Logical/rendered counts differ')
    encoded, accounting = {}, []
    for row, source in zip(rows, logical):
        replay, replay_ids = render_pair(tokenizer, source['messages'], source['target_json'])
        require(all(row[k] == v for k, v in replay.items()), 'Native chat-template replay changed: ' + row['id'])
        prompt_ids = tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
        ids = tokenizer(row['prompt'] + row['completion'], add_special_tokens=False)['input_ids']
        default_ids = tokenizer(row['prompt'] + row['completion'])['input_ids']
        require(default_ids == ids == replay_ids, 'Default training tokenization adds/changes special tokens: ' + row['id'])
        require(ids[:len(prompt_ids)] == prompt_ids, 'Prompt/completion boundary changed: ' + row['id'])
        require(len(prompt_ids) == row['input_tokens'] and len(ids) == row['total_tokens'] <= max_length
                and len(ids) - len(prompt_ids) == row['completion_tokens'], 'Full token counts differ')
        require(ids[-1] == tokenizer.eos_token_id, 'Completion must end at native EOS')
        encoded[row['id']] = ids
        accounting.append({'id': row['id'], 'input_tokens': len(prompt_ids), 'total_tokens': len(ids),
            'completion_tokens': len(ids) - len(prompt_ids), 'input_ids_sha256': sha(canonical(ids)),
            'default_special_tokens_unchanged': True, 'native_template_replay': True, 'truncation': False})
    return encoded, accounting


def verify_resolved_args(args, steps, max_length):
    expected = {'max_steps': steps, 'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 2,
        'num_train_epochs': 1, 'learning_rate': 1e-4, 'weight_decay': .01, 'max_grad_norm': 1.,
        'seed': 42, 'data_seed': 42, 'bf16': True, 'fp16': False, 'max_length': max_length,
        'completion_only_loss': True, 'loss_type': 'chunked_nll', 'packing': False,
        'padding_free': False, 'gradient_checkpointing': True,
        'gradient_checkpointing_kwargs': {'use_reentrant': False}, 'optim': 'adamw_torch',
        'lr_scheduler_type': 'linear', 'eval_strategy': 'no', 'save_strategy': 'no',
        'load_best_model_at_end': False, 'dataloader_drop_last': False}
    for key, value in expected.items():
        actual = getattr(args, key)
        require(getattr(actual, 'value', actual) == value, 'Resolved training argument differs: ' + key)
    require(args.get_warmup_steps(steps) == math.ceil(steps * .05), 'Resolved warmup differs')


def run(args):
    config_bytes, data = args.config.read_bytes(), args.data.read_bytes()
    cfg = validate_config(decode(config_bytes))
    max_length = cfg['training']['max_sequence_tokens']
    model_id, revision, model_type = MODELS[cfg['arm']]
    rows = read_rows(data, cfg['examples'], max_length)
    binding, logical = validate_protocol(args.protocol, config_bytes, cfg, data, rows)
    require(not args.output.exists(), 'Output directory exists; never overwrite or resume a prior run')
    selected = select_rows(rows, args.smoke); steps = 2 if args.smoke else cfg['training']['max_steps']
    args.output.mkdir(parents=True); (args.output/'code').mkdir()
    for name in CODE_FILES:
        shutil.copyfile(ROOT/name, args.output/'code'/Path(name).name)
    (args.output/'training-data.jsonl').write_bytes(data)
    (args.output/'training-config.json').write_bytes(config_bytes)
    (args.output/'training-protocol.json').write_bytes(args.protocol.read_bytes())
    manifest = {'schema_version': 1, 'status': 'preparing', 'started_at_utc': now(),
                'training_complete': False, 'smoke': args.smoke, 'preflight_only': args.preflight_only,
                'arm': cfg['arm'], 'base_model': cfg['base_model'], 'data_sha256': sha(data), 'config_source_sha256': sha(config_bytes),
                'training_protocol_sha256': binding['sha256'], 'protocol_verification': binding,
                'selected_ids': [r['id'] for r in selected], 'full_data_ids': [r['id'] for r in rows],
                'selected_examples': len(selected), 'selected_total_tokens': sum(r['total_tokens'] for r in selected),
                'selected_supervised_tokens': sum(r['completion_tokens'] for r in selected),
                'training_steps_requested': steps, 'nominal_example_presentations': len(selected),
                'shared_logical_data_sha256': binding['shared_logical_data_sha256'],
                'source_selected_question_count': len(rows),
                'quantization': QUANT, 'fresh_base_model': True, 'fresh_adapter': True,
                'resume_checkpoint': None, 'training_seed': 42, 'data_seed': 42,
                'code_sha256': {name: file_sha(ROOT/name) for name in CODE_FILES},
                'frameworks': decode((ROOT/'configs/frameworks.lock.json').read_bytes()),
                'backend': 'AlignTune TRLSFTTrainer', 'evaluation_labels_read': False,
                'cuda_allocator': {'PYTORCH_CUDA_ALLOC_CONF': CUDA_ALLOC_CONF,
                                   'PYTORCH_ALLOC_CONF': None,
                                   'scope': 'Explicit process-local allocator choice before torch import; no model/data/loss change.'},
                'independent_human_review': False, 'human_review_status': 'pending',
                'tokenizer_adjustment': 'Disable automatic BOS; native chat template is already rendered, then verify exact default token IDs.'}
    save(args.output/'manifest.json', manifest)
    stage = 'tokenizer_preflight'; sampler = None
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = CUDA_ALLOC_CONF
        os.environ.pop('PYTORCH_ALLOC_CONF', None)
        os.environ['WANDB_MODE'] = 'disabled'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_OFFLINE'] = '1'
        from transformers import AutoConfig, AutoTokenizer
        load_kwargs = {'revision': revision, 'trust_remote_code': False, 'local_files_only': True}
        tok = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        model_config = AutoConfig.from_pretrained(model_id, **load_kwargs)
        require(model_config.model_type == model_type and model_config._commit_hash == revision, 'Unexpected cached architecture or revision')
        require(model_config.max_position_embeddings >= max_length, 'Model context does not support full training inputs')
        encoded, accounting = verify_tokenizer_rows(rows, tok, max_length, logical)
        tok.save_pretrained(args.output/'tokenizer')
        model_config.save_pretrained(args.output/'model-config')
        save(args.output/'token-accounting.json', accounting)
        manifest.update(preflight_complete=True, tokenizer_files=inventory(args.output/'tokenizer'),
                        model_config_files=inventory(args.output/'model-config'),
                        token_accounting_sha256=file_sha(args.output/'token-accounting.json'),
                        maximum_full_sequence_tokens=max(r['total_tokens'] for r in rows),
                        complete_training_rows_verified=len(rows), status='prepared')
        save(args.output/'manifest.json', manifest)
        print(json.dumps({'stage': 'prepared', 'complete_rows': len(rows), 'selected_rows': len(selected),
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
                        cuda_allocator_backend=torch.cuda.memory.get_allocator_backend(),
                        chunked_lm_head_chunk_size=trl_sft._CHUNKED_LM_HEAD_CHUNK_SIZE,
                        trl_sft_source_sha256=file_sha(inspect.getfile(trl_sft)),
                        align_tune_trainer_source_sha256=file_sha(inspect.getfile(TRLSFTTrainer)))
        stage = 'load_fresh_base_and_adapter'
        sampler = DeviceMemorySampler(torch); sampler.start(); torch.cuda.reset_peak_memory_stats(0)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs, dtype=torch.bfloat16,
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
            model=ModelConfig(name_or_path=model_id, precision='bf16', max_seq_length=max_length,
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
                verify_resolved_args(hfargs, steps, max_length)
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
        require(saved['base_model_name_or_path'] == model_id and saved['peft_type'] == 'LORA'
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
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--smoke', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
