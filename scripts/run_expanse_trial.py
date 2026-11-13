"""Pinned 4-bit inference comparison; no training, offload, truncation or retries.

Inputs contain source text only. Labels and semantic assessment belong to a
separate file and are never read by this runner. Preparation is CPU-only and
precedes loading model weights. Every requested row survives in the accounting
and predictions, including length skips, format failures and runtime errors.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import re
import shutil
import threading
import time

from paperproof_eval import parse_prediction
from run_baseline import ROOT, messages_for, save_json
from serve_live import live_messages

MODEL_TYPES = {
    'CohereLabs/tiny-aya-global': 'cohere2',
    'CohereLabs/aya-expanse-8b': 'cohere',
    'CohereForAI/aya-expanse-8b': 'cohere',
}
TINY_REVISION = 'af89d219b53ed9b13b8a4645f9c8028973510324'
D_ADAPTER_SHA256 = '576eac6622ce5bc332912998e42bed51f5c24099888c543856ecca20c009d437'
D_MANIFEST_SHA256 = '7c66783becfc958c215847040d4b600414d17396c219bfbbea20011b4e669cdb'
QUANTIZATION = {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': 'bfloat16'}
FIXED = {'dtype': 'bfloat16', 'attention_implementation': 'sdpa',
         'max_total_tokens': 8192, 'max_new_tokens': 256, 'do_sample': False, 'seed': 42}
ROW_KEYS = {'id', 'paper_id', 'question', 'title', 'paragraphs', 'context', 'prompt_mode'}
SOURCE_FILES = ['scripts/run_expanse_trial.py', 'scripts/run_baseline.py',
                'scripts/serve_live.py', 'scripts/paperproof_eval.py']
PROTOCOL_CODE_FILES = set(SOURCE_FILES) | {
    'scripts/prepare_expanse_trial.py', 'scripts/summarize_expanse_trial.py',
    'scripts/analyze_run.py', 'scripts/summarize_ad_review.py', 'scripts/build_blind_review.py'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def logical_message_bytes(value):
    # Identical byte convention to the frozen protocol builder's json_bytes.
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key: ' + key)
        value[key] = item
    return value


def decode(data):
    return json.loads(data, object_pairs_hook=unique_object)


def validate_config(value):
    if not isinstance(value, dict) or set(value) - (set(FIXED) | {'model_id', 'revision', 'quantization', 'label'}):
        raise ValueError('Unknown configuration keys')
    cfg = {**FIXED, 'quantization': dict(QUANTIZATION), **value}
    if cfg.get('model_id') not in MODEL_TYPES:
        raise ValueError('Only the fixed Tiny Aya Global or Aya Expanse 8B families are allowed')
    if not isinstance(cfg.get('revision'), str) or not re.fullmatch('[0-9a-f]{40}', cfg['revision']):
        raise ValueError('An immutable 40-character commit revision is required')
    for key, expected in FIXED.items():
        if type(cfg[key]) is not type(expected) or cfg[key] != expected:
            raise ValueError('Unsupported inference setting: ' + key)
    if cfg['quantization'] != QUANTIZATION or any(type(cfg['quantization'].get(k)) is not type(v) for k, v in QUANTIZATION.items()):
        raise ValueError('This experiment requires NF4 double quantization with BF16 compute')
    if 'label' in cfg and (not isinstance(cfg['label'], str) or not cfg['label'].strip()):
        raise ValueError('Configuration label must be a nonempty string')
    return cfg


def read_rows(data):
    rows = [decode(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty input')
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError('Each row must contain exactly the seven source-only fields')
        for name in ROW_KEYS - {'paragraphs'}:
            if not isinstance(row[name], str):
                raise ValueError('Source field must be a string: ' + name)
        if not row['id'] or not row['paper_id'] or not row['question'].strip():
            raise ValueError('Missing question or source identity')
        if row['prompt_mode'] not in {'baseline', 'live'}:
            raise ValueError('Unknown prompt mode')
        paragraphs = row['paragraphs']
        if not isinstance(paragraphs, list) or not paragraphs:
            raise ValueError('Empty or invalid source paragraphs')
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict) or set(paragraph) != {'id', 'section', 'text'}:
                raise ValueError('Paragraph schema mismatch')
            if not all(isinstance(v, str) for v in paragraph.values()) or not paragraph['id'] or not paragraph['text'].strip():
                raise ValueError('Invalid source paragraph')
        if len({p['id'] for p in paragraphs}) != len(paragraphs):
            raise ValueError('Duplicate paragraph IDs')
        if row['prompt_mode'] == 'baseline' and not row['context'].strip():
            raise ValueError('The original baseline context is required')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate question IDs')
    return rows


def trial_messages(row):
    # Explicit projections keep identities, labels and provenance out of prompts.
    if row['prompt_mode'] == 'baseline':
        return messages_for({'context': row['context'], 'question': row['question']})
    return live_messages({'title': row['title'], 'question': row['question'], 'paragraphs': row['paragraphs']})


def validate_adapter(adapter, cfg):
    if adapter is None:
        return None
    if cfg['model_id'] != 'CohereLabs/tiny-aya-global' or cfg['revision'] != TINY_REVISION:
        raise ValueError('The D adapter is permitted only on its pinned Tiny Aya Global base, never 8B')
    adapter = Path(adapter)
    manifest_path = adapter.parent / 'manifest.json'
    weights = adapter / 'adapter_model.safetensors'
    config_path = adapter / 'adapter_config.json'
    reference = decode(manifest_path.read_bytes())
    base = reference.get('base_model', {})
    if (hash_file(manifest_path) != D_MANIFEST_SHA256 or reference.get('training_complete') is not True
            or base.get('model_id') != cfg['model_id'] or base.get('revision') != cfg['revision']
            or reference.get('adapter_sha256') != D_ADAPTER_SHA256 or hash_file(weights) != D_ADAPTER_SHA256):
        raise ValueError('D adapter lineage or file hash mismatch')
    adapter_cfg = decode(config_path.read_bytes())
    if adapter_cfg.get('peft_type') != 'LORA' or adapter_cfg.get('base_model_name_or_path') != cfg['model_id']:
        raise ValueError('D adapter configuration does not identify its expected base')
    return {'path': str(adapter.resolve()), 'label': 'D', 'weights_sha256': D_ADAPTER_SHA256,
            'training_manifest_sha256': D_MANIFEST_SHA256, 'config_sha256': hash_file(config_path),
            'base_model_id': cfg['model_id'], 'base_revision': cfg['revision'],
            'autocast_adapter_dtype': True,
            'note': 'PEFT may keep/promote adapter weights to FP32; effective loaded dtypes are audited separately.'}


def validate_protocol(protocol_path, cfg, rows, input_bytes, adapter, root=ROOT, config_source=None):
    """Read protocol metadata only; never open the evaluation-label file."""
    if protocol_path is None:
        return None
    raw = Path(protocol_path).read_bytes()
    protocol = decode(raw)
    if protocol.get('inputs_sha256') != sha(input_bytes):
        raise ValueError('Frozen protocol input SHA mismatch')
    if protocol.get('selected_question_ids') != [r['id'] for r in rows]:
        raise ValueError('Frozen protocol input order/IDs mismatch')
    if set(protocol.get('input_fields', [])) != ROW_KEYS:
        raise ValueError('Frozen protocol source schema mismatch')
    arm = cfg.get('label')
    expected = protocol.get('models', {}).get(arm)
    if not expected or any(expected.get(k) != cfg[k] for k in ('model_id', 'revision')):
        raise ValueError('Frozen protocol model arm/revision mismatch')
    if 'config_sha256' in expected and (config_source is None or sha(config_source) != expected['config_sha256']):
        raise ValueError('Frozen protocol configuration file SHA mismatch')
    expected_adapter = expected.get('adapter')
    if (adapter is None) != (expected_adapter is None):
        raise ValueError('Frozen protocol adapter presence mismatch')
    if adapter is not None:
        if (Path(adapter['path']).resolve() != (Path(root) / expected_adapter['path']).resolve()
                or any(adapter[k] != expected_adapter[k] for k in ('weights_sha256', 'config_sha256'))):
            raise ValueError('Frozen protocol adapter identity mismatch')
    expected_generation = {**{k: cfg[k] for k in ('max_total_tokens', 'max_new_tokens',
        'do_sample', 'seed', 'attention_implementation', 'dtype')}, 'max_input_tokens': 7936,
        'automatic_retries': 0, 'batch_size': 1}
    if protocol.get('generation') != expected_generation:
        raise ValueError('Frozen protocol generation settings mismatch')
    if protocol.get('quantization') != QUANTIZATION:
        raise ValueError('Frozen protocol quantization mismatch')
    hashes = protocol.get('logical_messages_sha256', {})
    computed = {r['id']: sha(logical_message_bytes(trial_messages(r))) for r in rows}
    if hashes != computed:
        raise ValueError('Frozen protocol logical messages mismatch')
    code = protocol.get('computation_files_sha256', {})
    if set(code) != PROTOCOL_CODE_FILES:
        raise ValueError('Frozen protocol computation-file allowlist mismatch')
    for filename, expected_hash in code.items():
        if hash_file(Path(root) / filename) != expected_hash:
            raise ValueError('Frozen protocol code hash mismatch: ' + filename)
    return {'path': str(Path(protocol_path).resolve()), 'sha256': sha(raw), 'arm': arm,
            'logical_messages_verified': len(rows), 'computation_files_sha256': code,
            'evaluation_labels_read': False}


def prepare_rows(rows, tokenizer, input_budget):
    prepared, accounting, prompts = [], [], []
    for row in rows:
        item = {'id': row['id'], 'paper_id': row['paper_id'], 'prompt_mode': row['prompt_mode'],
                'source_sha256': sha(canonical(row['paragraphs'])), 'input_tokens': None,
                'max_input_tokens': input_budget, 'fits': False, 'prepared': False,
                'truncation': False, 'messages_sha256': None, 'logical_messages_sha256': None,
                'canonical_messages_sha256': None, 'prompt_sha256': None}
        cpu = None
        prompt_record = {'id': row['id'], 'messages': None, 'rendered_prompt': None}
        try:
            messages = trial_messages(row)
            item['messages_sha256'] = sha(logical_message_bytes(messages))
            item['logical_messages_sha256'] = item['messages_sha256']
            item['canonical_messages_sha256'] = sha(canonical(messages))
            prompt_record['messages'] = messages
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not isinstance(prompt, str):
                raise ValueError('Chat template did not return text')
            item['prompt_sha256'] = sha(prompt.encode())
            prompt_record['rendered_prompt'] = prompt
            cpu = tokenizer(prompt, add_special_tokens=False, return_tensors='pt', truncation=False)
            count = int(cpu['input_ids'].shape[-1])
            if count < 1:
                raise ValueError('Empty tokenized input')
            item.update(input_tokens=count, fits=count <= input_budget, prepared=True,
                        input_ids_sha256=sha(canonical(cpu['input_ids'][0].tolist())))
        except Exception as exc:
            item.update(preflight_error=f'{type(exc).__name__}: {exc}')
            cpu = None
        prepared.append((row, cpu, item))
        accounting.append(item)
        prompts.append(prompt_record)
    return prepared, accounting, prompts


def generation_options(model, tokenizer, cfg):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    signature = inspect.signature(base.forward)
    # Cohere and Cohere2 versions differ. Never rely on permissive **kwargs.
    logits_key = next((name for name in ('logits_to_keep', 'num_logits_to_keep') if name in signature.parameters), None)
    eos = model.generation_config.eos_token_id
    options = {'max_new_tokens': cfg['max_new_tokens'], 'do_sample': False,
               'use_cache': True, 'pad_token_id': tokenizer.pad_token_id, 'eos_token_id': eos}
    if logits_key:
        options[logits_key] = 1
    return options, {'forward_signature': str(signature), 'last_logits_parameter': logits_key,
                     'last_logits_optimization': logits_key is not None}


class DeviceMemorySampler:
    """Global device use sampled at 20 Hz, not a process-specific exact peak."""
    def __init__(self, torch, interval=0.05):
        self.torch, self.interval = torch, interval
        self.stop_event = threading.Event()
        self.thread = None
        self.values, self.errors = [], []

    def sample(self):
        try:
            free, total = self.torch.cuda.mem_get_info(0)
            self.values.append({'free_bytes': int(free), 'total_bytes': int(total),
                                'used_bytes': int(total - free)})
        except Exception as exc:
            if len(self.errors) < 3:
                self.errors.append(type(exc).__name__ + ': ' + str(exc))

    def start(self):
        self.sample()
        def worker():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.sample()
        return {'sampling_interval_seconds': self.interval, 'samples': len(self.values),
                'before': self.values[0] if self.values else None,
                'after': self.values[-1] if self.values else None,
                'peak_device_used_bytes_sampled': max((v['used_bytes'] for v in self.values), default=None),
                'sample_errors': self.errors,
                'scope': 'Global GPU 0 usage includes other processes; sampled maximum is not an exact peak.'}


def run_case(row, cpu, item, model, tokenizer, torch, cfg, options, sampler_factory=DeviceMemorySampler):
    result = {**{k: item.get(k) for k in ('id', 'paper_id', 'prompt_mode', 'source_sha256',
               'messages_sha256', 'logical_messages_sha256', 'canonical_messages_sha256',
               'prompt_sha256', 'input_tokens', 'max_input_tokens')},
              'question': row['question'], 'preflight_fits': item['fits'],
              'raw_output': None, 'prediction': None, 'parse_error': None,
              'citations': [], 'generation_attempts': 0, 'automatic_retries': 0,
              'generation_seconds': None, 'gpu_generation_ms': None,
              'generated_tokens': None, 'hit_token_limit': False}
    if not item['prepared']:
        result.update(status='preflight_error', error=item['preflight_error'])
        return result
    if not item['fits']:
        result.update(status='skipped_over_token_budget')
        return result
    inputs = output = None
    sampler = sampler_factory(torch)
    started = None
    try:
        sampler.start()
        torch.cuda.reset_peak_memory_stats(0)
        inputs = {key: value.to('cuda:0') for key, value in cpu.items()}
        torch.cuda.synchronize(0)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start_event.record()
        result['generation_attempts'] = 1
        with torch.inference_mode():
            output = model.generate(**inputs, **options)
        end_event.record()
        end_event.synchronize()
        torch.cuda.synchronize(0)
        result['generation_seconds'] = time.perf_counter() - started
        result['gpu_generation_ms'] = float(start_event.elapsed_time(end_event))
        generated = output[0, item['input_tokens']:].tolist()
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        prediction, error, wrapper = parse_prediction(raw, {p['id'] for p in row['paragraphs']})
        eos = options['eos_token_id']
        eos = [eos] if isinstance(eos, int) else (eos or [])
        result.update(status='ok' if error is None else 'invalid_output', raw_output=raw,
                      prediction=prediction, parse_error=error, output_wrapper=wrapper,
                      generated_token_ids=generated, generated_tokens=len(generated),
                      hit_token_limit=len(generated) >= cfg['max_new_tokens'] and bool(generated) and generated[-1] not in eos)
        if prediction:
            by_id = {p['id']: p for p in row['paragraphs']}
            result['citations'] = [by_id[pid] for pid in prediction['evidence_ids']]
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status='cuda_out_of_memory', error=f'{type(exc).__name__}: {exc}')
    except Exception as exc:
        result.update(status='generation_error', error=f'{type(exc).__name__}: {exc}')
    finally:
        if started is not None:
            result['attempt_wall_seconds'] = time.perf_counter() - started
        try:
            result.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                          peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)))
            result['peak_allocated_gib'] = result['peak_allocated_bytes'] / 2**30
            result['peak_reserved_gib'] = result['peak_reserved_bytes'] / 2**30
        except Exception as exc:
            result['memory_stat_error'] = f'{type(exc).__name__}: {exc}'
        try:
            result['device_memory'] = sampler.finish()
        except Exception as exc:
            result['device_memory_error'] = f'{type(exc).__name__}: {exc}'
        del inputs, output
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            result['cleanup_error'] = f'{type(exc).__name__}: {exc}'
    return result


def audit_loaded_model(model, bnb):
    dtypes, devices, adapter_dtypes = {}, Counter(), {}
    def count(container, name, parameter):
        item = container.setdefault(name, {'tensors': 0, 'stored_elements': 0})
        item['tensors'] += 1
        item['stored_elements'] += parameter.numel()
    for name, parameter in model.named_parameters():
        count(dtypes, str(parameter.dtype), parameter)
        devices[str(parameter.device)] += 1
        if 'lora_' in name:
            count(adapter_dtypes, str(parameter.dtype), parameter)
    if set(devices) != {'cuda:0'}:
        raise RuntimeError('Model parameters are not entirely on CUDA 0; CPU/disk/meta offload is forbidden')
    quantized = []
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            state = getattr(module.weight, 'quant_state', None)
            if state is None or state.quant_type != 'nf4' or str(module.compute_dtype) != 'torch.bfloat16':
                raise RuntimeError('Loaded quantization is not initialized NF4 / BF16: ' + name)
            quantized.append({'name': name, 'compute_dtype': str(module.compute_dtype),
                              'storage_dtype': str(module.weight.dtype), 'quant_type': state.quant_type,
                              'nested_quantization': bool(getattr(state, 'nested', False))})
    if not quantized or not all(x['nested_quantization'] for x in quantized):
        raise RuntimeError('Expected double-quantized Linear4bit modules were not loaded')
    return {'class': type(model).__name__, 'is_loaded_in_4bit': bool(getattr(model, 'is_loaded_in_4bit', False)),
            'device_map': getattr(model, 'hf_device_map', None), 'parameter_device_counts': dict(devices),
            'parameter_storage_dtype_counts': dtypes, 'adapter_parameter_dtype_counts': adapter_dtypes,
            'linear4bit_module_count': len(quantized), 'linear4bit_modules': quantized,
            'reported_logical_parameter_count': model.num_parameters(),
            'dtype_note': 'Packed weight storage elements are not logical parameter counts. BF16 describes Linear4bit compute; adapter branches use their recorded loaded dtype.'}


def file_inventory(directory):
    return {str(p.relative_to(directory)): {'bytes': p.stat().st_size, 'sha256': hash_file(p)}
            for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def cached_model_inventory(cfg):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(cfg['model_id'], revision=cfg['revision'], local_files_only=True))
    files = [p for p in snapshot.iterdir() if p.is_file() and
             (p.suffix == '.safetensors' or p.name.endswith('.index.json') or p.name in {'config.json', 'generation_config.json'})]
    return {'snapshot_path': str(snapshot), 'requested_revision': cfg['revision'],
            'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hash_file(p)} for p in sorted(files)}}


def run_summary(results, preflight_fits):
    return {'questions': len(results), 'status_counts': dict(Counter(r['status'] for r in results)),
            'semantic_accuracy_assessed': False, 'preflight_fits': preflight_fits,
            'joint_comparison_subset_rule': 'Use the intersection of preflight fits, never only successful generations.',
            'valid_answered': sum(r['prediction'] is not None and r['prediction']['answerable'] for r in results),
            'valid_refused': sum(r['prediction'] is not None and not r['prediction']['answerable'] for r in results)}


def persist_failed_outcomes(run_dir, rows, accounting, prompts, error, stage):
    """Preserve any written prefix and explicitly account for every unrun row."""
    run_dir = Path(run_dir)
    if not accounting:
        for row in rows:
            messages = trial_messages(row)
            logical_hash = sha(logical_message_bytes(messages))
            accounting.append({'id': row['id'], 'paper_id': row['paper_id'], 'prompt_mode': row['prompt_mode'],
                'source_sha256': sha(canonical(row['paragraphs'])), 'input_tokens': None,
                'max_input_tokens': 7936, 'fits': False, 'prepared': False, 'truncation': False,
                'messages_sha256': logical_hash, 'logical_messages_sha256': logical_hash,
                'canonical_messages_sha256': sha(canonical(messages)), 'prompt_sha256': None,
                'preflight_error': error})
            prompts.append({'id': row['id'], 'messages': messages, 'rendered_prompt': None})
    if [r['id'] for r in accounting] != [r['id'] for r in rows]:
        raise RuntimeError('Cannot safely complete malformed preflight accounting')
    accounting_path = run_dir / 'token_accounting.json'
    prompts_path = run_dir / 'prompts.jsonl'
    if not accounting_path.exists():
        save_json(accounting_path, accounting)
    if not prompts_path.exists():
        with prompts_path.open('x', encoding='utf-8') as out:
            for record in prompts:
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                out.flush()
    prediction_path = run_dir / 'predictions.jsonl'
    existing = ([decode(line) for line in prediction_path.read_text().splitlines()]
                if prediction_path.exists() else [])
    if [r['id'] for r in existing] != [r['id'] for r in rows[:len(existing)]]:
        raise RuntimeError('Cannot safely append to non-prefix prediction records')
    with prediction_path.open('a', encoding='utf-8') as out:
        for row, item in zip(rows[len(existing):], accounting[len(existing):]):
            status = ('preflight_error' if not item['prepared'] else
                      'skipped_over_token_budget' if not item['fits'] else 'setup_failed')
            result = {**{k: item.get(k) for k in ('id', 'paper_id', 'prompt_mode', 'source_sha256',
                'messages_sha256', 'logical_messages_sha256', 'canonical_messages_sha256',
                'prompt_sha256', 'input_tokens', 'max_input_tokens')},
                'question': row['question'], 'preflight_fits': item['fits'],
                'status': status, 'failed_stage': stage, 'error': error,
                'raw_output': None, 'prediction': None, 'parse_error': None, 'citations': [],
                'generation_attempts': 0, 'automatic_retries': 0, 'generation_seconds': None,
                'gpu_generation_ms': None, 'generated_tokens': None, 'hit_token_limit': False}
            existing.append(result)
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
            out.flush()
    save_json(run_dir / 'run-summary.json', run_summary(existing, sum(i['fits'] for i in accounting)))
    return {'outcome_records_complete': len(existing) == len(rows),
            'status_counts': dict(Counter(r['status'] for r in existing)),
            'token_budget_passed': sum(i['fits'] for i in accounting),
            'token_budget_skipped': sum(i['prepared'] and not i['fits'] for i in accounting),
            'preflight_errors': sum(not i['prepared'] for i in accounting),
            'token_accounting_sha256': hash_file(accounting_path), 'prompts_sha256': hash_file(prompts_path),
            'predictions_sha256': hash_file(prediction_path), 'summary_sha256': hash_file(run_dir / 'run-summary.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model-id')
    parser.add_argument('--revision')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, help='Bind the frozen source-only experiment protocol; mandatory for formal comparison runs')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--allow-download', action='store_true', help='Default uses only the existing local HF cache')
    args = parser.parse_args()
    cfg_source = args.config.read_bytes()
    cfg = decode(cfg_source)
    for key, value in [('model_id', args.model_id), ('revision', args.revision)]:
        if value is not None:
            cfg[key] = value
    cfg = validate_config(cfg)
    data = args.data.read_bytes()
    rows = read_rows(data)
    adapter = validate_adapter(args.adapter, cfg)
    protocol = validate_protocol(args.protocol, cfg, rows, data, adapter, config_source=cfg_source)
    if args.run_dir.exists():
        raise FileExistsError('Run directory already exists; never overwrite previous outputs')
    args.run_dir.mkdir(parents=True)
    code = args.run_dir / 'code'
    code.mkdir()
    for source in SOURCE_FILES:
        shutil.copyfile(ROOT / source, code / Path(source).name)
    (args.run_dir / 'inputs.jsonl').write_bytes(data)
    (args.run_dir / 'requested-config.json').write_bytes(cfg_source)
    save_json(args.run_dir / 'effective-config.json', cfg)
    if args.protocol:
        shutil.copyfile(args.protocol, args.run_dir / 'protocol.json')
    if adapter:
        target = args.run_dir / 'adapter-provenance'
        target.mkdir()
        shutil.copyfile(args.adapter / 'adapter_config.json', target / 'adapter_config.json')
        shutil.copyfile(args.adapter.parent / 'manifest.json', target / 'training-manifest.json')
    manifest = {'schema_version': 1, 'status': 'preparing', 'started_at_utc': utc_now(),
                'config': cfg, 'config_sha256': sha(canonical(cfg)), 'config_source_sha256': sha(cfg_source),
                'data_path': str(args.data.resolve()), 'input_sha256': sha(data),
                'selected_question_ids': [r['id'] for r in rows], 'questions': len(rows),
                'unique_papers': len({r['paper_id'] for r in rows}),
                'prompt_mode_counts': dict(Counter(r['prompt_mode'] for r in rows)),
                'code_sha256': {source: hash_file(ROOT / source) for source in SOURCE_FILES}, 'protocol': protocol,
                'python': platform.python_version(), 'platform': platform.platform(), 'adapter': adapter,
                'quantization': QUANTIZATION, 'batch_size': 1, 'device': 'cuda:0', 'offload': False,
                'automatic_retries': 0, 'truncation': False, 'prepare_only': args.prepare_only,
                'local_files_only': not args.allow_download, 'model_loaded': False,
                'preflight_complete': False, 'inference_complete': False,
                'independent_human_review': False, 'human_review_status': 'pending'}
    save_json(args.run_dir / 'manifest.json', manifest)
    stage = 'import_runtime'
    prepared, accounting, prompts = [], [], []
    try:
        import torch
        import bitsandbytes as bnb
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        manifest['packages'] = {name: importlib.metadata.version(name) for name in
                               ['torch', 'transformers', 'bitsandbytes', 'accelerate', 'peft', 'huggingface-hub']}
        stage = 'load_tokenizer_and_config'
        load_kwargs = {'revision': cfg['revision'], 'trust_remote_code': False, 'local_files_only': not args.allow_download}
        model_config = AutoConfig.from_pretrained(cfg['model_id'], **load_kwargs)
        if model_config.model_type != MODEL_TYPES[cfg['model_id']]:
            raise ValueError('Unexpected model architecture: ' + model_config.model_type)
        actual_commit = getattr(model_config, '_commit_hash', None)
        if actual_commit is not None and actual_commit != cfg['revision']:
            raise ValueError('Resolved config commit differs from requested revision')
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], **load_kwargs)
        tokenizer.save_pretrained(args.run_dir / 'tokenizer')
        model_config.save_pretrained(args.run_dir / 'model-config')
        total = min(cfg['max_total_tokens'], int(model_config.max_position_embeddings))
        input_budget = total - cfg['max_new_tokens']
        if input_budget < 1:
            raise ValueError('Model has no usable input budget')
        manifest.update(model_type=model_config.model_type, resolved_config_commit=actual_commit,
                        tokenizer={'class': type(tokenizer).__name__, 'name_or_path': tokenizer.name_or_path,
                                   'vocab_size': tokenizer.vocab_size, 'total_token_count': len(tokenizer),
                                   'special_tokens_map': tokenizer.special_tokens_map,
                                   'files': file_inventory(args.run_dir / 'tokenizer')},
                        model_config_files=file_inventory(args.run_dir / 'model-config'),
                        max_total_tokens=total, max_input_tokens=input_budget, max_new_tokens=cfg['max_new_tokens'])
        stage = 'tokenize_all_rows'
        prepared, accounting, prompts = prepare_rows(rows, tokenizer, input_budget)
        save_json(args.run_dir / 'token_accounting.json', accounting)
        with (args.run_dir / 'prompts.jsonl').open('x', encoding='utf-8') as out:
            for prompt in prompts:
                out.write(json.dumps(prompt, ensure_ascii=False) + '\n')
                out.flush()
        manifest.update(preflight_complete=True, status='prepared',
                        token_budget_passed=sum(i['fits'] for i in accounting),
                        token_budget_skipped=sum(i['prepared'] and not i['fits'] for i in accounting),
                        preflight_errors=sum(not i['prepared'] for i in accounting),
                        token_accounting_sha256=hash_file(args.run_dir / 'token_accounting.json'),
                        prompts_sha256=hash_file(args.run_dir / 'prompts.jsonl'))
        save_json(args.run_dir / 'manifest.json', manifest)
        print(json.dumps({'stage': 'prepared', 'questions': len(rows), 'fits': manifest['token_budget_passed'],
                          'skipped': manifest['token_budget_skipped'], 'preflight_errors': manifest['preflight_errors']}), flush=True)
        if args.prepare_only:
            manifest['completed_at_utc'] = utc_now()
            save_json(args.run_dir / 'manifest.json', manifest)
            return
        if not manifest['token_budget_passed']:
            raise RuntimeError('No prepared inputs fit; no weights loaded')
        stage = 'load_model'
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 0 is required; CPU execution/offload is forbidden')
        torch.cuda.set_device(0)
        set_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        props = torch.cuda.get_device_properties(0)
        manifest.update(status='loading_model', gpu=props.name, gpu_total_memory_bytes=int(props.total_memory),
                        cuda_runtime=torch.version.cuda, cuda_device_count=torch.cuda.device_count())
        save_json(args.run_dir / 'manifest.json', manifest)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(0)
        model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], **load_kwargs,
                    dtype=torch.bfloat16, quantization_config=quantization,
                    device_map={'': 'cuda:0'}, attn_implementation=cfg['attention_implementation']).eval()
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False,
                                             autocast_adapter_dtype=True).eval()
        torch.cuda.synchronize(0)
        manifest.update(model_loaded=True, model_load_seconds=time.perf_counter() - started,
                        load_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                        load_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
                        loaded_model=audit_loaded_model(model, bnb),
                        actual_quantization_config=quantization.to_dict())
        model.generation_config.save_pretrained(args.run_dir / 'generation-config')
        options, support = generation_options(model, tokenizer, cfg)
        manifest.update(generation_kwargs=options, generation_support=support,
                        model_cache=cached_model_inventory(cfg), status='running')
        save_json(args.run_dir / 'manifest.json', manifest)
        stage = 'inference'
        results = []
        with (args.run_dir / 'predictions.jsonl').open('x', encoding='utf-8') as out:
            for row, cpu, item in prepared:
                result = run_case(row, cpu, item, model, tokenizer, torch, cfg, options)
                results.append(result)
                out.write(json.dumps(result, ensure_ascii=False) + '\n')
                out.flush()
                print(json.dumps({'completed': len(results), 'total': len(rows), 'id': row['id'],
                                  'status': result['status'], 'seconds': result.get('generation_seconds')}), flush=True)
        counts = dict(Counter(r['status'] for r in results))
        save_json(args.run_dir / 'run-summary.json', run_summary(results, manifest['token_budget_passed']))
        manifest.update(status='completed', inference_complete=True, outcome_records_complete=True, completed_at_utc=utc_now(),
                        status_counts=counts, predictions_sha256=hash_file(args.run_dir / 'predictions.jsonl'),
                        summary_sha256=hash_file(args.run_dir / 'run-summary.json'))
        save_json(args.run_dir / 'manifest.json', manifest)
    except Exception as exc:
        manifest.update(status='failed', inference_complete=False, failed_stage=stage, error_type=type(exc).__name__,
                        error=str(exc), failed_at_utc=utc_now())
        try:
            manifest.update(persist_failed_outcomes(args.run_dir, rows, accounting, prompts,
                            f'{type(exc).__name__}: {exc}', stage))
        except Exception as record_exc:
            manifest.update(outcome_records_complete=False,
                            failure_record_error=f'{type(record_exc).__name__}: {record_exc}')
        save_json(args.run_dir / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
