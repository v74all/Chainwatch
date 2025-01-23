from typing import Any, Optional, Union, Dict
import json
import os
import gzip
import hashlib
from functools import lru_cache

_file_cache: Dict[str, Any] = {}
_MAX_CACHE_SIZE = 100

def _get_file_hash(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""

@lru_cache(maxsize=_MAX_CACHE_SIZE)
def load_json_file(file_path: str, default: Any = None, use_cache: bool = True) -> Optional[Any]:
    try:
        gz_path = f"{file_path}.gz"
        if os.path.exists(gz_path):
            with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                if use_cache:
                    _file_cache[file_path] = {'data': data, 'hash': _get_file_hash(gz_path)}
                return data
                
        if not os.path.exists(file_path):
            return {} if default is None else default
            
        if use_cache:
            file_hash = _get_file_hash(file_path)
            if file_path in _file_cache and _file_cache[file_path]['hash'] == file_hash:
                return _file_cache[file_path]['data']
                
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if use_cache:
                _file_cache[file_path] = {'data': data, 'hash': _get_file_hash(file_path)}
            return data
            
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return {} if default is None else default
    except Exception as e:
        print(f"Unexpected error loading {file_path}: {e}")
        return {} if default is None else default

def save_json_file(data: Any, file_path: str, compress: bool = False) -> bool:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if compress:
            with gzip.open(f"{file_path}.gz", 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
        if file_path in _file_cache:
            _file_cache[file_path] = {'data': data, 'hash': _get_file_hash(file_path)}
            
        return True
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")
        return False

def clear_cache() -> None:
    _file_cache.clear()
    load_json_file.cache_clear()
