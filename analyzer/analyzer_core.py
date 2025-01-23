import os
import asyncio
import aiohttp
import pandas as pd
import time
import numpy as np
import networkx as nx
import random
from collections import Counter, deque
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy import stats
from datetime import timedelta
import warnings
import joblib
from dotenv import load_dotenv
from termcolor import colored
import cachetools
import logging
import json
from typing import TypeVar, Generic, Optional, Any, Dict, List, Set, Union, Tuple, Callable, List
import uuid
from dataclasses import dataclass
from ratelimit import limits, sleep_and_retry
from utils.file_utils import load_json_file, save_json_file
from concurrent.futures import ThreadPoolExecutor
from utils.logger import ChainWatchLogger
from utils.messages import ERROR_MESSAGES
from utils.json_utils import ChainJSONEncoder

default_exchanges = {
    'Ethereum': [],
    'BSC': [],
    'Tron': [],
    'Avalanche': [],
    'Cardano': [],
    'Litecoin': [],
    'Near': [],
    'Harmony': [],
    'Monero': [],
    'Polkadot': [],
    'Kusama': [],
    'Flow': [],
    'Hedera': [],
    'IOTA': [],
    'XRP': [],
    'EOS': [],
    'NEO': [],
    'WAVES': [],
    'ICON': [],
    'THETA': [],
    'ZILLIQA': [],
    'TEZOS': [],
    'Algorand': [],
    'Fantom': [],
    'Cosmos': [],
    'Elrond': [],
    'Linea': []
}

from .api_config import (
    get_chain_config,
    get_api_endpoint,
    get_validation_endpoint,
    CONFIG,
    API_ENDPOINTS,
    UNIT_CONVERSIONS,
    RATE_LIMITS,
    APIErrorHandler,
    API_RETRY_CONFIG
)

warnings.filterwarnings('ignore')

load_dotenv()

LANGUAGE = os.getenv("LANGUAGE", "en").lower()

def translate(text_fa: str, text_en: str, language: str = None) -> str:
    if language is None:
        language = os.getenv("LANGUAGE", "en").lower()
    return text_fa if language == 'fa' else text_en

BLACKLIST_FILE = 'data/blacklist_addresses.json'
KNOWN_EXCHES_FILE = 'data/known_exchanges.json'
FRAUD_PROBABILITY_THRESHOLD = 50
MAX_DEPTH = 3

class APIError(Exception):
    pass

class BlockchainError(Exception):
    pass

class AnalysisError(Exception):
    pass

class ChainAPIError(Exception):
    def __init__(self, chain: str, message: str, status_code: Optional[int] = None):
        self.chain = chain
        self.status_code = status_code
        super().__init__(f"{chain} API Error: {message}")

class APIRetryStrategy:
    def __init__(self, config: dict = None):
        self.config = config or API_RETRY_CONFIG
        
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        last_exception = None
        for attempt in range(self.config['max_retries']):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if not self._should_retry(e, attempt):
                    raise
                
                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)
        
        raise last_exception

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        if isinstance(exception, ChainAPIError):
            if APIErrorHandler.is_rate_limit_error(exception.chain, exception.status_code):
                return True
            if APIErrorHandler.is_server_error(exception.chain, exception.status_code):
                return True
        return False

    def _calculate_delay(self, attempt: int) -> float:
        delay = min(
            self.config['base_delay'] * (self.config['exponential_base'] ** attempt),
            self.config['max_delay']
        )
        jitter = random.uniform(-self.config['jitter'], self.config['jitter'])
        return delay * (1 + jitter)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('chainwatch_analyzer')

T = TypeVar('T')

@dataclass
class CacheEntry(Generic[T]):
    value: T
    timestamp: float
    access_count: int = 0

class EnhancedAPICache:
    def __init__(self, ttl=3600, max_size=10000, calls=5, period=1):
        self.cache = cachetools.TTLCache(maxsize=max_size, ttl=ttl)
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
        self.calls = calls
        self.period = period

    @sleep_and_retry
    @limits(calls=5, period=1)
    async def get(self, key: str) -> Optional[Any]:
        async with self.lock:
            try:
                entry = self.cache[key]
                entry.access_count += 1
                self.hits += 1
                return entry.value
            except KeyError:
                self.misses += 1
                return None

    @sleep_and_retry
    @limits(calls=5, period=1)
    async def set(self, key: str, value: Any) -> None:
        async with self.lock:
            self.cache[key] = CacheEntry(value=value, timestamp=time.time())

    def get_stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {
            'hit_ratio': self.hits / total if total > 0 else 0,
            'size': len(self.cache),
            'max_size': self.cache.maxsize
        }

class EnhancedMLAnalysis:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)

    def fit(self, X: pd.DataFrame):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.isolation_forest.fit(X_scaled)
        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        X_scaled = self.scaler.transform(X)
        return {
            'anomaly_scores': self.isolation_forest.predict(X_scaled),
            'principal_components': self.pca.transform(X_scaled)
        }

class ChainWatchAnalyzer:
    KNOWN_EXCHANGES_FILE = 'data/known_exchanges.json'
    RISK_MODEL_PATH = 'models/random_forest_risk_model.joblib'
    
    def __init__(self, addresses: List[str], log_callback=None, language: str = "en", config_override: dict = None):
        self.logger = ChainWatchLogger(language)
        self.addresses = addresses
        self.log_callback = log_callback or (lambda msg, color: print(colored(msg, color)))
        self.language = language
        self.timeout = aiohttp.ClientTimeout(total=60)
        self.api_calls = {}
        self.results = {'transactions': [], 'suspicious_addresses': [], 'fraud_probabilities': []}
        self.api_cache = EnhancedAPICache(ttl=3600)
        self.chain_configs = {chain: get_chain_config(chain) for chain in CONFIG.keys()}
        
        try:
            self.known_exchanges = load_json_file(self.KNOWN_EXCHANGES_FILE)
            if not isinstance(self.known_exchanges, dict):
                self.known_exchanges = default_exchanges
        except:
            self.known_exchanges = default_exchanges
            
        try:
            self.blacklist_addresses = set(load_json_file(BLACKLIST_FILE, []))
        except:
            self.blacklist_addresses = set()

        self.parsers = self._initialize_parsers()
        
        if config_override:
            self._apply_config_override(config_override)

        self._init_performance_monitoring()

        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self.api_semaphore = asyncio.Semaphore(10)
        
        self.performance_metrics = {
            'api_calls': Counter(),
            'processing_times': [],
            'memory_usage': [],
            'analysis_durations': []
        }

        self.risk_model = self._load_risk_model()
        self.ml_analyzer = EnhancedMLAnalysis()
        self.config = CONFIG  
        self.api_endpoints = API_ENDPOINTS
        self.unit_conversions = UNIT_CONVERSIONS
        self.rate_limits = RATE_LIMITS
        self.retry_strategy = APIRetryStrategy()
        self.api_connection_pool = aiohttp.TCPConnector(
            limit=50,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=False
        )
        self.session_pool = {}
        self.rate_limiters = {
            chain: AsyncLimiter(
                limit['calls'],
                limit['period']
            ) for chain, limit in RATE_LIMITS.items()
        }
        self.error_counts = Counter()
        self.last_errors = deque(maxlen=100)
    
    def translate(self, text_fa: str, text_en: str = None) -> str:
        if text_en is None:
            text_en = text_fa
        return text_fa if self.language == 'fa' else text_en

    def _apply_config_override(self, override: dict) -> None:
        for chain in CONFIG.keys():
            if chain in override:
                CONFIG[chain].update(override[chain])
                
        if 'analysis' in override:
            self.analysis_batch_size = override['analysis'].get('batch_size', 1000)
            self.api_cache.ttl = override['analysis'].get('cache_ttl', 3600)
                
    def _init_performance_monitoring(self) -> None:
        pass

    def _initialize_parsers(self) -> Dict[str, Callable]:
        base_parser = lambda tx, timestamp_field='timestamp', unit=1e18: {
            "hash": tx.get("hash"),
            "timestamp": pd.to_datetime(tx.get(timestamp_field), unit='s'),
            "ownerAddress": tx.get("from") or tx.get("ownerAddress"),
            "toAddress": tx.get("to") or tx.get("toAddress", tx.get("toAddressList", [""])[0]),
            "amount": float(tx.get("amount", tx.get("value", 0))) / unit
        }

        return {
            'Tron': self._parse_tron_tx,
            'Ethereum': lambda tx: base_parser(tx, unit=1e18),
            'BSC': lambda tx: base_parser(tx, unit=1e18),
            'Solana': lambda tx: base_parser(tx, timestamp_field='block_time', unit=1e9),
            'Linea': lambda tx: base_parser(tx, unit=1e18),
            'Cardano': lambda tx: base_parser(tx, unit=1e6),
            'Litecoin': lambda tx: base_parser(tx, unit=1e8),
            'Tezos': lambda tx: base_parser(tx, unit=1e6),
            'Cosmos': lambda tx: base_parser(tx, unit=1e6),
            'Algorand': lambda tx: base_parser(tx, unit=1e6),
            'VeChain': lambda tx: base_parser(tx, unit=1e18),
            'Dogecoin': lambda tx: base_parser(tx, unit=1e8),
            'Stellar': lambda tx: base_parser(tx),
            'Near': lambda tx: base_parser(tx, unit=1e24),
            'Harmony': lambda tx: base_parser(tx, unit=1e18),
            'Monero': lambda tx: base_parser(tx, unit=1e12),
            'Polkadot': lambda tx: base_parser(tx, unit=1e10),
            'Kusama': lambda tx: base_parser(tx, unit=1e12),
            'Flow': lambda tx: base_parser(tx, unit=1e8),
            'Hedera': lambda tx: base_parser(tx, unit=1e8),
            'IOTA': lambda tx: base_parser(tx, unit=1e6),
            'XRP': lambda tx: base_parser(tx, unit=1e6),
            'EOS': lambda tx: base_parser(tx),
            'NEO': lambda tx: base_parser(tx),
            'WAVES': lambda tx: base_parser(tx, unit=1e8),
            'ICON': lambda tx: base_parser(tx, unit=1e18),
            'THETA': lambda tx: base_parser(tx, unit=1e18),
            'ZILLIQA': lambda tx: base_parser(tx, unit=1e12),
            'TEZOS': lambda tx: base_parser(tx, unit=1e6)
        }

    def load_known_patterns(self) -> Dict[str, List[str]]:
        try:
            return load_json_file('data/known_patterns.json', {})
        except:
            return {
                'mixing_services': [],
                'high_risk_addresses': [],
                'temporal_patterns': [],
                'value_patterns': []
            }

    def initialize_visualizer(self):
        pass

    def set_language(self, language: str):
        self.language = language

    def _log_message(self, message: str, color: str = 'cyan', level: str = 'info') -> None:
        self.logger.log(message, level, color)

    def _log_error(self, error_key: str, *args) -> None:
        try:
            message = ERROR_MESSAGES[self.language].get(error_key, "An error occurred.")
            if args:
                try:
                    message = message.format(*args)
                except Exception:
                    message = f"{message} - {' '.join(str(arg) for arg in args)}"
            
            self.logger.error(message)
            
            if error_key == "tron_api_key_error":
                self.logger.error("Please check your TRON API key configuration")
            elif error_key == "api_error":
                self.logger.error("API request failed - check your network connection and API limits")
            elif error_key == "missing_address_or_blockchain":
                self.logger.error("Both wallet address and blockchain must be provided")
                
        except Exception as e:
            self.logger.error(f"Logging error: {str(e)}")

    def _get_tron_api_key(self) -> str:
        config = get_chain_config('Tron')
        api_keys = config.get('api_keys', [])
        
        if not api_keys:
            raise ChainAPIError("Tron", "No API keys configured")
            
        for key in api_keys:
            if key and isinstance(key, str) and len(key) > 0:
                return key
                
        raise ChainAPIError("Tron", "No valid API keys available")

    def _parse_tron_tx(self, tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not tx or not isinstance(tx, dict):
            return None

        try:
            timestamp = pd.to_datetime(tx.get("timestamp", 0), unit="ms")
            
            parsed = {
                "hash": tx.get("hash"),
                "timestamp": timestamp,
                "ownerAddress": tx.get("ownerAddress"),
                "toAddress": tx.get("toAddress", tx.get("toAddressList", [""])[0]),
                "amount": 0
            }

            if tx.get("contractType") == 1:
                amount = tx.get("amount", "0")
                parsed["amount"] = float(amount) / 1_000_000
            elif tx.get("contractType") == 2:
                contract_data = tx.get("contractData", {})
                amount = contract_data.get("amount", "0")
                parsed["amount"] = float(amount) / 1_000_000
            elif tx.get("contractType") == 31:
                trigger_info = tx.get("trigger_info", {})
                if trigger_info.get("methodName") == "transfer":
                    value = trigger_info.get("parameter", {}).get("_value", "0")
                    parsed["amount"] = float(value) / 1_000_000

            return parsed
        except Exception as e:
            self._log_error("error_parsing_tron_tx", str(e))
            return None

    async def fetch_solana_block_time(self, session: aiohttp.ClientSession, slot: int) -> Any:
        url = CONFIG['Solana']['quicknode_url']
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBlockTime",
            "params": [slot]
        }
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("result")
        except aiohttp.ClientError as e:
            self._log_error("error_fetching_block_time", slot, e)
            return None

    def detect_blockchain_from_address(self, address: str) -> str:
        if not address:
            return 'Unknown'

        patterns = {
            'Tron': lambda x: x.startswith('T') and len(x) == 34,
            'Ethereum': lambda x: x.startswith('0x') and len(x) == 42,
            'BSC': lambda x: x.startswith('0x') and len(x) == 42,
            'Solana': lambda x: len(x) == 44 and not x.startswith('0x'),
            'Polygon': lambda x: x.startswith('0x') and len(x) == 42,
            'Avalanche': lambda x: x.startswith('0x') and len(x) == 42,
            'Fantom': lambda x: x.startswith('0x') and len(x) == 42,
            'Arbitrum': lambda x: x.startswith('0x') and len(x) == 42,
            'Optimism': lambda x: x.startswith('0x') and len(x) == 42,
            'Cardano': lambda x: x.startswith('addr1') and len(x) > 50,
            'Litecoin': lambda x: x.startswith('L') and len(x) == 34,
            'Tezos': lambda x: x.startswith('tz1') and len(x) == 36,
            'Cosmos': lambda x: x.startswith('cosmos') and len(x) > 40,
            'Algorand': lambda x: x.startswith('ALG') and len(x) == 58,
            'VeChain': lambda x: x.startswith('0x') and len(x) == 42,
            'Dogecoin': lambda x: x.startswith('D') and len(x) == 34,
            'Stellar': lambda x: x.startswith('G') and len(x) == 56,
            'Near': lambda x: len(x) > 0,
            'Harmony': lambda x: x.startswith('one') and len(x) > 30
        }

        for chain, pattern in patterns.items():
            try:
                if pattern(address):
                    return chain
            except Exception:
                continue

        return 'Unknown'

    def validate_addresses(self, addresses: List[str]) -> Dict[str, List[str]]:
        if not addresses:
            return {chain: [] for chain in CONFIG.keys()}

        validated = {chain: [] for chain in CONFIG.keys()}

        for addr in addresses:
            if not isinstance(addr, str) or len(addr) < 26:
                self._log_error("invalid_address_format", addr)
                continue

            blockchain = self.detect_blockchain_from_address(addr)
            if blockchain != 'Unknown':
                validated[blockchain].append(addr.strip())
            else:
                self._log_error("blockchain_type_not_determined", addr)

        return validated

    async def fetch_transactions_solana(self, session: aiohttp.ClientSession, wallet_address: str) -> List[Dict[str, Any]]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getConfirmedSignaturesForAddress2",
            "params": [wallet_address, {"limit": 50}]
        }
        try:
            async with session.post(CONFIG['Solana']['quicknode_url'], json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("result", [])
        except aiohttp.ClientError as e:
            self._log_error("error_fetching_solana_txs", wallet_address, e)
            return []

    async def fetch_transactions_linea(self, session: aiohttp.ClientSession, wallet_address: str) -> List[Dict[str, Any]]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getTransactionByHash",
            "params": [wallet_address]
        }
        try:
            async with session.post(CONFIG['Linea']['infura_url'], json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return [data.get("result", {})] if data.get("result") else []
        except aiohttp.ClientError as e:
            self._log_error("error_fetching_linea_txs", wallet_address, e)
            return []

    async def _make_api_request(self, chain: str, url: str, params: dict = None, headers: dict = None) -> Dict[str, Any]:
        async def _request():
            try:
                session = await self._get_session(chain)
                
                request_id = str(uuid.uuid4())[:8]
                self._log_message(f"API Request {request_id} to {chain}: {url}", "cyan")
                
                async with self.rate_limiters[chain]:
                    async with session.get(url, params=params, headers=headers) as response:
                        status_code = response.status
                        try:
                            response_data = await response.json()
                        except:
                            response_data = await response.text()
                            raise ChainAPIError(chain, f"Invalid JSON response: {response_data[:200]}")

                        if not APIErrorHandler.validate_response(chain, response_data):
                            error_code = APIErrorHandler.get_error_code(chain, str(response_data))
                            self.error_counts[f"{chain}:{error_code}"] += 1
                            self.last_errors.append({
                                'chain': chain,
                                'code': error_code,
                                'time': time.time()
                            })
                            raise ChainAPIError(chain, f"Response validation failed: {error_code}")

                        self._log_message(f"API Request {request_id} completed successfully", "green")
                        return response_data

            except Exception as e:
                self._log_error(f"API Request {request_id} failed: {str(e)}")
                raise

        return await self.retry_strategy.execute(_request)

    async def close(self):
        for session in self.session_pool.values():
            await session.close()
        await self.api_connection_pool.close()

    def __del__(self):
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                if hasattr(self, 'session_pool'):
                    for session in self.session_pool.values():
                        if not session.closed:
                            session._connector.close()
                if hasattr(self, 'api_connection_pool') and not self.api_connection_pool.closed:
                    self.api_connection_pool.close()
                return

            if hasattr(self, 'session_pool'):
                for session in self.session_pool.values():
                    if not session.closed:
                        asyncio.create_task(session.close())
            if hasattr(self, 'api_connection_pool') and not self.api_connection_pool.closed:
                asyncio.create_task(self.api_connection_pool.close())
        except Exception:
            pass

    async def _get_session(self, chain: str) -> aiohttp.ClientSession:
        if chain not in self.session_pool:
            self.session_pool[chain] = aiohttp.ClientSession(
                connector=self.api_connection_pool,
                timeout=self.timeout,
                headers={'User-Agent': 'ChainWatch-Analyzer/1.0'}
            )
        return self.session_pool[chain]

    @sleep_and_retry
    @limits(calls=5, period=1)
    async def fetch_transactions(self, wallet_address: str, blockchain: str) -> List[Dict[str, Any]]:
        if not wallet_address or not blockchain:
            self._log_error("missing_address_or_blockchain")
            raise ValueError("Wallet address and blockchain must be specified")

        cache_key = f"{blockchain}:{wallet_address}"
        cached_data = await self.api_cache.get(cache_key)
        if cached_data:
            return cached_data

        try:
            endpoint = get_api_endpoint(blockchain, 'transaction')
            if not endpoint:
                raise ChainAPIError(blockchain, "Unsupported blockchain")

            headers = {}
            if blockchain == 'Tron':
                try:
                    tron_key = self._get_tron_api_key()
                    headers['TRON-PRO-API-KEY'] = tron_key
                except ChainAPIError as e:
                    self._log_error("tron_api_key_error", str(e))
                    raise

            response_data = await self._make_api_request(
                blockchain,
                endpoint["url"].format(address=wallet_address),
                endpoint.get("params", {}),
                headers
            )

            transactions = self._extract_transactions(blockchain, response_data)
            if not transactions:
                self._log_message(f"No transactions found for {wallet_address}", "yellow")
                return []

            parsed_txs = [
                tx for tx in (self.parsers[blockchain](tx) for tx in transactions)
                if tx is not None
            ]
            
            await self.api_cache.set(cache_key, parsed_txs)
            return parsed_txs

        except ChainAPIError as e:
            self._log_error("api_error", blockchain, str(e))
            raise
        except Exception as e:
            self._log_error("unexpected_error", blockchain, str(e))
            raise ChainAPIError(blockchain, f"Unexpected error: {str(e)}")

    def _extract_transactions(self, blockchain: str, response_data: dict) -> List[dict]:
        if blockchain in ['Tron', 'Cardano', 'Litecoin', 'Tezos', 'Cosmos', 'Algorand', 'VeChain', 'Dogecoin', 'Stellar', 'Near', 'Harmony']:
            return response_data.get("data", []) or response_data.get("transactions", []) or response_data.get("result", [])
        return response_data.get("result", [])

    async def _fetch_solana_transactions(self, wallet_address: str) -> List[Dict[str, Any]]:
        try:
            transactions = await self._make_api_request(
                'Solana',
                CONFIG['Solana']['quicknode_url'],
                headers={"Content-Type": "application/json"},
                params={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getConfirmedSignaturesForAddress2",
                    "params": [wallet_address, {"limit": 50}]
                }
            )

            block_times = await asyncio.gather(*[
                self._make_api_request(
                    'Solana',
                    CONFIG['Solana']['quicknode_url'],
                    headers={"Content-Type": "application/json"},
                    params={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "getBlockTime",
                        "params": [tx.get("slot")]
                    }
                ) for tx in transactions.get("result", [])
            ])

            return [
                {**tx, "blockTime": bt.get("result")}
                for tx, bt in zip(transactions.get("result", []), block_times)
            ]

        except ChainAPIError as e:
            self._log_error(f"Solana API error: {str(e)}")
            raise
        except Exception as e:
            self._log_error(f"Unexpected error in Solana API: {str(e)}")
            raise ChainAPIError("Solana", f"Unexpected error: {str(e)}")

    async def analyze_transactions(self, transactions: List[Dict[str, Any]], blockchain: str, batch_size: int = 10) -> pd.DataFrame:
        if not transactions:
            return pd.DataFrame()

        if blockchain not in self.parsers:
            self._log_error(f"No parser available for blockchain: {blockchain}")
            return pd.DataFrame()

        async def process_batch(batch):
            return [
                self.parsers.get(blockchain, lambda x: x)(tx)
                for tx in batch
                if isinstance(tx, dict)
            ]

        batches = [transactions[i:i + batch_size] for i in range(0, len(transactions), batch_size)]
        tasks = [process_batch(batch) for batch in batches]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        parsed_transactions = [tx for batch in results for tx in batch if tx]
        df = pd.DataFrame(parsed_transactions)
        
        if not df.empty:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['blockchain'] = blockchain
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            df['time_diff'] = df['timestamp'].diff()
            df['tx_velocity'] = df['time_diff'].dt.total_seconds().apply(lambda x: 1/x if x > 0 else 0)
            
            self.performance_metrics['analysis_durations'].append({
                'blockchain': blockchain,
                'tx_count': len(df),
                'duration': duration
            })

        return df

    async def execute_analysis(self, addresses: List[str]):
        try:
            start_time = time.time()
            
            api_status = await self.verify_api_keys()
            if not any(api_status.values()):
                raise APIError("No working APIs found")

            wallet_groups = self.validate_addresses(addresses)
            
            analysis_tasks = []
            for blockchain, wallets in wallet_groups.items():
                if not wallets:
                    continue
                task = asyncio.create_task(self.analyze_blockchain_group(blockchain, wallets))
                analysis_tasks.append(task)

            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            all_transactions = self.combine_analysis_results(results)
            
            report = await self.generate_detailed_report(all_transactions)
            
            self.record_analysis_metrics(start_time)
            
            return report

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise

    async def analyze_blockchain_group(self, blockchain: str, wallets: List[str]) -> pd.DataFrame:
        all_txs = []
        for wallet in wallets:
            txs = await self.fetch_transactions(wallet, blockchain)
            if txs:
                all_txs.extend(txs)
        df = await self.analyze_transactions(all_txs, blockchain)
        return df

    def combine_analysis_results(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        combined = pd.concat(results, ignore_index=True)
        return combined

    async def generate_detailed_report(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        report = {
            'summary': {},
            'risk_analysis': {},
            'pattern_analysis': {},
            'anomaly_detection': {},
            'network_analysis': {},
            'recommendations': []
        }

        report['summary'] = {
            'total_transactions': len(transactions),
            'total_volume': transactions['amount'].sum(),
            'unique_addresses': len(set(transactions['ownerAddress']) | 
                                  set(transactions['toAddress'])),
            'time_span': {
                'start': transactions['timestamp'].min(),
                'end': transactions['timestamp'].max()
            }
        }

        risk_metrics = self._calculate_risk_metrics(transactions)
        report['risk_analysis'] = risk_metrics

        patterns = self._analyze_patterns(transactions)
        report['pattern_analysis'] = patterns

        anomalies = self._detect_anomalies(transactions)
        report['anomaly_detection'] = anomalies

        network_metrics = self._analyze_network_metrics(transactions)
        report['network_analysis'] = network_metrics

        report['recommendations'] = self._generate_recommendations(
            risk_metrics, patterns, anomalies, network_metrics
        )

        return report

    def identify_suspicious_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_message("Identifying suspicious transactions...", "cyan")
        suspicious_df = pd.DataFrame()
        if 'blockchain' not in df.columns:
            self._log_error("missing_blockchain_column")
            return suspicious_df

        for blockchain in CONFIG.keys():
            blockchain_df = df[df["blockchain"] == blockchain]
            threshold = CONFIG[blockchain].get("threshold", 0)
            high_value = blockchain_df[blockchain_df["amount"] > threshold]
            if not high_value.empty:
                self._log_message(f"High value transactions detected on {blockchain}", "magenta")
                self._log_message(high_value[["hash", "timestamp", "ownerAddress", "toAddress", "amount"]].to_string(index=False), "yellow")
                suspicious_df = pd.concat([suspicious_df, high_value], ignore_index=True)
        return suspicious_df

    def identify_exchange_addresses(self, df: pd.DataFrame) -> Set[str]:
        self._log_message("Identifying exchange addresses...", "cyan")
        potential_exchanges = set()
        for blockchain in CONFIG.keys():
            blockchain_df = df[df["blockchain"] == blockchain]
            if blockchain_df.empty:
                continue
            address_counts = blockchain_df['toAddress'].value_counts()
            exchange_threshold = 50
            exchange_addresses = address_counts[address_counts > exchange_threshold].index.tolist()
            potential_exchanges.update(exchange_addresses)
        return potential_exchanges

    def identify_blacklist_addresses(self, df: pd.DataFrame) -> Set[str]:
        self._log_message("Identifying blacklist addresses...", "cyan")
        blacklist_candidates = set()
        if 'blockchain' not in df.columns:
            self._log_error("missing_blockchain_column")
            return blacklist_candidates

        for blockchain in CONFIG.keys():
            blockchain_df = df[df["blockchain"] == blockchain]
            if blockchain_df.empty:
                continue
            grouped = blockchain_df.groupby('toAddress')
            for address, group in grouped:
                unique_senders = group['ownerAddress'].nunique()
                if unique_senders > 100:
                    blacklist_candidates.add(address)
        return blacklist_candidates

    def update_known_exchanges(self, potential_exchanges: Set[str]) -> None:
        self._log_message("Updating known exchanges...", "cyan")
        new_exchanges = {}
        for exchange_address in potential_exchanges:
            blockchain = self.detect_blockchain_from_address(exchange_address)
            if blockchain != 'Unknown' and exchange_address not in self.known_exchanges.get(blockchain, []):
                self.known_exchanges.setdefault(blockchain, []).append(exchange_address)
                new_exchanges.setdefault(blockchain, []).append(exchange_address)

        if new_exchanges:
            self._log_message("Adding new exchanges...", "cyan")
            for bc, addresses in new_exchanges.items():
                self._log_message(f"{bc}: {', '.join(addresses)}", "cyan")
            save_json_file(self.known_exchanges, self.KNOWN_EXCHANGES_FILE)
        else:
            self._log_message("No new exchanges found.", "yellow")

    def update_blacklist_addresses(self, potential_blacklist: Set[str]) -> None:
        self._log_message("Updating blacklist addresses...", "cyan")
        new_blacklist = potential_blacklist - self.blacklist_addresses
        if new_blacklist:
            self._log_message("Adding new blacklist addresses...", "cyan")
            for address in new_blacklist:
                self._log_message(address, "red")
            self.blacklist_addresses.update(new_blacklist)
            save_json_file(list(self.blacklist_addresses), BLACKLIST_FILE)
        else:
            self._log_message("No new blacklist addresses found.", "yellow")

    def scan_for_suspicious_wallets(self, df: pd.DataFrame) -> Set[str]:
        self._log_message("Scanning for suspicious wallets...", "cyan")
        receiving_wallets = set(df["toAddress"].unique())

        known_exchange_wallets = set()
        for wallets in self.known_exchanges.values():
            known_exchange_wallets.update(wallets)
        receiving_wallets -= known_exchange_wallets
        receiving_wallets -= self.blacklist_addresses

        self._log_message("Identified suspicious wallets:", "magenta")
        if not receiving_wallets:
            self._log_message("No suspicious wallets found.", "yellow")
        else:
            self._log_message(", ".join(receiving_wallets), "red")

        suspicious_wallets_network = set()
        for _, row in df.iterrows():
            if row["toAddress"] in receiving_wallets:
                suspicious_wallets_network.add(row["ownerAddress"])

        self._log_message("Suspicious wallets network:", "magenta")
        if not suspicious_wallets_network:
            self._log_message("No connected suspicious wallets found.", "yellow")
        else:
            self._log_message(", ".join(suspicious_wallets_network), "red")

        return suspicious_wallets_network

    def calculate_probability_of_fraud(self, transactions: pd.DataFrame, suspicious_wallets: Optional[Set[str]] = None, detailed: bool = False) -> Union[float, Tuple[float, List[str]]]:
        try:
            if transactions.empty:
                return (0.0, []) if detailed else 0.0

            suspicious_wallets = suspicious_wallets or set()
            score = 0.0
            factors = []

            high_value_mask = transactions['amount'] > transactions['amount'].mean() * 2
            high_value_ratio = high_value_mask.mean()
            score += high_value_ratio * 20
            if high_value_ratio > 0.1:
                factors.append(f"High-value transactions: {high_value_ratio:.1%}")

            if len(transactions) >= 2:
                timestamps = pd.to_datetime(transactions['timestamp'])
                time_diffs = timestamps.diff().dt.total_seconds()
                rapid_txs = (time_diffs < 60).sum()
                rapid_ratio = rapid_txs / len(transactions)
                score += rapid_ratio * 25
                if rapid_ratio > 0.1:
                    factors.append(f"Rapid transactions: {rapid_ratio:.1%}")

            if suspicious_wallets:
                connected = set(transactions['toAddress']).intersection(suspicious_wallets)
                if connected:
                    connection_score = len(connected) / len(transactions['toAddress'].unique()) * 30
                    score += connection_score
                    factors.append(f"Suspicious connections: {len(connected)}")

            round_numbers = transactions['amount'].apply(
                lambda x: abs(x - round(x, -int(np.floor(np.log10(abs(x) + 1e-10))))) < 1e-10
            ).mean()
            score += round_numbers * 15
            if round_numbers > 0.3:
                factors.append(f"Round number transactions: {round_numbers:.1%}")

            final_score = min(100.0, max(0.0, score))
            
            return (final_score, factors) if detailed else final_score

        except Exception as e:
            self._log_error("error_fraud_probability_analysis", str(e))
            return (0.0, ["Analysis error"]) if detailed else 0.0

    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        risk_metrics = {
            'transaction_patterns': {},
            'temporal_risks': {},
            'network_risks': {},
            'amount_based_risks': {}
        }

        time_diffs = df['timestamp'].diff().dt.total_seconds()
        risk_metrics['transaction_patterns'] = {
            'rapid_transactions': sum(time_diffs < 60),
            'round_number_txs': sum(df['amount'].apply(lambda x: abs(x - round(x, 0)) < 1e-8)),
            'repetitive_amounts': df['amount'].value_counts().head(3).to_dict()
        }

        df['hour'] = df['timestamp'].dt.hour
        risk_metrics['temporal_risks'] = {
            'off_hours_ratio': len(df[(df['hour'] < 6) | (df['hour'] > 22)]) / len(df),
            'weekend_ratio': len(df[df['timestamp'].dt.dayofweek.isin([5, 6])]) / len(df),
            'burst_transactions': self._detect_transaction_bursts(df)
        }

        G = nx.from_pandas_edgelist(df, 'ownerAddress', 'toAddress', 'amount')
        risk_metrics['network_risks'] = {
            'avg_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else None,
            'clustering_coefficient': nx.average_clustering(G),
            'degree_centrality': nx.degree_centrality(G)
        }

        for blockchain in df['blockchain'].unique():
            threshold = CONFIG[blockchain].get('threshold', 0)
            blockchain_df = df[df['blockchain'] == blockchain]
            risk_metrics['amount_based_risks'][blockchain] = {
                'high_value_ratio': len(blockchain_df[blockchain_df['amount'] > threshold]) / len(blockchain_df),
                'avg_transaction_size': blockchain_df['amount'].mean(),
                'max_transaction': blockchain_df['amount'].max(),
                'amount_distribution': blockchain_df['amount'].describe().to_dict()
            }

        return risk_metrics

    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        patterns = {
            'transaction_patterns': self._analyze_transaction_patterns(df),
            'temporal_patterns': self._analyze_temporal_patterns(df), 
            'network_patterns': self._analyze_network_patterns(df),
            'amount_patterns': self._analyze_amount_patterns(df)
        }
        return patterns
        
    def _analyze_transaction_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def _analyze_network_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def _analyze_amount_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def _analyze_network_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        G = nx.from_pandas_edgelist(df, 'ownerAddress', 'toAddress', 'amount')
        
        metrics = {
            'centrality': {
                'degree': nx.degree_centrality(G),
                'betweenness': nx.betweenness_centrality(G),
                'eigenvector': nx.eigenvector_centrality_numpy(G),
                'pagerank': nx.pagerank(G)
            },
            'community': {
                'clustering': nx.average_clustering(G),
                'components': list(nx.strongly_connected_components(G)),
                'bridges': list(nx.bridges(G)) if nx.is_connected(G) else []
            },
            'flow': {
                'density': nx.density(G),
                'diameter': nx.diameter(G) if nx.is_connected(G) else None,
                'average_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else None
            }
        }
        
        return metrics

    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {
            'statistical': {},
            'temporal': {},
            'behavioral': {},
            'ml_based': {}
        }

        if len(df) < 10:
            results['status'] = 'insufficient_data'
            return results

        for col in ['amount', 'tx_velocity']:
            if (col in df.columns) and (df[col].std() != 0):
                z_scores = stats.zscore(df[col])
                results['statistical'][col] = {
                    'outliers': df.index[abs(z_scores) > 3].tolist(),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }

        df['hour'] = df['timestamp'].dt.hour
        hourly_counts = df.groupby('hour').size()
        if len(hourly_counts) > 1:
            hourly_z_scores = stats.zscore(hourly_counts)
            results['temporal']['unusual_hours'] = hourly_counts[abs(hourly_z_scores) > 2].to_dict()

        address_patterns = df.groupby('ownerAddress').agg({
            'amount': ['count', 'mean', 'std'],
            'toAddress': 'nunique'
        })
        
        if len(address_patterns) >= 10:
            clf = IsolationForest(contamination=0.1, random_state=42)
            address_patterns_normalized = StandardScaler().fit_transform(address_patterns)
            predictions = clf.fit_predict(address_patterns_normalized)
            results['behavioral']['suspicious_addresses'] = \
                address_patterns.index[predictions == -1].tolist()
        else:
            results['behavioral']['status'] = 'insufficient_samples'

        features = ['amount', 'tx_velocity']
        if all(f in df.columns for f in features):
            X = StandardScaler().fit_transform(df[features])
            
            if len(df) >= 5:
                try:
                    clustering = DBSCAN(
                        eps=0.5, 
                        min_samples=min(5, max(2, int(len(df) * 0.1)))
                    ).fit(X)
                    results['ml_based']['clusters'] = {
                        'normal': sum(clustering.labels_ != -1),
                        'anomalous': sum(clustering.labels_ == -1),
                        'anomalous_indices': df.index[clustering.labels_ == -1].tolist()
                    }
                except Exception as e:
                    results['ml_based']['error'] = str(e)
            else:
                results['ml_based']['status'] = 'insufficient_samples'

        return results

    def _generate_recommendations(self, risk_metrics: Dict, patterns: Dict, 
                                anomalies: Dict, network_metrics: Dict) -> List[str]:
        recommendations = []
        
        for blockchain, risks in risk_metrics.get('amount_based_risks', {}).items():
            if risks.get('high_value_ratio', 0) > 0.3:
                recommendations.append(
                    f"High concentration of large transactions on {blockchain}. "
                    "Consider implementing additional verification steps."
                )

        if patterns.get('cyclic'):
            recommendations.append(
                "Detected cyclic transaction patterns. "
                "Investigate possible wash trading activity."
            )

        if patterns.get('layering'):
            recommendations.append(
                "Complex layering patterns detected. "
                "Monitor for potential money laundering attempts."
            )

        if 'ml_based' in anomalies and anomalies['ml_based'].get('clusters', {}).get('anomalous', 0) > 0:
            recommendations.append(
                "Machine learning models detected unusual transaction patterns. "
                "Review flagged transactions for potential fraud."
            )

        if network_metrics.get('flow', {}).get('density', 0) < 0.1:
            recommendations.append(
                "Low network density detected. "
                "Possible indication of segregated transaction networks."
            )

        return recommendations

    async def sleep_func(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    def record_analysis_metrics(self, start_time: float) -> None:
        duration = time.time() - start_time
        self.performance_metrics['processing_times'].append(duration)

    async def analyze_address(self, address: str) -> Dict[str, Any]:
        try:
            blockchain = self.detect_blockchain_from_address(address)
            if blockchain == 'Unknown':
                raise ValueError(self.translate(
                    "Could not determine blockchain for address",
                    f"Could not determine blockchain for address: {address}"
                ))

            self._log_message(f"Analyzing address {address} on {blockchain}...", "cyan")

            transactions = await self.fetch_transactions(address, blockchain)
            if not transactions:
                return {
                    'address': address,
                    'blockchain': blockchain,
                    'transactions': [],
                    'risk_score': 0,
                    'fraud_probability': 0,
                    'suspicious': False,
                    'analysis_details': {'error': 'No transactions found'}
                }

            df = await self.analyze_transactions(transactions, blockchain)
            if df.empty:
                return {
                    'address': address,
                    'blockchain': blockchain,
                    'transactions': [],
                    'risk_score': 0,
                    'fraud_probability': 0,
                    'suspicious': False,
                    'analysis_details': {'error': 'No valid transactions to analyze'}
                }

            fraud_prob, factors = self.calculate_probability_of_fraud(df, detailed=True)
            
            is_suspicious = (
                fraud_prob > FRAUD_PROBABILITY_THRESHOLD or
                address in self.blacklist_addresses or
                any(address in exchanges for exchanges in self.known_exchanges.values())
            )

            stats = {
                'total_transactions': len(df),
                'total_volume': df['amount'].sum(),
                'avg_transaction_size': df['amount'].mean(),
                'max_transaction_size': df['amount'].max(),
                'unique_counterparties': len(set(df['toAddress'].unique()) | 
                                  set(df['ownerAddress'].unique())) - 1,
            }

            return json.loads(json.dumps({
                'address': address,
                'blockchain': blockchain,
                'transactions': df.to_dict('records'),
                'risk_score': fraud_prob,
                'fraud_probability': fraud_prob,
                'suspicious': is_suspicious,
                'analysis_details': {
                    'risk_factors': factors,
                    'statistics': stats,
                    'blacklisted': address in self.blacklist_addresses,
                    'known_exchange': any(address in exchanges for exchanges in self.known_exchanges.values())
                }
            }, cls=ChainJSONEncoder))

        except Exception as e:
            self._log_error("error_analyzing_address", address, str(e))
            return json.loads(json.dumps({
                'address': address,
                'error': str(e),
                'transactions': [],
                'risk_score': 0,
                'fraud_probability': 0,
                'suspicious': False,
                'analysis_details': {'error': str(e)}
            }, cls=ChainJSONEncoder))

    async def improved_analyze_address(self, address: str) -> Dict[str, Any]:
        basic_analysis = await self.analyze_address(address)
        
        if 'error' in basic_analysis:
            return basic_analysis
            
        df = pd.DataFrame(basic_analysis['transactions'])
        
        enhanced_analysis = {
            **basic_analysis,
            'risk_metrics': self._calculate_risk_metrics(df)
        }
        
        enhanced_analysis['risk_score'] = self._calculate_ml_risk_score(
            enhanced_analysis
        )
        
        return enhanced_analysis

    def _calculate_ml_risk_score(self, analysis: Dict[str, Any]) -> float:
        if self.risk_model is None:
            self._log_error("risk_model_not_loaded")
            return self._calculate_heuristic_risk_score(analysis)

        features = self._extract_risk_features(analysis)
        if not features:
            return self._calculate_heuristic_risk_score(analysis)
            
        try:
            risk_score = self.risk_model.predict_proba([features])[0][1] * 100
        except Exception as e:
            self._log_error("error_ml_risk_scoring", str(e))
            risk_score = self._calculate_heuristic_risk_score(analysis)
            
        return risk_score

    def _calculate_heuristic_risk_score(self, analysis: Dict[str, Any]) -> float:
        score = 0.0
        
        anomaly_scores = analysis.get('risk_metrics', {}).get('ml_patterns', {}).get('anomaly_scores', [])
        if anomaly_scores:
            score += 30.0 * (sum(1 for x in anomaly_scores if x == -1) / len(anomaly_scores))
            
        suspicious_sequences = analysis.get('suspicious_sequences', [])
        if suspicious_sequences:
            score += 20.0 * (len(suspicious_sequences) / analysis.get('summary', {}).get('total_transactions', 1))
            
        behavior = analysis.get('wallet_behavior', {})
        risk_indicators = behavior.get('risk_indicators', {})
        if risk_indicators:
            score += 25.0 * max(risk_indicators.values())
            
        centrality = behavior.get('network_influence', {}).get('centrality', {})
        if centrality:
            score += 25.0 * max(centrality.values())
            
        return min(100.0, score)

    def _extract_risk_features(self, analysis: Dict[str, Any]) -> Optional[List[float]]:
        features = []
        risk_metrics = analysis.get('risk_analysis', {})
        pattern_analysis = analysis.get('pattern_analysis', {})
        anomaly_detection = analysis.get('anomaly_detection', {})
        network_analysis = analysis.get('network_analysis', {})

        try:
            features.append(risk_metrics.get('transaction_patterns', {}).get('rapid_transactions', 0))
            features.append(risk_metrics.get('transaction_patterns', {}).get('round_number_txs', 0))
            features.append(len(risk_metrics.get('transaction_patterns', {}).get('repetitive_amounts', {})))
            features.append(risk_metrics.get('temporal_risks', {}).get('off_hours_ratio', 0))
            features.append(risk_metrics.get('temporal_risks', {}).get('weekend_ratio', 0))
            features.append(len(risk_metrics.get('temporal_risks', {}).get('burst_transactions', [])))
            features.append(network_analysis.get('centrality', {}).get('degree', {}).get(max(network_analysis.get('centrality', {}).get('degree', {}).keys(), key=lambda k: network_analysis.get('centrality', {}).get('degree', {}).get(k, 0)), 0))

            return features
        except Exception as e:
            self._log_error("error_extracting_risk_features", str(e))
            return None

    def _detect_transaction_bursts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if df.empty or 'timestamp' not in df.columns:
            return []
            
        bursts = []
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff()
        burst_threshold = timedelta(minutes=5)
        min_transactions = 3
        
        current_burst = []
        for idx, diff in enumerate(time_diffs):
            if pd.isna(diff) or diff <= burst_threshold:
                current_burst.append(idx)
            else:
                if len(current_burst) >= min_transactions:
                    burst_txs = df_sorted.iloc[current_burst]
                    bursts.append({
                        'start_time': burst_txs['timestamp'].min(),
                        'end_time': burst_txs['timestamp'].max(),
                        'transaction_count': len(burst_txs),
                        'total_amount': burst_txs['amount'].sum(),
                        'addresses_involved': list(set(burst_txs['ownerAddress']) | 
                                                set(burst_txs['toAddress']))
                    })
                current_burst = [idx]
                
        return bursts

    def _train_risk_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, 'models/random_forest_risk_model.joblib')
            logger.info("Risk model trained and saved successfully.")
        except Exception as e:
            logger.error(f"Error training risk model: {str(e)}")

    async def verify_api_keys(self) -> Dict[str, bool]:
        api_status = {}
        
        try:
            for chain in self.config.keys():
                validation_endpoint = get_validation_endpoint(chain)
                if not validation_endpoint:
                    self._log_message(f"No validation endpoint configured for {chain}", "yellow")
                    api_status[chain] = False
                    continue

                chain_config = get_chain_config(chain)
                api_endpoint = get_api_endpoint(chain, 'transaction')
                
                try:
                    status = await self._test_api_endpoint(
                        validation_endpoint,
                        chain_config,
                        api_endpoint
                    )
                    api_status[chain] = status
                    
                    msg = f"API validation - {chain}: {'Success' if status else 'Failed'}"
                    self._log_message(msg, "green" if status else "red")
                    
                except Exception as e:
                    self._log_error(f"API validation failed for {chain}: {str(e)}")
                    api_status[chain] = False

            working_apis = sum(1 for status in api_status.values() if status)
            total_apis = len(api_status)
            
            self._log_message(f"API Validation Summary: {working_apis}/{total_apis} APIs operational", 
                            "green" if working_apis > 0 else "red")

            return api_status

        except Exception as e:
            self._log_error(f"API validation process failed: {str(e)}")
            return api_status

    async def _test_api_endpoint(self, validation_endpoint: str, chain_config: dict, api_endpoint: dict) -> bool:
        if not validation_endpoint:
            return False
            
        try:
            headers = self._prepare_headers(chain_config, api_endpoint)
            timeout = aiohttp.ClientTimeout(total=10)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(validation_endpoint, headers=headers) as response:
                    if response.status == 200:
                        await response.read()
                        return True
                    else:
                        self._log_error(
                            f"API validation failed with status {response.status}: {validation_endpoint}")
                        return False

        except aiohttp.ClientError as e:
            self._log_error(f"Network error testing endpoint {validation_endpoint}: {str(e)}")
            return False
        except asyncio.TimeoutError:
            self._log_error(f"Timeout testing endpoint {validation_endpoint}")
            return False
        except Exception as e:
            self._log_error(f"Error testing endpoint {validation_endpoint}: {str(e)}")
            return False

    def _prepare_headers(self, chain_config: dict, api_endpoint: dict) -> dict:
        headers = {
            'User-Agent': 'ChainWatch-Analyzer/1.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        }
        
        try:
            if not api_endpoint or 'headers' not in api_endpoint:
                return headers

            for header in api_endpoint['headers']:
                if header.startswith('TRON-PRO-API-KEY'):
                    headers[header] = self._get_tron_api_key()
                elif header.startswith('X-API-Key'):
                    api_key = chain_config.get('api_key')
                    if not api_key:
                        self._log_error("missing_api_key", chain_config.get('name', 'Unknown'))
                    headers[header] = api_key
                elif header.startswith('project_id'):
                    headers[header] = chain_config.get('project_id')
                elif header.startswith('Content-Type'):
                    headers['Content-Type'] = 'application/json'

            return headers

        except Exception as e:
            self._log_error(f"Error preparing headers: {str(e)}")
            return headers

    def _load_risk_model(self) -> Optional[RandomForestClassifier]:
        try:
            if os.path.exists(self.RISK_MODEL_PATH):
                model = joblib.load(self.RISK_MODEL_PATH)
                self._log_message("Risk model loaded successfully", "green")
                return model
            else:
                self._log_message("Risk model file not found", "yellow")
                return None
        except Exception as e:
            self._log_error(f"Error loading risk model: {str(e)}")
            return None

    def get_error_stats(self) -> Dict[str, Any]:
        return {
            'counts': dict(self.error_counts),
            'recent': list(self.last_errors)
        }

class AsyncLimiter:
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.timestamps = deque(maxlen=calls)
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        async with self._lock:
            now = time.monotonic()
            while len(self.timestamps) >= self.calls:
                if now - self.timestamps[0] > self.period:
                    self.timestamps.popleft()
                else:
                    sleep_time = self.timestamps[0] + self.period - now
                    await asyncio.sleep(sleep_time)
                    now = time.monotonic()
            self.timestamps.append(now)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass



