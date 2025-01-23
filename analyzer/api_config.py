import os
from dotenv import load_dotenv
from enum import Enum
from typing import Dict, Any, Optional, List, Set, Union, Tuple

load_dotenv()

__all__ = [
    'get_chain_config',
    'get_api_endpoint',
    'get_rate_limit',
    'get_validation_endpoint',
    'CONFIG',
    'API_ENDPOINTS',
    'UNIT_CONVERSIONS',
    'RATE_LIMITS',
    'APIErrorHandler',
    'API_RETRY_CONFIG',
    'APIErrorType'
]

CONFIG = {
    "Tron": {
        "api_keys": [
            os.getenv("TRON_API_KEY_1"),
            os.getenv("TRON_API_KEY_2"),
            os.getenv("TRON_API_KEY_3")
        ],
        "base_url": "https://apilist.tronscan.org/api",
        "threshold": 5000000,
        "unit": "SUN"
    },
    "Ethereum": {
        "etherscan_api_key": os.getenv("ETHERSCAN_API_KEY"),
        "blockcypher_token": os.getenv("BLOCKCYPHER_TOKEN"),
        "base_url": "https://api.blockcypher.com/v1/eth/main",
        "threshold": 10,
        "unit": "ETH"
    },
    "Algorand": {
        "algod_api_key": os.getenv("ALGOD_API_KEY"),
        "indexer_api_key": os.getenv("INDEXER_API_KEY"),
        "base_url": "https://mainnet-api.algonode.cloud",
        "threshold": 1000000,
        "unit": "ALGO"
    },
    "Fantom": {
        "ftmscan_api_key": os.getenv("FTMSCAN_API_KEY"),
        "base_url": "https://api.ftmscan.com/api",
        "threshold": 10,
        "unit": "FTM"
    },
    "Cosmos": {
        "cosmos_api_key": os.getenv("COSMOS_API_KEY"),
        "base_url": "https://rest.cosmos.directory/cosmoshub",
        "threshold": 1000000,
        "unit": "ATOM"
    },
    "Elrond": {
        "elrond_api_key": os.getenv("ELROND_API_KEY"),
        "base_url": "https://api.elrond.com",
        "threshold": 10,
        "unit": "EGLD"
    },
    "Linea": {
        "infura_url": os.getenv("LINEA_INFURA_URL"),
        "infura_api_key": os.getenv("INFURA_API_KEY"),
        "threshold": 10,
        "unit": "ETH",
        "base_url": 'https://linea-mainnet.infura.io/v3',
        "headers": {
            'Authorization': f'Bearer {os.getenv("INFURA_API_KEY")}'
        }
    },
    "Avalanche": {
        "snowtrace_api_key": os.getenv("SNOWTRACE_API_KEY"),
        "base_url": "https://api.snowtrace.io/api",
        "threshold": 10,
        "unit": "AVAX"
    },
    "Cardano": {
        "blockfrost_api_key": os.getenv("BLOCKFROST_API_KEY"),
        "base_url": "https://cardano-mainnet.blockfrost.io/api/v0",
        "project_id": os.getenv("BLOCKFROST_PROJECT_ID"),
        "headers": {
            'project_id': os.getenv("BLOCKFROST_PROJECT_ID")
        },
        "threshold": 1000000,
        "unit": "ADA"
    },
    "Litecoin": {
        "blockcypher_token": os.getenv("BLOCKCYPHER_TOKEN"),
        "base_url": "https://api.blockcypher.com/v1/ltc/main",
        "threshold": 10,
        "unit": "LTC"
    },
    "Near": {
        "base_url": "https://rpc.mainnet.near.org",
        "threshold": 10,
        "unit": "NEAR"
    },
    "Harmony": {
        "base_url": "https://api.s0.t.hmny.io",
        "threshold": 10,
        "unit": "ONE"
    },
    "Monero": {
        "base_url": "https://xmr-node.cakewallet.com:18081",
        "threshold": 10,
        "unit": "XMR"
    },
    "Polkadot": {
        "subscan_api_key": os.getenv("SUBSCAN_API_KEY"),
        "base_url": "https://polkadot.api.subscan.io/api/scan",
        "threshold": 10,
        "unit": "DOT"
    },
    "Kusama": {
        "subscan_api_key": os.getenv("SUBSCAN_API_KEY"),
        "base_url": "https://kusama.api.subscan.io/api/scan",
        "threshold": 10,
        "unit": "KSM"
    },
    "Flow": {
        "blocto_api_key": os.getenv("BLOCTO_API_KEY"),
        "base_url": "https://flow-api-mainnet.blocto.app",
        "threshold": 10,
        "unit": "FLOW"
    },
    "Hedera": {
        "base_url": "https://mainnet-public.mirrornode.hedera.com/api/v1",
        "threshold": 10,
        "unit": "HBAR"
    },
    "IOTA": {
        "base_url": "https://api.iota-mainnet.org/api/v1",
        "api_key": os.getenv("IOTA_API_KEY"),
        "threshold": 1000000,
        "unit": "MIOTA",
        "headers": {
            'Authorization': f'Bearer {os.getenv("IOTA_API_KEY")}'
        }
    },
    "XRP": {
        "base_url": "https://data.ripple.com/v2",
        "threshold": 10,
        "unit": "XRP"
    },
    "WAVES": {
        "base_url": "https://nodes.wavesnodes.com",
        "threshold": 10,
        "unit": "WAVES"
    },
    "ICON": {
        "base_url": "https://tracker.icon.foundation/v3",
        "threshold": 10,
        "unit": "ICX"
    },
    "THETA": {
        "base_url": 'https://api.thetanetwork.org/v1',
        "api_key": os.getenv("THETA_API_KEY"),
        "threshold": 10,
        "unit": "THETA",
        "headers": {
            'X-API-Key': os.getenv("THETA_API_KEY")
        }
    },
    "ZILLIQA": {
        "base_url": 'https://api.zilliqa.com',
        "api_key": os.getenv("VIEWBLOCK_API_KEY"),
        "threshold": 1000000,
        "unit": "ZIL",
        "headers": {
            'X-APIKEY': os.getenv("VIEWBLOCK_API_KEY")
        }
    },
    "TEZOS": {
        "base_url": "https://api.tzkt.io/v1",
        "threshold": 1000000,
        "unit": "XTZ"
    }
}

API_ENDPOINTS = {
    'Ethereum': {
        'transaction': {
            'url': 'https://api.etherscan.io/api',
            'method': 'GET',
            'params': {
                'module': 'account',
                'action': 'txlist',
                'startblock': 0,
                'endblock': 99999999,
                'sort': 'asc'
            }
        }
    },
    'Tron': {
        'transaction': {
            'url': 'https://apilist.tronscan.org/api/transaction',
            'method': 'GET',
            'headers': ['TRON-PRO-API-KEY']
        }
    },
    'BSC': {
        'transaction': {
            'url': 'https://api.bscscan.com/api',
            'method': 'GET',
            'params': {
                'module': 'account',
                'action': 'txlist',
                'startblock': 0,
                'endblock': 99999999,
                'sort': 'asc'
            }
        }
    },
    'Solana': {
        'transaction': {
            'url': 'https://api.mainnet-beta.solana.com',
            'method': 'POST',
            'headers': ['Content-Type: application/json'],
            'body': {
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'getConfirmedSignaturesForAddress2',
                'params': []
            }
        }
    },
    'Polygon': {
        'transaction': {
            'url': 'https://api.polygonscan.com/api',
            'method': 'GET',
            'params': {
                'module': 'account',
                'action': 'txlist',
                'startblock': 0,
                'endblock': 99999999,
                'sort': 'asc'
            }
        }
    },
    'Algorand': {
        'transaction': {
            'url': 'https://mainnet-api.algonode.cloud/v2/transactions',
            'method': 'GET'
        }
    },
    'Fantom': {
        'transaction': {
            'url': 'https://api.ftmscan.com/api',
            'method': 'GET',
            'params': {
                'module': 'account',
                'action': 'txlist',
                'startblock': 0,
                'endblock': 99999999,
                'sort': 'asc'
            }
        }
    },
    'Cosmos': {
        'transaction': {
            'url': 'https://rest.cosmos.directory/cosmoshub/cosmos/tx/v1beta1/txs',
            'method': 'GET',
            'params': {
                'events': 'message.sender=\'{address}\''
            }
        }
    },
    'Elrond': {
        'transaction': {
            'url': 'https://api.elrond.com/transactions',
            'method': 'GET',
            'params': {
                'sender': '{address}'
            }
        }
    },
    'Linea': {
        'transaction': {
            'url': f"https://linea-mainnet.infura.io/v3/{os.getenv('INFURA_API_KEY')}",
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {os.getenv("INFURA_API_KEY")}'
            },
            'body': {
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'eth_getTransactionByHash',
                'params': []
            }
        }
    },
    'Avalanche': {
        'transaction': {
            'url': 'https://api.snowtrace.io/api',
            'method': 'GET',
            'params': {
                'module': 'account',
                'action': 'txlist',
                'startblock': 0,
                'endblock': 99999999,
                'sort': 'asc'
            }
        }
    },
    'Cardano': {
        'transaction': {
            'url': 'https://cardano-mainnet.blockfrost.io/api/v0/addresses/{address}/transactions',
            'method': 'GET',
            'headers': {
                'project_id': os.getenv("BLOCKFROST_PROJECT_ID")
            }
        }
    },
    'Litecoin': {
        'transaction': {
            'url': 'https://api.blockcypher.com/v1/ltc/main',
            'method': 'GET',
            'params': {
                'token': os.getenv("BLOCKCYPHER_TOKEN")
            }
        }
    },
    'Near': {
        'transaction': {
            'url': 'https://rpc.mainnet.near.org',
            'method': 'POST',
            'headers': ['Content-Type: application/json'],
            'body': {
                'jsonrpc': '2.0',
                'id': 'dontcare',
                'method': 'tx',
                'params': []
            }
        }
    },
    'Harmony': {
        'transaction': {
            'url': 'https://api.s0.t.hmny.io',
            'method': 'POST',
            'headers': ['Content-Type: application/json'],
            'body': {
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'hmy_getTransactionsHistory',
                'params': []
            }
        }
    },
    'Monero': {
        'transaction': {
            'url': 'https://xmr-node.cakewallet.com:18081/json_rpc',
            'method': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'body': {
                'jsonrpc': '2.0',
                'id': '0',
                'method': 'get_transactions'
            }
        }
    },
    'Polkadot': {
        'transaction': {
            'url': 'https://polkadot.api.subscan.io/api/scan/transfers',
            'method': 'POST',
            'headers': ['Content-Type: application/json'],
            'body': {
                'address': '',
                'row': 20,
                'page': 0
            }
        }
    },
    'Kusama': {
        'transaction': {
            'url': 'https://kusama.api.subscan.io/api/scan/transfers',
            'method': 'POST',
            'headers': ['Content-Type: application/json'],
            'body': {
                'address': '',
                'row': 20,
                'page': 0
            }
        }
    },
    'Flow': {
        'transaction': {
            'url': 'https://flow-api-mainnet.blocto.app',
            'method': 'POST',
            'headers': ['Content-Type: application/json'],
            'body': {
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'flow_getAccount',
                'params': []
            }
        }
    },
    'Hedera': {
        'transaction': {
            'url': 'https://mainnet-public.mirrornode.hedera.com/api/v1/transactions',
            'method': 'GET'
        }
    },
    'IOTA': {
        'transaction': {
            'url': 'https://api.iota-mainnet.org/api/v1/transactions',
            'method': 'GET',
            'headers': {
                'Authorization': f'Bearer {os.getenv("IOTA_API_KEY")}'
            }
        }
    },
    'XRP': {
        'transaction': {
            'url': 'https://data.ripple.com/v2/accounts/{address}/transactions',
            'method': 'GET'
        }
    },
    'WAVES': {
        'transaction': {
            'url': 'https://nodes.wavesnodes.com/transactions/address/{address}/limit/100',
            'method': 'GET'
        }
    },
    'ICON': {
        'transaction': {
            'url': 'https://tracker.icon.foundation/v3/address/txList',
            'method': 'GET',
            'params': {
                'address': '',
                'page': 1,
                'count': 100
            }
        }
    },
    'THETA': {
        'transaction': {
            'url': 'https://api.thetanetwork.org/v1/transactions',
            'method': 'GET',
            'headers': {
                'X-API-Key': os.getenv("THETA_API_KEY")
            }
        }
    },
    'ZILLIQA': {
        'transaction': {
            'url': 'https://api.zilliqa.com/',
            'method': 'POST',
            'headers': {
                'X-APIKEY': os.getenv("VIEWBLOCK_API_KEY"),
                'Content-Type': 'application/json'
            },
            'body': {
                'id': '1',
                'jsonrpc': '2.0',
                'method': 'GetTransactionsForTxBlock',
                'params': []
            }
        }
    },
    'TEZOS': {
        'transaction': {
            'url': 'https://api.tzkt.io/v1/accounts/{address}/operations',
            'method': 'GET'
        }
    }
}

UNIT_CONVERSIONS = {
    'Ethereum': 1e18,
    'Tron': 1e6,
    'BSC': 1e18,
    'Solana': 1e9,
    'Polygon': 1e18,
    'Avalanche': 1e18,
    'Cardano': 1e6,
    'Litecoin': 1e8,
    'Near': 1e24,
    'Harmony': 1e18,
    'Monero': 1e12,
    'Polkadot': 1e10,
    'Kusama': 1e12,
    'Flow': 1e8,
    'Hedera': 1e8,
    'IOTA': 1e6,
    'XRP': 1e6,
    'WAVES': 1e8,
    'ICON': 1e18,
    'THETA': 1e18,
    'ZILLIQA': 1e12,
    'TEZOS': 1e6,
    'Algorand': 1e6,
    'Fantom': 1e18,
    'Cosmos': 1e6,
    'Elrond': 1e18,
    'Linea': 1e18
}

RATE_LIMITS = {
    'Ethereum': {'calls': 5, 'period': 1},
    'Tron': {'calls': 3, 'period': 1},
    'BSC': {'calls': 5, 'period': 1},
    'Solana': {'calls': 10, 'period': 1},
    'Polygon': {'calls': 5, 'period': 1},
    'Algorand': {'calls': 5, 'period': 1},
    'Fantom': {'calls': 5, 'period': 1},
    'Cosmos': {'calls': 5, 'period': 1},
    'Elrond': {'calls': 5, 'period': 1},
    'Linea': {'calls': 5, 'period': 1},
    'Avalanche': {'calls': 5, 'period': 1},
    'Cardano': {'calls': 5, 'period': 1},
    'Litecoin': {'calls': 5, 'period': 1},
    'Near': {'calls': 5, 'period': 1},
    'Harmony': {'calls': 5, 'period': 1},
    'Monero': {'calls': 5, 'period': 1},
    'Polkadot': {'calls': 5, 'period': 1},
    'Kusama': {'calls': 5, 'period': 1},
    'Flow': {'calls': 5, 'period': 1},
    'Hedera': {'calls': 5, 'period': 1},
    'IOTA': {'calls': 5, 'period': 1},
    'XRP': {'calls': 5, 'period': 1},
    'WAVES': {'calls': 5, 'period': 1},
    'ICON': {'calls': 5, 'period': 1},
    'THETA': {'calls': 5, 'period': 1},
    'ZILLIQA': {'calls': 5, 'period': 1},
    'TEZOS': {'calls': 5, 'period': 1}
}

class APIErrorType(Enum):
    RATE_LIMIT = 429
    INVALID_KEY = 403
    NOT_FOUND = 404
    SERVER_ERROR = 500
    PARSE_ERROR = 1001
    VALIDATION_ERROR = 1002
    CHAIN_ERROR = 1003
    TIMEOUT_ERROR = 1004

ERROR_CODES = {
    'rate_limit': APIErrorType.RATE_LIMIT.value,
    'invalid_key': APIErrorType.INVALID_KEY.value,
    'not_found': APIErrorType.NOT_FOUND.value, 
    'server_error': APIErrorType.SERVER_ERROR.value
}

VALIDATION_ENDPOINTS = {
    'Ethereum': 'https://api.etherscan.io/api?module=proxy&action=eth_blockNumber&apikey={api_key}',
    'Tron': 'https://apilist.tronscan.org/api/system/status',
    'BSC': 'https://api.bscscan.com/api?module=proxy&action=eth_blockNumber&apikey={api_key}',
    'Solana': 'https://api.mainnet-beta.solana.com',
    'Polygon': 'https://api.polygonscan.com/api?module=proxy&action=eth_blockNumber',
    'Avalanche': 'https://api.snowtrace.io/api?module=proxy&action=eth_blockNumber',
    'Cardano': 'https://cardano-mainnet.blockfrost.io/api/v0/network',
    'Litecoin': 'https://api.blockcypher.com/v1/ltc/main',
    'Near': 'https://rpc.mainnet.near.org/status',
    'Harmony': 'https://api.s0.t.hmny.io',
    'Monero': 'https://xmr-node.cakewallet.com:18081/json_rpc',
    'Polkadot': 'https://polkadot.api.subscan.io/api/scan/metadata',
    'Kusama': 'https://kusama.api.subscan.io/api/scan/metadata',
    'Flow': 'https://rest-mainnet.onflow.org/v1/network/parameters',
    'Hedera': 'https://mainnet-public.mirrornode.hedera.com/api/v1/network/nodes',
    'IOTA': 'https://api.iota-mainnet.org/api/v1/info',
    'XRP': 'https://xrplcluster.com/',
    'WAVES': 'https://nodes.wavesnodes.com/blocks/height',
    'ICON': 'https://tracker.icon.foundation/v3/status',
    'THETA': 'https://api.thetanetwork.org/v1/status',
    'ZILLIQA': 'https://api.zilliqa.com/',
    'TEZOS': 'https://api.tzkt.io/v1/head',
    'Algorand': 'https://mainnet-api.algonode.cloud/v2/status',
    'Fantom': 'https://api.ftmscan.com/api?module=proxy&action=eth_blockNumber',
    'Cosmos': 'https://rest.cosmos.directory/cosmoshub/cosmos/base/tendermint/v1beta1/node_info',
    'Elrond': 'https://gateway.multiversx.com/network/config',
    'Linea': f"https://linea-mainnet.infura.io/v3/{os.getenv('INFURA_API_KEY')}",
}

API_ERROR_CODES = {
    'Ethereum': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Tron': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Avalanche': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Cardano': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Litecoin': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Near': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Harmony': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Monero': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Polkadot': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Kusama': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Flow': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Hedera': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'IOTA': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'XRP': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'WAVES': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'ICON': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'THETA': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'ZILLIQA': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'TEZOS': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Algorand': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Fantom': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Cosmos': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Elrond': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    },
    'Linea': {
        'rate_limit': [429],
        'invalid_key': [403, 401],
        'not_found': [404],
        'server_error': range(500, 600)
    }
}

API_RETRY_CONFIG = {
    'max_retries': 3,
    'base_delay': 1,
    'max_delay': 30,
    'exponential_base': 2,
    'jitter': 0.1,
    'timeout': 15
}

API_RESPONSE_SCHEMAS = {
    'Ethereum': {
        'required_fields': ['result', 'status'],
        'error_field': 'message',
        'success_status': '1'
    },
    'Tron': {
        'required_fields': ['data'],
        'error_field': 'error',
        'success_status': None
    },
    'Algorand': {
        'required_fields': ['version'],
        'error_field': 'message',
        'success_status': None
    },
    'Fantom': {
        'required_fields': ['result'],
        'error_field': 'error',
        'success_status': '1'
    },
    'Cosmos': {
        'required_fields': ['node_info'],
        'error_field': 'error',
        'success_status': None
    },
    'Elrond': {
        'required_fields': ['status'],
        'error_field': 'error',
        'success_status': None
    },
    'Linea': {
        'required_fields': ['result'],
        'error_field': 'error',
        'success_status': None
    },
    'THETA': {
        'required_fields': ['status'],
        'error_field': 'error',
        'success_status': 'ok'
    },
    'ZILLIQA': {
        'required_fields': ['result'],
        'error_field': 'error',
        'success_status': None
    },
    'IOTA': {
        'required_fields': ['nodeInfo'],
        'error_field': 'error',
        'success_status': None
    },
    'Monero': {
        'required_fields': ['result'],
        'error_field': 'error',
        'success_status': None
    },
    'Cardano': {
        'required_fields': ['network'],
        'error_field': 'error_message',
        'success_status': None
    }
}

BLOCKCHAIN_ERROR_CODES = {
    'Tron': {
        'ADDRESS_NOT_FOUND': '1001',
        'INVALID_ADDRESS': '1002', 
        'NETWORK_ERROR': '1003'
    },
    'Ethereum': {
        'INVALID_ADDRESS': 'Invalid address',
        'CONTRACT_ERROR': 'Execution reverted',
        'NONCE_ERROR': 'Nonce too low'
    }
}

class APIErrorHandler:
    @staticmethod 
    def is_rate_limit_error(chain: str, status_code: int) -> bool:
        if (status_code == APIErrorType.RATE_LIMIT.value):
            return True
        if chain in API_ERROR_CODES:
            rate_limits = API_ERROR_CODES[chain].get('rate_limit', [])
            return status_code in rate_limits
        return False

    @staticmethod
    def is_invalid_key_error(chain: str, status_code: int) -> bool:
        if status_code in (401, 403):
            return True
        return status_code in API_ERROR_CODES.get(chain, {}).get('invalid_key', [])

    @staticmethod
    def is_not_found_error(chain: str, status_code: int) -> bool:
        if status_code == 404:
            return True
        return status_code in API_ERROR_CODES.get(chain, {}).get('not_found', [])

    @staticmethod
    def is_server_error(chain: str, status_code: int) -> bool:
        if 500 <= status_code < 600:
            return True
        return status_code in API_ERROR_CODES.get(chain, {}).get('server_error', [])

    @staticmethod
    def validate_response(chain: str, response_data: dict) -> bool:
        try:
            schema = API_RESPONSE_SCHEMAS.get(chain)
            if not schema:
                return True
            if not isinstance(response_data, dict):
                return False
            for field in schema['required_fields']:
                if field not in response_data:
                    return False
            if schema['success_status'] is not None:
                status = str(response_data.get('status', ''))
                return status == schema['success_status']
            return True
        except Exception:
            return False

    @staticmethod
    def get_error_message(chain: str, response_data: dict) -> str:
        schema = API_RESPONSE_SCHEMAS.get(chain)
        if not schema:
            return "Unknown error"
        error_field = schema['error_field']
        return str(response_data.get(error_field, "Unknown error"))

    @staticmethod
    def get_error_code(chain: str, error_msg: str) -> Optional[str]:
        codes = BLOCKCHAIN_ERROR_CODES.get(chain, {})
        for code, msg in codes.items():
            if msg.lower() in error_msg.lower():
                return code
        return None

def validate_config() -> bool:
    try:
        for chain, config in CONFIG.items():
            if not isinstance(config, dict):
                print(f"Warning: Invalid config type for {chain}")
                continue
            if 'base_url' not in config:
                print(f"Warning: Missing base_url for {chain}, using default")
                CONFIG[chain]['base_url'] = f"https://api.{chain.lower()}.org"
            if 'threshold' not in config:
                print(f"Warning: Missing threshold for {chain}, using default")
                CONFIG[chain]['threshold'] = 10
            if 'unit' not in config:
                print(f"Warning: Missing unit for {chain}, using default")
                CONFIG[chain]['unit'] = chain.upper()
        for chain in CONFIG.keys():
            if chain not in API_ENDPOINTS:
                print(f"Warning: Missing API endpoint for {chain}, adding default")
                API_ENDPOINTS[chain] = {
                    'transaction': {
                        'url': CONFIG[chain]['base_url'],
                        'method': 'GET'
                    }
                }
        for chain in CONFIG.keys():
            if chain not in RATE_LIMITS:
                print(f"Warning: Missing rate limit for {chain}, adding default")
                RATE_LIMITS[chain] = {'calls': 5, 'period': 1}
        return True
    except Exception as e:
        print(f"Configuration validation warning: {str(e)}")
        return False

if not validate_config():
    print("Warning: Some configuration validation failed, using defaults where necessary")

class ConfigValidator:
    @staticmethod
    def validate_chain_config(chain: str, config: dict) -> bool:
        required_fields = {'base_url', 'threshold', 'unit'}
        return all(field in config for field in required_fields)

    @staticmethod
    def validate_api_endpoint(endpoint: dict) -> bool:
        required_fields = {'url', 'method'}
        return all(field in endpoint for field in required_fields)

    @staticmethod
    def validate_rate_limit(rate_limit: dict) -> bool:
        required_fields = {'calls', 'period'}
        return all(field in rate_limit for field in required_fields)

class APIEndpointBuilder:
    @staticmethod
    def build_url(endpoint: dict, **kwargs) -> str:
        url = endpoint['url']
        for key, value in kwargs.items():
            url = url.replace(f"{{{key}}}", str(value))
        return url

    @staticmethod
    def build_headers(endpoint: dict, config: dict) -> dict:
        headers = {
            'User-Agent': 'ChainWatch-Analyzer/1.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        }
        
        if 'headers' in endpoint:
            for header in endpoint['headers']:
                if header.startswith('TRON-PRO-API-KEY'):
                    headers[header] = config.get('api_keys', [None])[0]
                elif header.startswith('X-API-Key'):
                    headers[header] = config.get('api_key')
                elif header.startswith('project_id'):
                    headers[header] = config.get('project_id')
                elif header.startswith('Content-Type'):
                    headers['Content-Type'] = 'application/json'
        
        return headers

for chain, config in CONFIG.items():
    if not ConfigValidator.validate_chain_config(chain, config):
        raise ValueError(f"Invalid configuration for chain: {chain}")

for chain, endpoints in API_ENDPOINTS.items():
    for action, endpoint in endpoints.items():
        if not ConfigValidator.validate_api_endpoint(endpoint):
            raise ValueError(f"Invalid API endpoint configuration for {chain}.{action}")

for chain, rate_limit in RATE_LIMITS.items():
    if not ConfigValidator.validate_rate_limit(rate_limit):
        raise ValueError(f"Invalid rate limit configuration for chain: {chain}")

def get_chain_config(chain: str) -> dict:
    return CONFIG.get(chain, {})

def get_api_endpoint(chain: str, action: str) -> dict:
    return API_ENDPOINTS.get(chain, {}).get(action, {})

def get_unit_conversion(chain: str) -> float:
    return UNIT_CONVERSIONS.get(chain, 1.0)

def get_rate_limit(chain: str) -> dict:
    return RATE_LIMITS.get(chain, {'calls': 1, 'period': 1})

def get_validation_endpoint(chain: str) -> str:
    return VALIDATION_ENDPOINTS.get(chain, '')
