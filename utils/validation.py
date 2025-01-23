from typing import Dict, Any, List, Callable, Optional 
from dataclasses import dataclass
from enum import Enum
import re

class ValidationLevel(Enum):
    STRICT = "strict"
    LENIENT = "lenient" 
    NONE = "none"

@dataclass
class ValidationRule:
    field: str
    validator: Callable
    message: str
    level: ValidationLevel = ValidationLevel.STRICT

class DataValidator:
    def __init__(self):
        self.rules: List[ValidationRule] = []

    def add_rule(self, rule: ValidationRule) -> None:
        self.rules.append(rule)

    def validate(self, data: Dict[str, Any], level: ValidationLevel = ValidationLevel.STRICT) -> List[str]:
        errors = []
        for rule in self.rules:
            if rule.level.value > level.value:
                continue
            if rule.field in data:
                try:
                    if not rule.validator(data[rule.field]):
                        errors.append(rule.message)
                except Exception:
                    errors.append(f"Validation error for {rule.field}")
        return errors

class BlockchainAddressValidator:
    PATTERNS = {
        'ETH': r'^0x[a-fA-F0-9]{40}$',
        'BTC': r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', 
        'TRX': r'^T[a-zA-Z0-9]{33}$',
        'SOL': r'^[1-9A-HJ-NP-Za-km-z]{32,44}$',
        'ALGO': r'^[A-Z2-7]{58}$',
        'DOT': r'^[1-9A-HJ-NP-Za-km-z]{47,48}$',
        'AVAX': r'^0x[a-fA-F0-9]{40}$',
        'MATIC': r'^0x[a-fA-F0-9]{40}$',
        'FTM': r'^0x[a-fA-F0-9]{40}$',
        'ATOM': r'^cosmos[0-9a-z]{39}$',
        'NEAR': r'^[0-9a-z\.\-\_]{2,64}$',
        'ONE': r'^one[0-9a-z]{38}$',
        'ADA': r'^addr1[0-9a-z]{98}$',
        'XTZ': r'^tz[1-3][0-9a-zA-Z]{33}$'
    }

    @classmethod
    def validate_address(cls, address: str, chain: Optional[str] = None) -> bool:
        if not address:
            return False
            
        if chain:
            pattern = cls.PATTERNS.get(chain.upper())
            if pattern and re.match(pattern, address):
                return True
            return False

        return any(re.match(pattern, address) 
                  for pattern in cls.PATTERNS.values())

    @classmethod
    def get_chain(cls, address: str) -> Optional[str]:
        for chain, pattern in cls.PATTERNS.items():
            if re.match(pattern, address):
                return chain
        return None

    @classmethod
    def get_validation_rule(cls, chain: str) -> Optional[ValidationRule]:
        if chain.upper() in cls.PATTERNS:
            return ValidationRule(
                field='address',
                validator=lambda addr: cls.validate_address(addr, chain),
                message=f'Invalid {chain} address format',
                level=ValidationLevel.STRICT
            )
        return None
