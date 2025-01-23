from typing import Dict
from enum import Enum

class ErrorLevel(Enum):
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

def get_message(key: str, lang: str = 'en') -> str:
    return ANALYSIS_MESSAGES.get(lang, {}).get(key, key)

ANALYSIS_MESSAGES: Dict[str, Dict[str, Dict[str, str]]] = {
    'en': {
        'start_analysis': 'Starting transaction analysis...',
        'api_error': 'API request error: {}',
        'tx_parse_error': 'Error parsing transaction: {}',
        'no_transactions': 'No transactions were analyzable',
        'high_value_tx': 'High-value transaction detected: {}',
        'suspicious_pattern': 'Suspicious pattern detected: {}',
        'risk_alert': 'High risk activity detected: {}',
        'ml_analysis': 'Running machine learning analysis...',
        'pattern_detected': 'Pattern detected: {} (confidence: {}%)',
        'network_alert': 'Network anomaly detected: {}',
        'wallet_risk': 'Wallet risk score: {} ({} level)',
        'sequence_alert': 'Suspicious transaction sequence found: {}',
        'rate_limit_exceeded': 'API rate limit exceeded. Waiting...',
        'cache_miss': 'Cache miss for {}',
        'level': ErrorLevel.INFO.value
    },
    'fa': {
        'start_analysis': 'شروع تحلیل تراکنش‌ها...',
        'api_error': 'خطا در درخواست API: {}',
        'tx_parse_error': 'خطا در پردازش تراکنش: {}', 
        'no_transactions': 'هیچ تراکنشی قابل تحلیل نبود',
        'high_value_tx': 'تراکنش با ارزش بالا شناسایی شد: {}',
        'suspicious_pattern': 'الگوی مشکوک شناسایی شد: {}',
        'risk_alert': 'فعالیت پرریسک شناسایی شد: {}',
        'ml_analysis': 'در حال اجرای تحلیل یادگیری ماشین...',
        'pattern_detected': 'الگو شناسایی شد: {} (اطمینان: {}٪)',
        'network_alert': 'ناهنجاری شبکه شناسایی شد: {}',
        'wallet_risk': 'امتیاز ریسک کیف پول: {} (سطح {})',
        'sequence_alert': 'توالی تراکنش مشکوک یافت شد: {}',
        'rate_limit_exceeded': 'محدودیت نرخ API تجاوز شد. در حال انتظار...',
        'cache_miss': 'از دست رفتن کش برای {}',
        'level': ErrorLevel.INFO.value
    }
}

ERROR_MESSAGES: Dict[str, Dict[str, str]] = {
    'en': {
        'tron_api_key_error': 'TRON API key error: {}',
        'api_error': '{} API error: {}',
        'unexpected_error': 'Unexpected error in {} API: {}',
        'missing_address_or_blockchain': 'Wallet address and blockchain must be specified',
        'missing_blockchain_column': 'Blockchain column missing in transaction data',
        'error_parsing_tron_tx': 'Error parsing TRON transaction: {}',
        'error_fraud_probability_analysis': 'Error in fraud probability analysis: {}',
        'error_ml_risk_scoring': 'Error in ML risk scoring: {}',
        'error_loading_risk_model': 'Error loading risk model: {}',
        'error_extracting_risk_features': 'Error extracting risk features: {}',
        'risk_model_not_loaded': 'Risk model not loaded - using heuristic scoring',
        'blockchain_type_not_determined': 'Could not determine blockchain type for address: {}',
        'invalid_address_format': 'Invalid address format: {}',
        'api_validation_failed': 'API validation failed: {}',
        'api_connection_failed': 'API connection failed for {}: {}',
        'invalid_language': 'Invalid language selection. Use "en" or "fa".'
    },
    'fa': {
        'tron_api_key_error': 'خطای کلید API ترون: {}',
        'api_connection': 'خطا در اتصال به API',
        'invalid_address': 'فرمت آدرس کیف پول نامعتبر است',
        'timeout': 'زمان درخواست به پایان رسید',
        'rate_limit': 'محدودیت تعداد درخواست API',
        'analysis_error': 'تحلیل با خطا مواجه شد: {}',
        'ml_error': 'تحلیل یادگیری ماشین با خطا مواجه شد: {}',
        'network_error': 'تحلیل شبکه با خطا مواجه شد: {}',
        'invalid_language': 'انتخاب زبان نامعتبر است. از "en" یا "fa" استفاده کنید.',
        'missing_blockchain_column': 'ستون blockchain در داده‌ها موجود نیست',
        'risk_model_not_loaded': 'مدل ریسک بارگذاری نشد',
        'error_ml_risk_scoring': 'خطا در امتیازدهی ریسک ML: {}',
        'error_extracting_risk_features': 'خطا در استخراج ویژگی‌های ریسک: {}', 
        'error_parsing_tron_tx': 'خطا در تجزیه تراکنش ترون: {}',
        'blockchain_type_not_determined': 'نوع بلاکچین برای آدرس {} قابل تشخیص نیست',
        'invalid_address_format': 'فرمت آدرس نامعتبر است: {}',
        'api_validation_failed': 'اعتبارسنجی API با شکست مواجه شد: {}',
        'api_connection_failed': 'اتصال API برای {} با شکست مواجه شد: {}',
        'invalid_language': 'انتخاب زبان نامعتبر است. از "en" یا "fa" استفاده کنید.'
    }
}

RISK_LEVELS: Dict[str, Dict[str, str]] = {
    'en': {
        'low': 'Low',
        'medium': 'Medium',
        'high': 'High',
        'critical': 'Critical'
    },
    'fa': {
        'low': 'کم',
        'medium': 'متوسط',
        'high': 'بالا',
        'critical': 'بحرانی'
    }
}
