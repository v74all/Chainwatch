import argparse
import asyncio
import sys
from termcolor import colored
from typing import Optional, Any
import json
from analyzer.analyzer_core import ChainWatchAnalyzer
from models.modeling import EnhancedMLAnalysis
import signal
from importlib.metadata import version, PackageNotFoundError
import os
from tqdm import tqdm
from utils.messages import ERROR_MESSAGES

VERSION = "1.5.0-beta"
BUILD_DATE = "2025"

def check_dependencies():
    package_mapping = {
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'networkx': 'networkx',
        'scipy': 'scipy',
        'xgboost': 'xgboost',
        'statsmodels': 'statsmodels',
        'seaborn': 'seaborn',
        'ratelimit': 'ratelimit',
        'aiohttp': 'aiohttp',
        'python-dotenv': 'python-dotenv',
        'joblib': 'joblib',
        'psutil': 'psutil',
        'termcolor': 'termcolor'
    }
    
    missing = []
    for pkg, name in package_mapping.items():
        try:
            version(name)
        except PackageNotFoundError:
            missing.append(name)
    
    if missing:
        print(colored("\n❌ Missing Dependencies:", "red"))
        print(colored(", ".join(missing), "white"))
        print(colored("\nRun: pip install -r requirements.txt", "green"))
        sys.exit(1)

def display_banner(args):
    banner = """
██╗   ██╗███████╗██╗  ████████╗██╗  ██╗██████╗  ██████╗ ███╗   ██╗██╗   ██╗██╗  ██╗
██║   ██║╚════██║██║  ╚══██╔══╝██║  ██║██╔══██╗██╔═══██╗████╗  ██║╚██╗ ██╔╝╚██╗██╔╝
██║   ██║    ██╔╝██║     ██║   ███████║██████╔╝██║   ██║██╔██╗ ██║ ╚████╔╝  ╚███╔╝ 
╚██╗ ██╔╝   ██╔╝ ██║     ██║   ██╔══██║██╔══██╗██║   ██║██║╚██╗██║  ╚██╔╝   ██╔██╗ 
 ╚████╔╝    ██║  ███████╗██║   ██║  ██║██║  ██║╚██████╔╝██║ ╚████║   ██║   ██╔╝ ██╗
  ╚═══╝     ╚═╝  ╚══════╝╚═╝   ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝
    """
    print(colored(banner, "green"))
    print(colored(f"ChainWatch Analyzer v{VERSION} (Build: {BUILD_DATE})", "cyan"))
    if args.lang not in ['en', 'fa']:
        print(colored(ERROR_MESSAGES['en']['invalid_language'], 'red'))
        sys.exit(1)

async def verify_system_health(args) -> bool:
    print(colored("\n🔍 Running System Health Check...", "cyan"))
    
    all_checks_passed = True
    
    try:
        check_dependencies()
        print(colored("✓ All dependencies are installed", "green"))
    except Exception as e:
        print(colored(f"✗ Dependency check failed: {e}", "red"))
        all_checks_passed = False

    required_dirs = ['assets', 'models', 'data', 'cache']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
                print(colored(f"✓ Created missing directory: {dir_name}", "yellow"))
            except Exception as e:
                print(colored(f"✗ Cannot create directory {dir_name}: {e}", "red"))
                all_checks_passed = False
        else:
            print(colored(f"✓ Directory exists: {dir_name}", "green"))

    print(colored("\n3. Checking API Connectivity:", "cyan"))
    analyzer = ChainWatchAnalyzer(
        addresses=["0x0000000000000000000000000000000000000000"],
        log_callback=lambda msg, color: print(colored(msg, color)),
        language=args.lang
    )
    api_status = await analyzer.verify_api_keys()
    if any(api_status.values()):
        print(colored("✓ At least one blockchain API is accessible", "green"))
    else:
        print(colored("✗ No blockchain APIs are accessible", "red"))
        all_checks_passed = False

    try:
        ml_analyzer = EnhancedMLAnalysis()
        if ml_analyzer.run_self_test():
            print(colored("✓ ML system validated", "green"))
        else:
            print(colored("✗ ML system validation failed", "red"))
            all_checks_passed = False
    except Exception as e:
        print(colored(f"✗ ML system error: {e}", "red"))
        all_checks_passed = False

    print(colored("\n5. Checking Asset Generation:", "cyan"))
    from utils.create_assets import init_assets
    try:
        assets_status = init_assets()
        if all(assets_status.values()):
            print(colored("✓ All assets generated successfully", "green"))
        else:
            failed_assets = [k for k, v in assets_status.items() if not v]
            print(colored(f"✗ Some assets failed to generate: {failed_assets}", "red"))
            all_checks_passed = False
    except Exception as e:
        print(colored(f"✗ Asset generation failed: {e}", "red"))
        all_checks_passed = False

    print(colored("\n=== System Health Check Complete ===", "cyan"))
    if all_checks_passed:
        print(colored("✅ All systems operational", "green"))
    else:
        print(colored("⚠️ Some systems need attention", "yellow"))

    return all_checks_passed

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='''
ChainWatch Analyzer CLI - Advanced Blockchain Transaction Analysis Tool

This tool provides comprehensive blockchain transaction analysis across multiple networks,
including pattern detection, risk assessment, and fraud probability calculation.

Supported Blockchains:
  - Ethereum (ETH)    - Binance Smart Chain (BSC)    - Tron (TRX)
  - Solana (SOL)      - Polygon (MATIC)              - Avalanche (AVAX)
  - Cardano (ADA)     - Fantom (FTM)                 - Arbitrum (ARB)
  - Optimism (OP)     - And many more...

Key Features:
  - Multi-chain transaction analysis
  - Fraud pattern detection
  - Risk scoring and assessment
  - Network topology analysis
  - Temporal pattern detection
  - Exchange path identification
  - Anomaly detection
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  1. Basic Analysis:
     python cli.py -a 0x123... --format json --output results.json

  2. Deep Analysis with Risk Assessment:
     python cli.py -a 0x123... -b ethereum --mode deep --risk-threshold 0.8 --verbose

  3. Multi-Address Analysis:
     python cli.py -a 0x123... 0x456... Tz1... --batch-size 200 --depth 4

  4. Generate HTML Report:
     python cli.py -a 0x123... --format html --output report.html --lang en

  5. Debug Mode Analysis:
     python cli.py -a 0x123... --debug --verbose --no-progress

  6. Cross-Chain Analysis:
     python cli.py -a 0x123... Tz1... --mode deep --output cross_chain_analysis.json

        ''')
    
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('-a', '--addresses', nargs='+',
                         help='One or more wallet addresses to analyze. Supports multiple blockchain formats.')
    
    analysis = parser.add_argument_group('Analysis Configuration')
    analysis.add_argument('-b', '--blockchain', 
                         choices=['ethereum', 'bsc', 'tron', 'solana', 'polygon', 'all'],
                         help='Specific blockchain to analyze. Use "all" for auto-detection.')
    analysis.add_argument('-d', '--depth', type=int, default=3,
                         help='Analysis depth for transaction traversal (default: 3)')
    analysis.add_argument('--mode', choices=['quick', 'deep'], default='quick',
                         help='''Analysis mode:
                              quick: Basic transaction analysis
                              deep: Comprehensive analysis with pattern detection
                              (default: quick)''')
    analysis.add_argument('--risk-threshold', type=float, default=0.75,
                         help='Risk threshold for flagging suspicious transactions (0.0-1.0, default: 0.75)')
    analysis.add_argument('--batch-size', type=int, default=100,
                         help='Number of transactions to process in each batch (default: 100)')
    
    output = parser.add_argument_group('Output Options')
    output.add_argument('-o', '--output',
                       help='Output file path for analysis results')
    output.add_argument('-f', '--format', choices=['json', 'csv', 'html'], 
                       default='json',
                       help='''Output format:
                            json: Detailed JSON report
                            csv: Transaction data in CSV format
                            html: Interactive HTML report
                            (default: json)''')
    
    additional = parser.add_argument_group('Additional Options')
    additional.add_argument('--verbose', action='store_true',
                          help='Enable detailed output including analysis steps')
    additional.add_argument('--lang', choices=['en', 'fa'], default='en',
                          help='Interface language (en: English, fa: Farsi, default: en)')
    additional.add_argument('--no-progress', action='store_true',
                          help='Disable progress bars during analysis')
    additional.add_argument('--debug', action='store_true',
                          help='Enable debug mode with additional error information')
    additional.add_argument('--version', action='version', 
                          version=f'ChainWatch Analyzer v{VERSION}',
                          help='Show program version and exit')
    additional.add_argument('--health-check', action='store_true',
                          help='Run system health check to verify all components')
    
    advanced = parser.add_argument_group('Advanced Features')
    advanced.add_argument('--export-graphs', action='store_true',
                         help='Export transaction network graphs')
    advanced.add_argument('--include-metadata', action='store_true',
                         help='Include additional transaction metadata in output')
    advanced.add_argument('--cache-timeout', type=int, default=3600,
                         help='Cache timeout in seconds (default: 3600)')
    advanced.add_argument('--max-api-retries', type=int, default=3,
                         help='Maximum API retry attempts (default: 3)')
    
    return parser

async def analyze_addresses(args) -> Optional[dict[str, Any]]:
    try:
        analyzer = ChainWatchAnalyzer(
            addresses=args.addresses,
            log_callback=lambda msg, color: print(colored(msg, color)),
            language=args.lang,
            config_override={
                'analysis': {
                    'batch_size': args.batch_size,
                    'max_depth': args.depth,
                    'risk_threshold': args.risk_threshold,
                    'debug': args.debug
                }
            }
        )

        if not args.no_progress:
            pbar = tqdm(total=len(args.addresses), desc="Analyzing addresses")
        
        results = []
        for address in args.addresses:
            try:
                result = await analyzer.analyze_address(address)
                if result:
                    results.append(result)
                if not args.no_progress:
                    pbar.update(1)
            except Exception as e:
                print(colored(f"Error analyzing {address}: {e}", "red"))
                results.append({'address': address, 'error': str(e)})

        if not args.no_progress:
            pbar.close()

        if args.mode == 'deep' and results:
            try:
                ml_analyzer = EnhancedMLAnalysis(use_multiprocessing=True)
                patterns = ml_analyzer.analyze_suspicious_patterns(results)
                if patterns:
                    analyzer.results['ml_patterns'] = patterns
            except Exception as e:
                print(colored(f"Error in ML analysis: {e}", "yellow"))

        return analyzer.results

    except Exception as e:
        print(colored(f"Analysis error: {str(e)}", "red"))
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None

def save_results(results: dict, args) -> bool:
    if not results or not args.output:
        return False

    try:
        if args.format == 'json':
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        elif args.format == 'csv':
            import pandas as pd
            df = pd.DataFrame(results.get('transactions', []))
            df.to_csv(args.output, index=False)
        elif args.format == 'html':
            from analyzer.analyzer_core import generate_html_report
            with open(args.output, 'w') as f:
                f.write(generate_html_report(
                    results.get('fraud_probabilities', []),
                    results.get('suspicious_addresses', [])
                ))
        
        print(colored(f"\nResults saved to {args.output}", "green"))
        return True

    except Exception as e:
        print(colored(f"Error saving results: {str(e)}", "red"))
        return False

async def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.health_check:
        system_ok = await verify_system_health(args)
        if not system_ok and not args.debug:
            sys.exit(1)
        if not args.addresses:
            return
    elif not args.addresses:
        parser.error("the following arguments are required: -a/--addresses")

    if args.debug:
        print(colored("Debug mode enabled", "yellow"))
        print(colored(f"Arguments: {args}", "cyan"))

    try:
        results = await analyze_addresses(args)
        if results:
            if args.output:
                save_results(results, args)
            else:
                print(colored("\nAnalysis Results:", "cyan"))
                print(json.dumps(results, indent=2))
                
        print(colored("\n✅ Analysis completed", "green"))

    except KeyboardInterrupt:
        print(colored("\n⚠️ Analysis interrupted by user", "yellow"))
        sys.exit(1)
    except Exception as e:
        print(colored(f"\n❌ Fatal error: {str(e)}", "red"))
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def handle_interrupt(signum, frame):
    print(colored("\n\nAnalysis interrupted by user. Exiting...", "yellow"))
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_interrupt)
    check_dependencies()
    parser = create_parser()
    args = parser.parse_args()
    display_banner(args)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(colored("\nExiting...", "yellow"))
        sys.exit(0)
    except Exception as e:
        print(colored(f"Fatal error: {str(e)}", "red"))
        sys.exit(1)
