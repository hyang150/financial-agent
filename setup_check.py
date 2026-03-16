#!/usr/bin/env python3
"""
Environment Setup Checker for Finance Agent
Validates that all dependencies and configurations are properly set up
"""
import sys
import os
from pathlib import Path
from typing import Tuple, List

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_check(name: str, passed: bool, message: str = ""):
    """Print check result"""
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"{status} {name}")
    if message:
        prefix = "     " if passed else f"     {Colors.YELLOW}"
        suffix = "" if passed else Colors.RESET
        print(f"{prefix}{message}{suffix}")


def check_python_version() -> Tuple[bool, str]:
    """Check Python version"""
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}"

    if major >= 3 and minor >= 10:
        return True, f"Python {version}"
    else:
        return False, f"Python {version} (requires >= 3.10)"


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"{package_name} {version}"
    except ImportError:
        return False, f"{package_name} not installed"


def check_env_file() -> Tuple[bool, str]:
    """Check if .env file exists"""
    env_path = Path(".env")
    if env_path.exists():
        return True, ".env file found"
    else:
        return False, ".env file not found (copy .env.example to .env)"


def check_env_variables() -> List[Tuple[str, bool, str]]:
    """Check environment variables"""
    from dotenv import load_dotenv
    load_dotenv()

    checks = []

    # Required
    hf_token = os.getenv("HF_TOKEN")
    checks.append((
        "HF_TOKEN",
        bool(hf_token and not hf_token.startswith("hf_xxx")),
        "Required for downloading models"
    ))

    # Optional but recommended
    tavily_key = os.getenv("TAVILY_API_KEY")
    checks.append((
        "TAVILY_API_KEY",
        bool(tavily_key and not tavily_key.startswith("tvly-xxx")),
        "Optional: Required for web search"
    ))

    sec_email = os.getenv("SEC_EMAIL")
    checks.append((
        "SEC_EMAIL",
        bool(sec_email and "@" in sec_email and not sec_email.endswith("example.com")),
        "Required for SEC EDGAR downloads"
    ))

    return checks


def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, f"CUDA available: {device_name} ({memory:.1f} GB)"
        else:
            return False, "CUDA not available (will use CPU, training will be slow)"
    except ImportError:
        return False, "PyTorch not installed"


def check_directories() -> List[Tuple[str, bool, str]]:
    """Check required directories"""
    checks = []

    directories = {
        "config/": "Configuration directory",
        "data/": "Data storage directory",
        "src/": "Source code directory",
        "notebooks/": "Jupyter notebooks directory"
    }

    for dir_path, description in directories.items():
        exists = Path(dir_path).exists()
        checks.append((
            dir_path,
            exists,
            description
        ))

    return checks


def check_vector_db() -> Tuple[bool, str]:
    """Check if vector database exists"""
    db_path = Path("data/vector_db")
    if db_path.exists() and any(db_path.iterdir()):
        return True, "Vector database found"
    else:
        return False, "Vector database not found (run python src/ingestion.py)"


def main():
    """Run all checks"""
    print(f"\n{Colors.BOLD}Finance Agent - Environment Setup Checker{Colors.RESET}")
    print(f"Validating your environment...\n")

    all_passed = True

    # Python version
    print_header("1. Python Version")
    passed, msg = check_python_version()
    print_check("Python version", passed, msg)
    all_passed &= passed

    # Core packages
    print_header("2. Core Dependencies")

    core_packages = [
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("langchain", "langchain"),
        ("langchain-community", "langchain_community"),
        ("langchain-huggingface", "langchain_huggingface"),
        ("chromadb", "chromadb"),
        ("peft", "peft"),
        ("python-dotenv", "dotenv"),
    ]

    for package, import_name in core_packages:
        passed, msg = check_package(package, import_name)
        print_check(package, passed, msg)
        all_passed &= passed

    # Optional packages
    print_header("3. Optional Dependencies")

    optional_packages = [
        ("sec-edgar-downloader", "sec_edgar_downloader"),
        ("tavily-python", "tavily"),
        ("sentence-transformers", "sentence_transformers"),
    ]

    for package, import_name in optional_packages:
        passed, msg = check_package(package, import_name)
        print_check(package, passed, msg)

    # CUDA
    print_header("4. GPU/CUDA")
    passed, msg = check_cuda()
    print_check("CUDA availability", passed, msg)

    # Configuration
    print_header("5. Configuration")

    passed, msg = check_env_file()
    print_check(".env file", passed, msg)
    all_passed &= passed

    if passed:
        env_checks = check_env_variables()
        for var_name, var_passed, description in env_checks:
            print_check(var_name, var_passed, description)
            if "Required" in description:
                all_passed &= var_passed

    # Directories
    print_header("6. Project Structure")

    dir_checks = check_directories()
    for dir_name, dir_passed, description in dir_checks:
        print_check(dir_name, dir_passed, description)
        all_passed &= dir_passed

    # Vector DB
    print_header("7. Data Preparation")
    passed, msg = check_vector_db()
    print_check("Vector database", passed, msg)

    # Summary
    print_header("Summary")

    if all_passed:
        print(f"{Colors.GREEN}✓ All critical checks passed!{Colors.RESET}")
        print(f"{Colors.GREEN}  Your environment is ready to use.{Colors.RESET}\n")
        print("Next steps:")
        print("  1. If you haven't already, run: python src/ingestion.py")
        print("  2. Then run the agent: python main.py\n")
    else:
        print(f"{Colors.RED}✗ Some checks failed.{Colors.RESET}")
        print(f"{Colors.YELLOW}  Please fix the issues above before proceeding.{Colors.RESET}\n")
        print("Common fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Copy .env.example to .env and fill in your API keys")
        print("  3. Create missing directories: mkdir -p config data src notebooks\n")

    # Additional recommendations
    if not check_cuda()[0]:
        print(f"{Colors.YELLOW}⚠️  Recommendation:{Colors.RESET}")
        print("   GPU/CUDA not available. The agent will run on CPU.")
        print("   For better performance, consider:")
        print("   - Using a machine with NVIDIA GPU")
        print("   - Installing CUDA toolkit and PyTorch with CUDA support")
        print("   - Using smaller models or quantization\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
