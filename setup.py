#!/usr/bin/env python3
"""
AEGIS - Advanced Encrypted Governance and Intelligence System
Setup script for installation and distribution
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure Python version compatibility
if sys.version_info < (3, 8):
    sys.exit("AEGIS requires Python 3.8 or higher")

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
def read_requirements(filename):
    """Read requirements from file, filtering out comments and empty lines."""
    requirements = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle git+https URLs and version specifiers
                    if line.startswith('git+'):
                        requirements.append(line)
                    elif '==' in line or '>=' in line or '<=' in line or '>' in line or '<' in line:
                        requirements.append(line)
                    else:
                        requirements.append(line)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
    return requirements

# Core requirements
install_requires = read_requirements('requirements.txt')

# Development requirements
dev_requires = read_requirements('requirements-dev.txt')

# Optional requirements for different features
extras_require = {
    'dev': dev_requires,
    'test': [
        'pytest>=7.0.0',
        'pytest-asyncio>=0.21.0',
        'pytest-cov>=4.0.0',
        'pytest-mock>=3.10.0',
        'pytest-benchmark>=4.0.0',
        'pytest-xdist>=3.0.0',
        'coverage>=7.0.0',
        'factory-boy>=3.2.0',
        'faker>=18.0.0',
    ],
    'docs': [
        'sphinx>=6.0.0',
        'sphinx-rtd-theme>=1.2.0',
        'sphinx-autodoc-typehints>=1.22.0',
        'myst-parser>=1.0.0',
        'sphinx-copybutton>=0.5.0',
    ],
    'monitoring': [
        'prometheus-client>=0.16.0',
        'grafana-api>=1.0.3',
        'elasticsearch>=8.0.0',
        'loguru>=0.7.0',
    ],
    'security': [
        'bandit>=1.7.0',
        'safety>=2.3.0',
        'semgrep>=1.0.0',
        'cryptography>=40.0.0',
    ],
    'performance': [
        'uvloop>=0.17.0',
        'orjson>=3.8.0',
        'cython>=0.29.0',
        'numba>=0.57.0',
    ],
    'cloud': [
        'boto3>=1.26.0',
        'google-cloud-storage>=2.7.0',
        'azure-storage-blob>=12.14.0',
        'kubernetes>=26.1.0',
    ],
}

# All extras combined
extras_require['all'] = list(set(sum(extras_require.values(), [])))

# Entry points for command-line scripts
entry_points = {
    'console_scripts': [
        'aegis=main:main',
        'aegis-node=main:start_node',
        'aegis-test=run_tests:main',
        'aegis-benchmark=benchmarks.run_benchmarks:main',
        'aegis-monitor=monitoring_dashboard:main',
        'aegis-backup=backup_system:main',
        'aegis-crypto=crypto_framework:main',
        'aegis-p2p=p2p_network:main',
        'aegis-consensus=consensus_algorithm:main',
        'aegis-storage=storage_system:main',
        'aegis-web=web_dashboard:main',
    ],
}

# Classifiers for PyPI
classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: System Administrators',
    'Intended Audience :: Information Technology',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Topic :: Security :: Cryptography',
    'Topic :: System :: Distributed Computing',
    'Topic :: System :: Networking',
    'Topic :: Database :: Database Engines/Servers',
    'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: System :: Systems Administration',
    'Topic :: Security',
    'Environment :: Console',
    'Environment :: Web Environment',
    'Framework :: AsyncIO',
    'Natural Language :: English',
    'Natural Language :: Spanish',
]

# Keywords for better discoverability
keywords = [
    'blockchain', 'cryptography', 'p2p', 'consensus', 'distributed-systems',
    'security', 'encryption', 'decentralized', 'network', 'storage',
    'monitoring', 'dashboard', 'backup', 'testing', 'framework',
    'quantum-resistant', 'post-quantum', 'hybrid-consensus', 'paxos',
    'raft', 'pbft', 'tor', 'onion-routing', 'privacy', 'anonymity'
]

# Project URLs
project_urls = {
    'Homepage': 'https://github.com/AEGIS-Project/AEGIS',
    'Documentation': 'https://aegis-project.readthedocs.io/',
    'Repository': 'https://github.com/AEGIS-Project/AEGIS',
    'Bug Tracker': 'https://github.com/AEGIS-Project/AEGIS/issues',
    'Changelog': 'https://github.com/AEGIS-Project/AEGIS/blob/main/docs/CHANGELOG.md',
    'Security Policy': 'https://github.com/AEGIS-Project/AEGIS/security/policy',
    'Discussions': 'https://github.com/AEGIS-Project/AEGIS/discussions',
    'Wiki': 'https://github.com/AEGIS-Project/AEGIS/wiki',
}

# Package data to include
package_data = {
    'aegis': [
        'config/*.yml',
        'config/*.yaml',
        'config/*.json',
        'templates/*.html',
        'templates/*.css',
        'templates/*.js',
        'static/*',
        'schemas/*.json',
        'certs/*.pem',
        'docs/*.md',
        'tests/fixtures/*.json',
        'tests/fixtures/*.yml',
    ],
}

# Data files to include in the distribution
data_files = [
    ('config', ['config/default.yml', 'config/logging.yml']),
    ('docs', ['README.md', 'LICENSE', 'docs/CHANGELOG.md']),
    ('scripts', ['scripts/install.sh', 'scripts/setup_environment.py']),
]

setup(
    # Basic package information
    name='aegis-framework',
    version='1.0.0',
    author='AEGIS Development Team',
    author_email='dev@aegis-project.org',
    maintainer='AEGIS Security Team',
    maintainer_email='security@aegis-project.org',
    
    # Package description
    description='Advanced Encrypted Governance and Intelligence System - A quantum-resistant distributed framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    # URLs and metadata
    url='https://github.com/AEGIS-Project/AEGIS',
    project_urls=project_urls,
    
    # Package discovery and structure
    packages=find_packages(exclude=['tests*', 'benchmarks*', 'docs*', 'scripts*']),
    package_data=package_data,
    data_files=data_files,
    include_package_data=True,
    
    # Dependencies
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires='>=3.8',
    
    # Entry points
    entry_points=entry_points,
    
    # Classification
    classifiers=classifiers,
    keywords=keywords,
    license='MIT',
    
    # Additional metadata
    platforms=['any'],
    zip_safe=False,  # Due to data files and potential C extensions
    
    # Options for different build systems
    options={
        'build_py': {
            'compile': True,
            'optimize': 2,
        },
        'bdist_wheel': {
            'universal': False,  # Not universal due to potential C extensions
        },
        'egg_info': {
            'tag_build': '',
            'tag_date': False,
        },
    },
    
    # Custom commands
    cmdclass={},
    
    # Namespace packages (if any)
    namespace_packages=[],
)

# Post-installation message
def print_post_install_message():
    """Print helpful information after installation."""
    print("\n" + "="*60)
    print("🛡️  AEGIS Framework Successfully Installed! 🛡️")
    print("="*60)
    print("\n📋 Quick Start:")
    print("   aegis --help                 # Show all available commands")
    print("   aegis-node --config config/  # Start AEGIS node")
    print("   aegis-test                   # Run test suite")
    print("   aegis-monitor                # Start monitoring dashboard")
    print("\n📚 Documentation:")
    print("   README.md                    # Getting started guide")
    print("   docs/API_REFERENCE.md        # Complete API reference")
    print("   docs/DEPLOYMENT_GUIDE.md     # Deployment instructions")
    print("   docs/SECURITY_GUIDE.md       # Security best practices")
    print("\n🔧 Configuration:")
    print("   config/default.yml           # Default configuration")
    print("   config/logging.yml           # Logging configuration")
    print("\n🚀 Next Steps:")
    print("   1. Review the configuration files in config/")
    print("   2. Set up your environment variables (.env file)")
    print("   3. Initialize the system: aegis-node --init")
    print("   4. Start the node: aegis-node --start")
    print("\n⚠️  Security Notice:")
    print("   - Generate new cryptographic keys before production use")
    print("   - Review security settings in config/security.yml")
    print("   - Enable monitoring and logging for production deployments")
    print("\n📞 Support:")
    print("   - Documentation: https://aegis-project.readthedocs.io/")
    print("   - Issues: https://github.com/AEGIS-Project/AEGIS/issues")
    print("   - Discussions: https://github.com/AEGIS-Project/AEGIS/discussions")
    print("\n" + "="*60)
    print("Thank you for using AEGIS! 🙏")
    print("="*60 + "\n")

# Custom command to show post-install message
class PostInstallCommand:
    """Custom command to run after installation."""
    
    def run(self):
        print_post_install_message()

# Add custom command if running setup.py directly
if __name__ == '__main__':
    # Check if we're in install mode
    if 'install' in sys.argv:
        # Add post-install hook
        import atexit
        atexit.register(print_post_install_message)
    
    # Run setup
    setup()