#!/usr/bin/env python3
"""
Enhanced configuration and encryption management CLI for AURA AI.
Provides command-line interface for managing encrypted configuration data.
"""

import os
import sys
import argparse
import logging
from getpass import getpass
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.secure_config_manager_enhanced import EnhancedSecureConfigManager
from src.utils.encryption import SecureStorage, EncryptionError

# Constants
CONFIG_FILE_PATH = "config/config.json"
SCHEMA_FILE_PATH = "config/schema.json"
PASSWORD_PROMPT = "Enter encryption password: "


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def init_encryption(args):
    """Initialize encryption for the configuration."""
    print("Initializing encryption for AURA AI configuration...")
    
    config_path = args.config or CONFIG_FILE_PATH
    schema_path = args.schema or SCHEMA_FILE_PATH
    
    # Check if already encrypted
    if os.path.exists(f"{config_path}.encrypted"):
        print("Configuration encryption already initialized!")
        return 0
    
    password = None
    if args.password:
        password = getpass(PASSWORD_PROMPT)
    
    try:
        config_manager = EnhancedSecureConfigManager(
            config_path=config_path,
            schema_path=schema_path,
            encryption_key=password
        )
        
        print("✅ Encryption initialized successfully!")
        print(f"Configuration: {config_path}")
        print(f"Schema: {schema_path}")
        
        # Perform initial security validation
        security_issues = config_manager.validate_configuration_security()
        if security_issues:
            print("⚠️  Security issues found:")
            for issue in security_issues:
                print(f"  - {issue}")
        else:
            print("✅ No security issues found")
            
        return 0
        
    except Exception as e:
        print(f"❌ Failed to initialize encryption: {e}")
        return 1


def encrypt_value(args):
    """Encrypt a specific configuration value."""
    config_path = args.config or CONFIG_FILE_PATH
    schema_path = args.schema or SCHEMA_FILE_PATH
    
    password = None
    if args.password:
        password = getpass(PASSWORD_PROMPT)
    
    try:
        config_manager = EnhancedSecureConfigManager(
            config_path=config_path,
            schema_path=schema_path,
            encryption_key=password
        )
        
        key = args.key
        value = args.value or getpass(f"Enter value for {key}: ")
        
        success = config_manager.set_encrypted(key, value)
        if success:
            print(f"✅ Successfully encrypted and stored: {key}")
        else:
            print(f"❌ Failed to encrypt: {key}")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"❌ Failed to encrypt value: {e}")
        return 1


def decrypt_value(args):
    """Decrypt a specific configuration value."""
    config_path = args.config or CONFIG_FILE_PATH
    schema_path = args.schema or SCHEMA_FILE_PATH
    
    password = None
    if args.password:
        password = getpass(PASSWORD_PROMPT)
    
    try:
        config_manager = EnhancedSecureConfigManager(
            config_path=config_path,
            schema_path=schema_path,
            encryption_key=password
        )
        
        key = args.key
        value = config_manager.get_encrypted(key)
        
        if value is not None:
            if args.show:
                print(f"{key}: {value}")
            else:
                print(f"✅ Successfully decrypted: {key} (use --show to display value)")
        else:
            print(f"❌ Key not found or failed to decrypt: {key}")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"❌ Failed to decrypt value: {e}")
        return 1


def security_status(args):
    """Check security status of configuration."""
    config_path = args.config or CONFIG_FILE_PATH
    schema_path = args.schema or SCHEMA_FILE_PATH
    
    password = None
    if args.password:
        password = getpass(PASSWORD_PROMPT)
    
    try:
        config_manager = EnhancedSecureConfigManager(
            config_path=config_path,
            schema_path=schema_path,
            encryption_key=password
        )
        
        print("🔍 Security Status Check")
        print("=" * 50)
        
        # Get security summary
        summary = config_manager.get_security_summary()
        
        print(f"Configuration File: {config_path}")
        print(f"Schema File: {schema_path}")
        print(f"Encryption Enabled: {'✅' if summary['encryption_enabled'] else '❌'}")
        print(f"Valid Configuration: {'✅' if summary['config_valid'] else '❌'}")
        print(f"Sensitive Data Protected: {'✅' if summary['sensitive_data_protected'] else '❌'}")
        print(f"Last Modified: {summary['last_modified']}")
        print(f"Config Hash: {summary['config_hash'][:16]}...")
        
        # Check for security issues
        security_issues = config_manager.validate_configuration_security()
        if security_issues:
            print("\n⚠️  Security Issues Found:")
            for i, issue in enumerate(security_issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\n✅ No security issues found")
        
        # Environment validation
        print("\n🌍 Environment Validation:")
        try:
            config_manager._validate_environment_comprehensive()
            print("✅ Environment variables are valid")
        except Exception as e:
            print(f"❌ Environment validation failed: {e}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to check security status: {e}")
        return 1


def validate_config(args):
    """Validate configuration against schema and security requirements."""
    config_path = args.config or CONFIG_FILE_PATH
    schema_path = args.schema or SCHEMA_FILE_PATH
    
    try:
        config_manager = EnhancedSecureConfigManager(
            config_path=config_path,
            schema_path=schema_path
        )
        
        print("🔍 Configuration Validation")
        print("=" * 50)
        
        # Validate configuration
        config_manager._validate_configuration_comprehensive()
        print("✅ Configuration validation passed")
        
        # Validate environment
        config_manager._validate_environment_comprehensive()
        print("✅ Environment validation passed")
        
        # Security validation
        security_issues = config_manager.validate_configuration_security()
        if security_issues:
            print("\n⚠️  Security Issues:")
            for issue in security_issues:
                print(f"  - {issue}")
        else:
            print("✅ Security validation passed")
        
        return 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AURA AI Configuration and Encryption Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init --password              Initialize encryption
  %(prog)s encrypt --key api_key        Encrypt a configuration value
  %(prog)s decrypt --key api_key        Decrypt a configuration value
  %(prog)s status                       Check security status
  %(prog)s validate                     Validate configuration
        """
    )
    
    # Global options
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--schema', '-s', help='Schema file path')
    parser.add_argument('--password', '-p', action='store_true', help='Prompt for encryption password')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize encryption')
    init_parser.set_defaults(func=init_encryption)
    
    # Encrypt command
    encrypt_parser = subparsers.add_parser('encrypt', help='Encrypt a configuration value')
    encrypt_parser.add_argument('--key', required=True, help='Configuration key to encrypt')
    encrypt_parser.add_argument('--value', help='Value to encrypt (will prompt if not provided)')
    encrypt_parser.set_defaults(func=encrypt_value)
    
    # Decrypt command
    decrypt_parser = subparsers.add_parser('decrypt', help='Decrypt a configuration value')
    decrypt_parser.add_argument('--key', required=True, help='Configuration key to decrypt')
    decrypt_parser.add_argument('--show', action='store_true', help='Show decrypted value')
    decrypt_parser.set_defaults(func=decrypt_value)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check security status')
    status_parser.set_defaults(func=security_status)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.set_defaults(func=validate_config)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if hasattr(args, 'func'):
        try:
            return args.func(args)
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return 1
        except Exception as e:
            if args.verbose:
                import traceback
                traceback.print_exc()
            else:
                print(f"❌ Error: {e}")
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
