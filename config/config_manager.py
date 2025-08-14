"""
Configuration Management

Centralized configuration management for the entire application.
Handles environment variables, default settings, and configuration validation.
"""

import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
from loguru import logger


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment variables"""
        # Load default configuration
        self._config = self._get_default_config()
        
        # Load from config file if provided
        if self.config_path and Path(self.config_path).exists():
            self._load_from_file()
        
        # Override with environment variables
        self._load_from_environment()
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Configuration loaded successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'data': {
                'path': 'data/concatenated_data/concatenated_data.json',
                'cache_enabled': True,
                'cache_size': 1000
            },
            'analysis': {
                'output_dir': 'reports',
                'log_level': 'INFO',
                'parallel_processing': False,
                'max_workers': 4
            },
            'rag': {
                'model': 'gemini-1.5-flash',
                'api_key_env': 'GEMINI_API_KEY',
                'max_tokens': 4096,
                'temperature': 0.1
            },
            'dashboard': {
                'port': 8501,
                'host': '0.0.0.0',
                'theme': 'light'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/analysis.log',
                'rotation': '10 MB',
                'retention': '30 days'
            }
        }
    
    def _load_from_file(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_config(file_config)
                    logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from file {self.config_path}: {e}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'DATA_PATH': ('data', 'path'),
            'DATA_CACHE_ENABLED': ('data', 'cache_enabled'),
            'ANALYSIS_OUTPUT_DIR': ('analysis', 'output_dir'),
            'ANALYSIS_LOG_LEVEL': ('analysis', 'log_level'),
            'RAG_MODEL': ('rag', 'model'),
            'RAG_API_KEY': ('rag', 'api_key_env'),
            'DASHBOARD_PORT': ('dashboard', 'port'),
            'DASHBOARD_HOST': ('dashboard', 'host'),
            'LOG_LEVEL': ('logging', 'level'),
            'LOG_FILE': ('logging', 'file')
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_path, value)
                logger.debug(f"Loaded {env_var}={value}")
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing config"""
        def merge_dicts(base: Dict[str, Any], update: Dict[str, Any]):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dicts(base[key], value)
                else:
                    base[key] = value
        
        merge_dicts(self._config, new_config)
    
    def _set_nested_value(self, path: tuple, value: Any):
        """Set a nested configuration value"""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate data path
        data_path = self.get('data.path')
        if data_path and not Path(data_path).exists():
            logger.warning(f"Data path does not exist: {data_path}")
        
        # Validate output directory
        output_dir = self.get('analysis.output_dir')
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate log directory
        log_file = self.get('logging.file')
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate RAG API key
        api_key_env = self.get('rag.api_key_env')
        if api_key_env and not os.getenv(api_key_env):
            logger.warning(f"RAG API key environment variable {api_key_env} not set")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation key.
        
        Args:
            key: Configuration key (e.g., 'data.path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        current = self._config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by dot notation key.
        
        Args:
            key: Configuration key (e.g., 'data.path')
            value: Value to set
        """
        keys = key.split('.')
        self._set_nested_value(tuple(keys), value)
    
    def get_data_path(self) -> str:
        """Get data path configuration"""
        return self.get('data.path', 'data/concatenated_data/concatenated_data.json')
    
    def get_output_dir(self) -> str:
        """Get output directory configuration"""
        return self.get('analysis.output_dir', 'reports')
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration"""
        return self.get('rag', {})
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration"""
        return self.get('dashboard', {})
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration"""
        return self.get('analysis', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary"""
        return self._config.copy()
    
    def save_config(self, path: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration (uses config_path if not provided)
        """
        save_path = path or self.config_path
        if not save_path:
            logger.error("No path provided for saving configuration")
            return
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def init_config(config_path: Optional[str] = None) -> ConfigManager:
    """Initialize global configuration"""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager 