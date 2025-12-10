"""
Configuration management module.
Loads and manages configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the fraud detection system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config YAML file. If None, uses default.
        """
        if config_path is None:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        Supports nested keys with dot notation (e.g., 'paths.data.raw').
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save config. If None, overwrites original file.
        """
        save_path = path if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Get all path configurations."""
        return self._config.get('paths', {})
    
    @property
    def dataset(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self._config.get('dataset', {})
    
    @property
    def cleaning(self) -> Dict[str, Any]:
        """Get cleaning configuration."""
        return self._config.get('cleaning', {})
    
    @property
    def features(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self._config.get('features', {})
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config.get('model', {})
    
    def __repr__(self) -> str:
        return f"Config(path='{self.config_path}')"


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to config file. Only used on first call.
        
    Returns:
        Config instance
    """
    global _config
    
    if _config is None:
        _config = Config(config_path)
    
    return _config


def reload_config(config_path: str = None):
    """
    Reload configuration from file.
    
    Args:
        config_path: Path to config file
    """
    global _config
    _config = Config(config_path)


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Raw data path: {config.get('paths.data.raw')}")
    print(f"Test size: {config.get('dataset.test_size')}")
    print(f"Model algorithm: {config.get('model.supervised.algorithm')}")
