"""
Integration of safe operating modes with the SelfImprovingRLModel.

This module provides functions to integrate safe operating modes and alerting
into the SelfImprovingRLModel class, implementing transitions between operating modes
and providing fallback mechanisms with automatic recovery.
"""

import logging
import os
import time
import datetime
import json
from typing import Dict, Any, Optional, List, Tuple

from src.ml.safe_operating_modes import (
    SafeOperatingSystem,
    OperatingMode,
    AlertConfig,
    RecoveryConfig
)
from src.ml.fallback_manager import FallbackLevel

logger = logging.getLogger(__name__)


def integrate_safe_operating_modes(model, 
                                  alert_config: Optional[AlertConfig] = None,
                                  recovery_config: Optional[RecoveryConfig] = None,
                                  start_monitoring: bool = True,
                                  monitoring_interval_sec: float = 60.0) -> SafeOperatingSystem:
    """
    Integrate safe operating modes into a SelfImprovingRLModel instance.
    
    This function adds safe operating modes and alerting to an existing
    SelfImprovingRLModel instance, implementing transitions between operating modes
    (reduced functionality → watch-only → trading halt) based on model health.
    
    Args:
        model: SelfImprovingRLModel instance to enhance
        alert_config: Optional AlertConfig instance
        recovery_config: Optional RecoveryConfig instance
        start_monitoring: Whether to start background monitoring
        monitoring_interval_sec: Monitoring interval in seconds
        
    Returns:
        SafeOperatingSystem instance
    """
    # Create default configs if not provided
    if alert_config is None:
        alert_config = AlertConfig(
            enable_console_alerts=True,
            enable_file_alerts=True,
            alert_log_path=os.path.join(os.getcwd(), "alerts.log"),
            # Enable Prometheus integration if available
            enable_prometheus_alerts=os.environ.get("ENABLE_PROMETHEUS_ALERTS", "false").lower() == "true",
            prometheus_gateway=os.environ.get("PROMETHEUS_GATEWAY", "http://localhost:9091"),
            # Enable Slack integration if available
            enable_slack_alerts=os.environ.get("ENABLE_SLACK_ALERTS", "false").lower() == "true",
            slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL", None)
        )
    
    if recovery_config is None:
        recovery_config = RecoveryConfig(
            enable_auto_recovery=True,
            initial_backoff_sec=60.0,
            max_backoff_sec=3600.0,
            backoff_factor=2.0,
            max_recovery_attempts=5,
            recovery_reset_hours=24.0
        )
    
    # Create safe operating system
    safe_system = SafeOperatingSystem(
        model_health_monitor=getattr(model, 'model_health_monitor', None),
        fallback_manager=getattr(model, 'fallback_manager', None),
        alert_config=alert_config,
        recovery_config=recovery_config
    )
    
    # Store in model
    model.safe_system = safe_system
    
    # Patch model methods
    _patch_act_method(model)
    _patch_update_method(model)
    _patch_train_method(model)
    
    # Set up synchronization between fallback manager and safe operating system
    if hasattr(model, 'fallback_manager') and model.fallback_manager is not None:
        _synchronize_fallback_with_safe_modes(model)
    
    # Start background monitoring if requested
    if start_monitoring:
        safe_system.start_monitoring(monitoring_interval_sec)
        
        # Schedule periodic effectiveness analysis
        _schedule_effectiveness_analysis(model)
    
    logger.info(
        "Safe operating modes integrated with monitoring=%s, interval=%.1f sec",
        start_monitoring, monitoring_interval_sec
    )
    
    return safe_system


def _patch_act_method(model):
    """
    Patch the act method to respect operating mode.
    
    Args:
        model: SelfImprovingRLModel instance to patch
    """
    original_act = model.act
    
    def act_with_safe_modes(state, epsilon=0.0, use_uncertainty=True):
        # Check current operating mode
        current_mode = model.safe_system.current_mode
        
        if current_mode == OperatingMode.NORMAL:
            # Normal operation - use original method
            return original_act(state, epsilon, use_uncertainty)
        
        elif current_mode == OperatingMode.REDUCED:
            # Reduced functionality - use more conservative epsilon
            conservative_epsilon = min(0.05, epsilon)  # Cap exploration
            return original_act(state, conservative_epsilon, True)  # Always use uncertainty
        
        elif current_mode == OperatingMode.WATCH_ONLY:
            # Watch-only mode - return action but log that it's not for execution
            action, config, metadata = original_act(state, 0.0, True)  # No exploration
            logger.warning(
                "Model in WATCH_ONLY mode - action %d generated but should not be executed",
                action
            )
            
            # Add watch-only flag to metadata
            metadata["watch_only"] = True
            metadata["operating_mode"] = "WATCH_ONLY"
            
            return action, config, metadata
        
        elif current_mode == OperatingMode.HALTED:
            # Halted mode - don't use model at all
            logger.error(
                "Model in HALTED mode - cannot generate actions"
            )
            
            # Return default action with halted flag
            import numpy as np
            return 0, np.zeros(model.config_size), {
                "halted": True,
                "operating_mode": "HALTED",
                "message": "Model is halted and cannot generate actions"
            }
    
    # Replace the original method
    model.act = act_with_safe_modes


def _patch_update_method(model):
    """
    Patch the update method to respect operating mode.
    
    Args:
        model: SelfImprovingRLModel instance to patch
    """
    original_update = model.update
    
    def update_with_safe_modes(step_in_episode_count=0):
        # Check current operating mode
        current_mode = model.safe_system.current_mode
        
        if current_mode == OperatingMode.HALTED:
            # Halted mode - don't update model
            logger.error(
                "Model in HALTED mode - update skipped"
            )
            
            # Return empty metrics
            from src.ml.metrics import TrainingMetrics
            return TrainingMetrics(
                loss=0.0,
                q_value_mean=0.0,
                q_value_std=0.0,
                td_error_mean=0.0,
                reward_mean=0.0,
                step=model.total_steps
            )
        
        # For other modes, allow updates but assess health afterward
        metrics = original_update(step_in_episode_count)
        
        # Assess health after update
        if hasattr(model, 'model_health_monitor'):
            model.model_health_monitor.assess_health(model)
            
            # Update operating mode based on health
            model.safe_system.assess_and_update_mode()
        
        return metrics
    
    # Replace the original method
    model.update = update_with_safe_modes


def create_effectiveness_report(model, filepath: str) -> bool:
    """
    Create and save an effectiveness report for the model.
    
    Args:
        model: SelfImprovingRLModel instance
        filepath: Path to save report
        
    Returns:
        True if report was saved, False otherwise
    """
    if not hasattr(model, 'safe_system'):
        logger.error("Model does not have safe_system attribute")
        return False
    
    return model.safe_system.save_effectiveness_report(filepath)
def _patch_train_method(model):
    """
    Patch the train method to respect operating mode.
    
    Args:
        model: SelfImprovingRLModel instance to patch
    """
    # Check if model has a train method
    if not hasattr(model, 'train'):
        logger.warning("Model does not have a train method, skipping patch")
        return

    original_train = model.train

    def _handle_halted_mode():
        logger.error("Model in HALTED mode - training skipped")
        return None

    def _handle_watch_only_mode(episodes):
        logger.warning("Model in WATCH_ONLY mode - training with reduced episodes")
        return max(1, int(episodes * 0.1))

    def _handle_reduced_mode(max_steps_per_episode, kwargs):
        logger.info("Model in REDUCED mode - training with conservative parameters")
        if max_steps_per_episode > 100:
            max_steps_per_episode = max(100, int(max_steps_per_episode * 0.5))
        if 'learning_rate' in kwargs:
            kwargs['learning_rate'] = kwargs['learning_rate'] * 0.5
        if 'epsilon' in kwargs:
            kwargs['epsilon'] = min(0.05, kwargs['epsilon'])
        return max_steps_per_episode, kwargs

    def train_with_safe_modes(episodes, max_steps_per_episode=1000, **kwargs):
        current_mode = model.safe_system.current_mode

        if current_mode == OperatingMode.HALTED:
            return _handle_halted_mode()

        if current_mode == OperatingMode.WATCH_ONLY:
            episodes = _handle_watch_only_mode(episodes)

        if current_mode == OperatingMode.REDUCED:
            max_steps_per_episode, kwargs = _handle_reduced_mode(max_steps_per_episode, kwargs)

        result = original_train(episodes, max_steps_per_episode, **kwargs)

        if hasattr(model, 'model_health_monitor'):
            model.model_health_monitor.assess_health(model)
            model.safe_system.assess_and_update_mode()

        return result

    model.train = train_with_safe_modes


def _synchronize_fallback_with_safe_modes(model):
    """
    Synchronize fallback manager with safe operating modes.
    
    This ensures that fallback levels and operating modes are aligned.
    
    Args:
        model: SelfImprovingRLModel instance
    """
    # Create a mapping between fallback levels and operating modes
    fallback_to_mode_map = {
        FallbackLevel.NORMAL: OperatingMode.NORMAL,
        FallbackLevel.SIMPLIFIED_RL: OperatingMode.REDUCED,
        FallbackLevel.RULE_BASED: OperatingMode.WATCH_ONLY,
        FallbackLevel.CONSERVATIVE: OperatingMode.WATCH_ONLY,
        FallbackLevel.HALT: OperatingMode.HALTED
    }
    
    # Store original escalate_fallback method
    original_escalate = model.fallback_manager.escalate_fallback
    
    def escalate_with_mode_sync(reason=""):
        # Call original method
        new_level = original_escalate(reason)
        
        # Synchronize with operating mode
        if new_level in fallback_to_mode_map:
            target_mode = fallback_to_mode_map[new_level]
            
            # Only change if the mode would be more restrictive
            if target_mode.value > model.safe_system.current_mode.value:
                model.safe_system.set_operating_mode(
                    target_mode,
                    f"Synchronized with fallback level {new_level.name}: {reason}"
                )
        
        return new_level
    
    # Replace the method
    model.fallback_manager.escalate_fallback = escalate_with_mode_sync
    
    # Also patch the reset method
    original_reset = model.fallback_manager.reset
    
    def reset_with_mode_sync():
        # Call original method
        original_reset()
        
        # Synchronize with operating mode if appropriate
        if model.safe_system.current_mode != OperatingMode.NORMAL:
            # Check if health is good enough for normal operation
            if hasattr(model, 'model_health_monitor'):
                health_metrics = model.model_health_monitor.assess_health(model)
                if health_metrics.overall_health > 0.8:
                    model.safe_system.set_operating_mode(
                        OperatingMode.NORMAL,
                        "Synchronized with fallback reset"
                    )
    
    # Replace the method
    model.fallback_manager.reset = reset_with_mode_sync
    
    logger.info("Fallback manager synchronized with safe operating modes")


def _log_mode_history(report):
    if "mode_history" in report and report["mode_history"]:
        recent_modes = report["mode_history"][-3:]
        logger.info(
            "Recent mode changes: %s",
            ", ".join([f"{m['old_mode']} → {m['new_mode']}" for m in recent_modes])
        )

def _log_recovery_attempts(report):
    if "recovery_attempts" in report:
        logger.info(
            "Recovery attempts: %d (last attempt: %s)",
            report["recovery_attempts"],
            report.get("last_recovery_time") or "never"
        )

def _log_fallback_effectiveness(report):
    for level_name, stats in report.items():
        if isinstance(stats, dict) and "activation_count" in stats:
            success_rate = 0
            if stats["activation_count"] > 0:
                success_rate = stats["success_count"] / stats["activation_count"] * 100
            logger.info(
                "Fallback %s: %d activations, %.1f%% success rate, avg recovery time: %.1f sec",
                level_name,
                stats["activation_count"],
                success_rate,
                stats.get("avg_recovery_time_sec") or 0
            )

def _save_effectiveness_report(model):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"effectiveness_report_{timestamp}.json"
    model.safe_system.save_effectiveness_report(filepath)

def _schedule_effectiveness_analysis(model, interval_hours=6):
    """
    Schedule periodic effectiveness analysis.

    Args:
        model: SelfImprovingRLModel instance
        interval_hours: Interval between analyses in hours
    """
    import threading

    def analysis_task():
        while True:
            try:
                time.sleep(interval_hours * 3600)
                report = model.safe_system.get_effectiveness_report()
                logger.info("Effectiveness analysis completed:")
                _log_mode_history(report)
                _log_recovery_attempts(report)
                _log_fallback_effectiveness(report)
                _save_effectiveness_report(model)
            except Exception as e:
                logger.error("Error in effectiveness analysis: %s", str(e))

    analysis_thread = threading.Thread(
        target=analysis_task,
        daemon=True
    )
    analysis_thread.start()

    logger.info(
        "Scheduled effectiveness analysis every %d hours",
        interval_hours
    )


def _calculate_fallback_summary(report):
    summary = {
        "total_fallback_activations": 0,
        "overall_success_rate": 0,
        "avg_recovery_time_sec": 0,
        "most_common_fallback": None,
        "most_common_fallback_count": 0
    }
    total_activations = 0
    total_successes = 0
    total_recovery_time = 0
    recovery_count = 0

    for level_name, stats in report.items():
        if isinstance(stats, dict) and "activation_count" in stats:
            total_activations += stats["activation_count"]
            total_successes += stats["success_count"]

            if stats.get("avg_recovery_time_sec") and stats["success_count"] > 0:
                total_recovery_time += stats["avg_recovery_time_sec"] * stats["success_count"]
                recovery_count += stats["success_count"]

            if stats["activation_count"] > summary["most_common_fallback_count"]:
                summary["most_common_fallback"] = level_name
                summary["most_common_fallback_count"] = stats["activation_count"]

    summary["total_fallback_activations"] = total_activations

    if total_activations > 0:
        summary["overall_success_rate"] = total_successes / total_activations

    if recovery_count > 0:
        summary["avg_recovery_time_sec"] = total_recovery_time / recovery_count

    return summary

def analyze_fallback_effectiveness(model, output_file=None):
    """
    Analyze the effectiveness of fallback mechanisms.

    This function generates a detailed report on fallback activations,
    success rates, and recovery times.

    Args:
        model: SelfImprovingRLModel instance
        output_file: Optional file path to save the report

    Returns:
        Dictionary with effectiveness metrics
    """
    if not hasattr(model, 'safe_system'):
        logger.error("Model does not have safe_system attribute")
        return None

    report = model.safe_system.get_effectiveness_report()
    report["analysis_timestamp"] = datetime.datetime.now().isoformat()
    report["summary"] = _calculate_fallback_summary(report)

    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info("Fallback effectiveness analysis saved to %s", output_file)
        except Exception as e:
            logger.error("Failed to save effectiveness analysis: %s", str(e))

    return report