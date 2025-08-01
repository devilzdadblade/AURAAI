"""
Safe operating modes and alerting system for reinforcement learning models.

This module provides mechanisms for transitioning between different operating modes
based on model health, as well as alerting operators when issues are detected.
"""

import logging
import time
import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import threading
import json
import os
import socket
import urllib.request
import urllib.parse
import ssl
import smtplib
from email.message import EmailMessage

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Constants for history management
MAX_HISTORY_SIZE = 1000


class OperatingMode(Enum):
    """Enumeration of safe operating modes."""
    NORMAL = 0  # Full functionality
    REDUCED = 1  # Reduced functionality (e.g., lower risk)
    WATCH_ONLY = 2  # Monitoring only, no trading
    HALTED = 3  # Complete system halt


@dataclass
class AlertConfig:
    """Configuration for alerting system."""
    
    enable_console_alerts: bool = True
    enable_file_alerts: bool = True
    enable_webhook_alerts: bool = False
    enable_email_alerts: bool = False
    enable_prometheus_alerts: bool = False
    enable_slack_alerts: bool = False
    
    alert_log_path: str = "alerts.log"
    webhook_url: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    prometheus_gateway: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    
    min_alert_interval_sec: float = 60.0  # Minimum time between alerts
    alert_cooldown_sec: float = 300.0  # Cooldown after alert
    
    # Alert severity thresholds for different operating modes
    reduced_mode_alert_level: str = "warning"
    watch_only_mode_alert_level: str = "error"
    halted_mode_alert_level: str = "critical"


@dataclass
class RecoveryConfig:
    """Configuration for automatic recovery attempts."""
    
    enable_auto_recovery: bool = True
    initial_backoff_sec: float = 60.0  # Initial backoff time
    max_backoff_sec: float = 3600.0  # Maximum backoff time (1 hour)
    backoff_factor: float = 2.0  # Exponential backoff factor
    max_recovery_attempts: int = 5  # Maximum number of recovery attempts
    recovery_reset_hours: float = 24.0  # Time after which recovery count resets


class SafeOperatingSystem:
    """
    Manages safe operating modes and alerting for RL models.
    
    This class provides mechanisms for transitioning between different operating
    modes based on model health, as well as alerting operators when issues are detected.
    """
    
    def __init__(self, 
                 model_health_monitor=None,
                 fallback_manager=None,
                 alert_config: Optional[AlertConfig] = None,
                 recovery_config: Optional[RecoveryConfig] = None):
        """
        Initialize the safe operating system.
        
        Args:
            model_health_monitor: Optional ModelHealthMonitor instance
            fallback_manager: Optional FallbackManager instance
            alert_config: Configuration for alerting system
            recovery_config: Configuration for automatic recovery
        """
        self.model_health_monitor = model_health_monitor
        self.fallback_manager = fallback_manager
        self.alert_config = alert_config or AlertConfig()
        self.recovery_config = recovery_config or RecoveryConfig()
        
        # Operating mode state
        self.current_mode = OperatingMode.NORMAL
        self.mode_history = []
        self.mode_change_time = datetime.datetime.now()
        
        # Recovery state
        self.recovery_attempts = 0
        self.last_recovery_time = None
        self.current_backoff_sec = self.recovery_config.initial_backoff_sec
        
        # Alerting state
        self.last_alert_time = None
        self.alert_history = []
        
        # Background monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
        logger.info(
            "SafeOperatingSystem initialized in %s mode",
            self.current_mode.name
        )
    
    def set_operating_mode(self, mode: OperatingMode, reason: str) -> bool:
        """
        Set the current operating mode.
        
        Args:
            mode: New operating mode
            reason: Reason for the mode change
            
        Returns:
            True if mode was changed, False otherwise
        """
        if mode == self.current_mode:
            return False
        
        old_mode = self.current_mode
        self.current_mode = mode
        self.mode_change_time = datetime.datetime.now()
        
        self.mode_history.append({
            "timestamp": self.mode_change_time,
            "old_mode": old_mode.name,
            "new_mode": mode.name,
            "reason": reason
        })
        if len(self.mode_history) > MAX_HISTORY_SIZE:
            self.mode_history = self.mode_history[-MAX_HISTORY_SIZE:]
        
        logger.warning(
            "Operating mode changed from %s to %s: %s",
            old_mode.name, mode.name, reason
        )
        
        self.send_alert(
            title=f"Operating Mode Changed to {mode.name}",
            message=f"Operating mode changed from {old_mode.name} to {mode.name}: {reason}",
            severity="warning" if mode.value > old_mode.value else "info"
        )
        
        self._update_fallback_manager(mode)
        
        return True

    def _update_fallback_manager(self, mode: OperatingMode) -> None:
        """Update fallback manager based on operating mode."""
        if self.fallback_manager is None:
            return
        if mode == OperatingMode.NORMAL:
            self.fallback_manager.reset()
        elif mode == OperatingMode.REDUCED:
            if self.fallback_manager.current_level.value < 1:
                self.fallback_manager.escalate_fallback(f"Mode change to {mode.name}")
        elif mode == OperatingMode.WATCH_ONLY:
            if self.fallback_manager.current_level.value < 2:
                self.fallback_manager.escalate_fallback(f"Mode change to {mode.name}")
        elif mode == OperatingMode.HALTED:
            if self.fallback_manager.current_level.value < 4:
                self.fallback_manager.escalate_fallback(f"Mode change to {mode.name}")
    
    def assess_and_update_mode(self) -> OperatingMode:
        """
        Assess model health and update operating mode accordingly.
        
        Returns:
            Current operating mode after assessment
        """
        if self.model_health_monitor is None:
            logger.warning("Cannot assess health: model_health_monitor is None")
            return self.current_mode
        
        # Get health assessment
        health_metrics = self.model_health_monitor.assess_health()
        health_report = self.model_health_monitor.get_health_report()
        
        # Determine appropriate mode based on health
        new_mode = self._determine_operating_mode_from_health(health_metrics, health_report)
        
        # Apply mode change if needed
        if new_mode != self.current_mode:
            reason = self._get_mode_change_reason(new_mode, health_report)
            self.set_operating_mode(new_mode, reason)
        
        return self.current_mode

    def _determine_operating_mode_from_health(self, health_metrics, health_report) -> OperatingMode:
        """Determine the appropriate operating mode based on health metrics."""
        if health_metrics.status == "critical" or health_metrics.overall_health < 0.3:
            return OperatingMode.HALTED
        elif health_metrics.status == "warning" or health_metrics.overall_health < 0.7:
            # Multiple warnings suggest watch-only, few warnings suggest reduced mode
            if len(health_report.warnings) > 3:
                return OperatingMode.WATCH_ONLY
            else:
                return OperatingMode.REDUCED
        else:
            return OperatingMode.NORMAL

    def _get_mode_change_reason(self, new_mode: OperatingMode, health_report) -> str:
        """Generate a descriptive reason for the mode change."""
        if new_mode == OperatingMode.HALTED:
            return f"Critical health issues: {', '.join(health_report.critical_issues)}"
        elif new_mode == OperatingMode.WATCH_ONLY:
            return f"Multiple health warnings: {', '.join(health_report.warnings[:3])}..."
        elif new_mode == OperatingMode.REDUCED:
            return f"Health warnings: {', '.join(health_report.warnings)}"
        elif new_mode == OperatingMode.NORMAL:
            return "Model health restored to normal"
        else:
            return f"Mode change to {new_mode.name}"
    
    def attempt_recovery(self) -> bool:
        """
        Attempt to recover to a better operating mode.
        
        Uses exponential backoff to avoid rapid oscillation between modes.
        
        Returns:
            True if recovery was successful, False otherwise
        """
        if not self.recovery_config.enable_auto_recovery:
            return False
        
        # Check basic recovery conditions
        if not self._can_attempt_recovery():
            return False
        
        # Update recovery state and get target mode
        target_mode = self._prepare_recovery_attempt()
        
        # Attempt the actual recovery
        return self._execute_recovery(target_mode)

    def _can_attempt_recovery(self) -> bool:
        """Check if recovery can be attempted based on current conditions."""
        # Already in normal mode
        if self.current_mode == OperatingMode.NORMAL:
            return True
        
        # Exceeded max attempts
        if self.recovery_attempts >= self.recovery_config.max_recovery_attempts:
            logger.warning(
                "Maximum recovery attempts (%d) reached, manual intervention required",
                self.recovery_config.max_recovery_attempts
            )
            return False
        
        # Check timing constraints
        return self._check_recovery_timing()

    def _check_recovery_timing(self) -> bool:
        """Check if enough time has passed for recovery attempt."""
        if self.last_recovery_time is None:
            return True
        
        current_time = datetime.datetime.now()
        elapsed_seconds = (current_time - self.last_recovery_time).total_seconds()
        
        if elapsed_seconds < self.current_backoff_sec:
            logger.info(
                "Too soon to attempt recovery, waiting %.1f more seconds",
                self.current_backoff_sec - elapsed_seconds
            )
            return False
        
        # Check if recovery count should be reset
        hours_elapsed = elapsed_seconds / 3600
        if hours_elapsed > self.recovery_config.recovery_reset_hours:
            logger.info("Resetting recovery attempts counter due to time elapsed")
            self.recovery_attempts = 0
            self.current_backoff_sec = self.recovery_config.initial_backoff_sec
        
        return True

    def _prepare_recovery_attempt(self) -> OperatingMode:
        """Prepare for recovery attempt and return target mode."""
        self.recovery_attempts += 1
        self.last_recovery_time = datetime.datetime.now()
        
        # Try to recover to one level better
        target_mode = OperatingMode(max(0, self.current_mode.value - 1))
        
        logger.info(
            "Attempting recovery #%d from %s to %s mode",
            self.recovery_attempts,
            self.current_mode.name,
            target_mode.name
        )
        
        return target_mode

    def _execute_recovery(self, target_mode: OperatingMode) -> bool:
        """Execute the actual recovery attempt."""
        # Check if health allows recovery
        if not self._health_supports_recovery(target_mode):
            return self._handle_recovery_failure()
        
        # Attempt the mode change
        success = self.set_operating_mode(
            target_mode,
            f"Automatic recovery attempt #{self.recovery_attempts}"
        )
        
        if success:
            return self._handle_recovery_success(target_mode)
        else:
            return self._handle_recovery_failure()

    def _health_supports_recovery(self, target_mode: OperatingMode) -> bool:
        """Check if current health supports recovery to target mode."""
        if self.model_health_monitor is None:
            return False
        
        health_metrics = self.model_health_monitor.assess_health()
        
        health_thresholds = {
            OperatingMode.NORMAL: 0.8,
            OperatingMode.REDUCED: 0.6,
            OperatingMode.WATCH_ONLY: 0.4
        }
        
        return health_metrics.overall_health > health_thresholds.get(target_mode, 0.0)

    def _handle_recovery_success(self, target_mode: OperatingMode) -> bool:
        """Handle successful recovery."""
        logger.info("Recovery to %s mode successful", target_mode.name)
        
        if target_mode == OperatingMode.NORMAL:
            # Reset recovery state on full recovery
            self.recovery_attempts = 0
            self.current_backoff_sec = self.recovery_config.initial_backoff_sec
        else:
            # Increase backoff for partial recovery
            self._increase_backoff()
        
        return True

    def _handle_recovery_failure(self) -> bool:
        """Handle failed recovery attempt."""
        logger.warning("Recovery attempt failed, will try again later")
        self._increase_backoff()
        return False

    def _increase_backoff(self) -> None:
        """Increase backoff time for next recovery attempt."""
        self.current_backoff_sec = min(
            self.current_backoff_sec * self.recovery_config.backoff_factor,
            self.recovery_config.max_backoff_sec
        )
    
    def send_alert(self, title: str, message: str, severity: str = "warning") -> bool:
        """
        Send an alert to configured channels.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            
        Returns:
            True if alert was sent, False otherwise
        """
        # Check alert cooldown
        if not self._can_send_alert():
            return False
        
        # Create and store alert record
        alert_record = self._create_alert_record(title, message, severity)
        self._store_alert_record(alert_record)
        
        # Send to all configured channels
        self._send_to_all_channels(title, message, severity, alert_record)
        
        return True

    def _can_send_alert(self) -> bool:
        """Check if alert can be sent (cooldown check)."""
        if self.last_alert_time is None:
            return True
        
        current_time = datetime.datetime.now()
        elapsed_seconds = (current_time - self.last_alert_time).total_seconds()
        
        if elapsed_seconds < self.alert_config.min_alert_interval_sec:
            logger.debug("Alert cooldown active, skipping alert")
            return False
        
        return True

    def _create_alert_record(self, title: str, message: str, severity: str) -> Dict[str, Any]:
        """Create an alert record with metadata."""
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "title": title,
            "message": message,
            "severity": severity,
            "operating_mode": self.current_mode.name
        }

    def _store_alert_record(self, alert_record: Dict[str, Any]) -> None:
        """Store alert record in history."""
        self.alert_history.append(alert_record)
        self.last_alert_time = datetime.datetime.now()
        
        # Keep history bounded
        if len(self.alert_history) > MAX_HISTORY_SIZE:
            self.alert_history = self.alert_history[-MAX_HISTORY_SIZE:]

    def _send_to_all_channels(self, title: str, message: str, severity: str, 
                            alert_record: Dict[str, Any]) -> None:
        """Send alert to all configured channels."""
        self._send_console_alert(title, message, severity)
        self._send_file_alert(alert_record)
        self._send_webhook_alert(alert_record)
        self._send_email_alert(title, message, severity)
        self._send_prometheus_alert(title, severity)
        self._send_slack_alert(title, message, severity)

    def _send_console_alert(self, title: str, message: str, severity: str) -> None:
        """Send alert to console via logger."""
        if not self.alert_config.enable_console_alerts:
            return
        
        alert_msg = f"ALERT: {title} - {message}"
        
        if severity == "info":
            logger.info(alert_msg)
        elif severity == "warning":
            logger.warning(alert_msg)
        elif severity == "error":
            logger.error(alert_msg)
        elif severity == "critical":
            logger.critical(alert_msg)

    def _send_file_alert(self, alert_record: Dict[str, Any]) -> None:
        """Send alert to file."""
        if not (self.alert_config.enable_file_alerts and self.alert_config.alert_log_path):
            return
        
        try:
            with open(self.alert_config.alert_log_path, "a") as f:
                f.write(
                    f"{alert_record['timestamp']} [{alert_record['severity'].upper()}] "
                    f"[{alert_record['operating_mode']}] {alert_record['title']}: "
                    f"{alert_record['message']}\n"
                )
        except Exception as e:
            logger.error("Failed to write alert to file: %s", str(e))

    def _send_webhook_alert(self, alert_record: Dict[str, Any]) -> None:
        """Send alert via webhook."""
        if not (self.alert_config.enable_webhook_alerts and self.alert_config.webhook_url):
            return
        
        try:
            import requests
            requests.post(
                self.alert_config.webhook_url,
                json=alert_record,
                timeout=5
            )
        except Exception as e:
            logger.error("Failed to send webhook alert: %s", str(e))

    def _send_email_alert(self, title: str, message: str, severity: str) -> None:
        """Send alert via email."""
        if not (self.alert_config.enable_email_alerts and self.alert_config.email_recipients):
            return
        
        try:
            msg = EmailMessage()
            msg.set_content(self._format_email_content(message, severity))
            msg['Subject'] = f"[{severity.upper()}] {title}"
            msg['From'] = "alerts@example.com"
            msg['To'] = ", ".join(self.alert_config.email_recipients)
            
            # This would need proper configuration in production
            logger.info("Email alert would be sent to %s", self.alert_config.email_recipients)
        except Exception as e:
            logger.error("Failed to send email alert: %s", str(e))

    def _format_email_content(self, message: str, severity: str) -> str:
        """Format email content for alerts."""
        return f"""
Alert Details:
--------------
Severity: {severity.upper()}
Operating Mode: {self.current_mode.name}
Time: {datetime.datetime.now().isoformat()}

{message}

This is an automated alert from the AURA AI trading system.
        """

    def _send_prometheus_alert(self, title: str, severity: str) -> None:
        """Send alert to Prometheus."""
        if not (self.alert_config.enable_prometheus_alerts and self.alert_config.prometheus_gateway):
            return
        
        try:
            metrics = self._format_prometheus_metrics(title, severity)
            url = f"{self.alert_config.prometheus_gateway}/metrics/job/aura_trading_system"
            
            import requests
            requests.post(url, data=metrics, headers={"Content-Type": "text/plain"}, timeout=5)
            
            logger.info("Prometheus metrics pushed to gateway: %s", self.alert_config.prometheus_gateway)
        except Exception as e:
            logger.error("Failed to send Prometheus alert: %s", str(e))

    def _format_prometheus_metrics(self, title: str, severity: str) -> str:
        """Format Prometheus metrics for alert."""
        severity_value = {"info": 0, "warning": 1, "error": 2, "critical": 3}.get(severity, 1)
        mode_value = self.current_mode.value
        
        return f"""
# TYPE aura_alert_triggered gauge
# HELP aura_alert_triggered Alert triggered with severity and operating mode
aura_alert_triggered{{title="{title}", severity="{severity}", operating_mode="{self.current_mode.name}"}} 1
# TYPE aura_alert_severity gauge
# HELP aura_alert_severity Severity level of the alert (0=info, 1=warning, 2=error, 3=critical)
aura_alert_severity{{title="{title}"}} {severity_value}
# TYPE aura_operating_mode gauge
# HELP aura_operating_mode Current operating mode (0=normal, 1=reduced, 2=watch_only, 3=halted)
aura_operating_mode{{instance="aura_trading_system"}} {mode_value}
"""

    def _send_slack_alert(self, title: str, message: str, severity: str) -> None:
        """Send alert to Slack."""
        if not (self.alert_config.enable_slack_alerts and self.alert_config.slack_webhook_url):
            return
        
        try:
            payload = self._format_slack_payload(title, message, severity)
            
            import requests
            requests.post(self.alert_config.slack_webhook_url, json=payload, timeout=5)
            
            logger.info("Slack alert sent")
        except Exception as e:
            logger.error("Failed to send Slack alert: %s", str(e))

    def _format_slack_payload(self, title: str, message: str, severity: str) -> Dict[str, Any]:
        """Format Slack message payload."""
        color = {
            "info": "#36a64f",
            "warning": "#ffcc00", 
            "error": "#ff9900",
            "critical": "#ff0000"
        }.get(severity, "#ffcc00")
        
        return {
            "attachments": [
                {
                    "fallback": f"{title}: {message}",
                    "color": color,
                    "title": title,
                    "text": message,
                    "fields": [
                        {"title": "Severity", "value": severity.upper(), "short": True},
                        {"title": "Operating Mode", "value": self.current_mode.name, "short": True},
                        {"title": "Time", "value": datetime.datetime.now().isoformat(), "short": True}
                    ],
                    "footer": "AURA AI Trading System"
                }
            ]
        }
    
    def start_monitoring(self, interval_sec: float = 60.0) -> bool:
        """
        Start background monitoring thread.
        
        Args:
            interval_sec: Monitoring interval in seconds
            
        Returns:
            True if monitoring was started, False otherwise
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return False
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_sec,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Background monitoring started with interval %.1f seconds", interval_sec)
        return True
    
    def stop_monitoring(self) -> bool:
        """
        Stop background monitoring thread.
        
        Returns:
            True if monitoring was stopped, False otherwise
        """
        if not self.monitoring_active:
            logger.warning("Monitoring not active")
            return False
        
        self.monitoring_active = False
        if self.monitoring_thread is not None:
            self.monitoring_thread.join(timeout=5.0)
            self.monitoring_thread = None
        
        logger.info("Background monitoring stopped")
        return True
    
    def _monitoring_loop(self, interval_sec: float) -> None:
        """
        Background monitoring loop.
        
        Args:
            interval_sec: Monitoring interval in seconds
        """
        logger.info("Monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Assess and update mode
                self.assess_and_update_mode()
                
                # Attempt recovery if needed
                if self.current_mode != OperatingMode.NORMAL:
                    self.attempt_recovery()
                
                # Sleep for interval
                time.sleep(interval_sec)
            except Exception as e:
                logger.error("Error in monitoring loop: %s", str(e))
                time.sleep(interval_sec)
    
    def get_effectiveness_report(self) -> Dict[str, Any]:
        """
        Get effectiveness report for fallback activations.
        
        Returns:
            Dictionary with effectiveness metrics
        """
        if self.fallback_manager is None:
            return {"error": "No fallback manager available"}
        
        # Get fallback effectiveness report
        fallback_report = self.fallback_manager.get_effectiveness_report()
        
        # Convert to serializable format
        report = {}
        for level, effectiveness in fallback_report.items():
            report[level.name] = {
                "activation_count": effectiveness.activation_count,
                "success_count": effectiveness.success_count,
                "failure_count": effectiveness.failure_count,
                "avg_recovery_time_sec": effectiveness.avg_recovery_time_sec,
                "last_activation": effectiveness.last_activation.isoformat() if effectiveness.last_activation else None,
                "cumulative_reward": effectiveness.cumulative_reward
            }
        
        # Add operating mode history
        report["mode_history"] = self.mode_history[-10:]  # Last 10 mode changes
        
        # Add recovery attempts
        report["recovery_attempts"] = self.recovery_attempts
        report["last_recovery_time"] = self.last_recovery_time.isoformat() if self.last_recovery_time else None
        report["current_backoff_sec"] = self.current_backoff_sec
        
        return report
    
    def save_effectiveness_report(self, filepath: str) -> bool:
        """
        Save effectiveness report to file.
        
        Args:
            filepath: Path to save report
            
        Returns:
            True if report was saved, False otherwise
        """
        try:
            report = self.get_effectiveness_report()
            
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)
            
            logger.info("Effectiveness report saved to %s", filepath)
            return True
        except Exception as e:
            logger.error("Failed to save effectiveness report: %s", str(e))
            return False