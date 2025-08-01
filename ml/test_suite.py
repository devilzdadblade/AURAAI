"""
ComprehensiveTestSuite for automated testing of RL model components.

This module provides a comprehensive testing framework for reinforcement learning
model components, including unit tests, integration tests, property-based tests,
and regression tests.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
import os
import json
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from collections import defaultdict
import unittest
import pytest
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class TestResults:
    """Results of a test suite."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time: float
    results: List[TestResult]
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of tests."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests


class ComprehensiveTestSuite:
    """
    Comprehensive testing framework for RL model components.
    
    This class provides methods for:
    - Running unit tests for individual components
    - Running integration tests for component interactions
    - Running property-based tests for RL invariants
    - Running regression tests against baseline models
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_data_dir: str = "tests/test_data",
        baseline_model_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the test suite.
        
        Args:
            model: The model to test
            test_data_dir: Directory containing test data
            baseline_model_path: Path to baseline model for regression testing
            logger: Logger instance
        """
        self.model = model
        self.test_data_dir = test_data_dir
        self.baseline_model_path = baseline_model_path
        self.logger = logger or logging.getLogger(__name__)
        
        # Create test data directory if it doesn't exist
        os.makedirs(test_data_dir, exist_ok=True)
        
        # Generate test cases
        self.test_cases = self._generate_test_cases()
    
    def _generate_test_cases(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate test cases for different test types.
        
        Returns:
            Dictionary of test cases organized by test type
        """
        test_cases = {
            "unit": self._generate_unit_test_cases(),
            "integration": self._generate_integration_test_cases(),
            "property": self._generate_property_test_cases(),
            "regression": self._generate_regression_test_cases()
        }
        
        return test_cases
    
    def _generate_unit_test_cases(self) -> Dict[str, Any]:
        """
        Generate unit test cases for individual components.
        
        Returns:
            Dictionary of unit test cases
        """
        # Create test cases for different input shapes and types
        test_cases = {}
        
        # Test case for basic functionality
        test_cases["basic_functionality"] = {
            "input": torch.randn(32, 64),  # Batch size 32, input size 64
            "expected_output_shape": (32, None),  # Output shape depends on model
            "description": "Basic functionality test with random input"
        }
        
        # Test case for edge cases
        test_cases["edge_cases"] = {
            "inputs": [
                torch.zeros(1, 64),  # All zeros
                torch.ones(1, 64),   # All ones
                torch.full((1, 64), 0.5),  # All 0.5
                torch.randn(1, 64) * 1000,  # Large values
                torch.randn(1, 64) * 0.001  # Small values
            ],
            "description": "Edge cases with extreme input values"
        }
        
        # Test case for gradient flow
        test_cases["gradient_flow"] = {
            "input": torch.randn(16, 64, requires_grad=True),
            "description": "Gradient flow through the model"
        }
        
        # Test case for different batch sizes
        test_cases["batch_sizes"] = {
            "inputs": [
                torch.randn(1, 64),    # Batch size 1
                torch.randn(16, 64),   # Batch size 16
                torch.randn(128, 64),  # Batch size 128
                torch.randn(512, 64)   # Batch size 512
            ],
            "description": "Different batch sizes"
        }
        
        return test_cases
    
    def _generate_integration_test_cases(self) -> Dict[str, Any]:
        """
        Generate integration test cases for component interactions.
        
        Returns:
            Dictionary of integration test cases
        """
        test_cases = {}
        
        # Test case for model components interaction
        test_cases["component_interaction"] = {
            "input": torch.randn(32, 64),
            "components_to_check": [
                "feature_extraction",
                "attention_mechanism",
                "value_stream",
                "advantage_stream"
            ],
            "description": "Interaction between model components"
        }
        
        # Test case for training loop integration
        test_cases["training_loop"] = {
            "batch_size": 32,
            "input_size": 64,
            "num_iterations": 10,
            "description": "Integration with training loop"
        }
        
        # Test case for optimizer integration
        test_cases["optimizer_integration"] = {
            "optimizers": ["Adam", "RMSprop", "SGD"],
            "learning_rates": [0.01, 0.001, 0.0001],
            "description": "Integration with different optimizers"
        }
        
        # Test case for loss function integration
        test_cases["loss_function_integration"] = {
            "loss_functions": ["MSE", "Huber", "SmoothL1"],
            "description": "Integration with different loss functions"
        }
        
        return test_cases
    
    def _generate_property_test_cases(self) -> Dict[str, Any]:
        """
        Generate property-based test cases for RL invariants.
        
        Returns:
            Dictionary of property test cases
        """
        test_cases = {}
        
        # Test case for Q-value bounds
        test_cases["q_value_bounds"] = {
            "num_samples": 100,
            "input_size": 64,
            "description": "Q-values should be bounded"
        }
        
        # Test case for advantage mean
        test_cases["advantage_mean"] = {
            "num_samples": 50,
            "input_size": 64,
            "description": "Advantage mean should be close to zero"
        }
        
        # Test case for Q-value ranking invariance
        test_cases["q_value_ranking"] = {
            "num_samples": 30,
            "input_size": 64,
            "description": "Q-value ranking should be invariant to constant shifts"
        }
        
        # Test case for exploration decay
        test_cases["exploration_decay"] = {
            "num_steps": 1000,
            "input_size": 64,
            "description": "Exploration should decay over time"
        }
        
        # Test case for Q-value convergence
        test_cases["q_value_convergence"] = {
            "num_iterations": 100,
            "batch_size": 32,
            "input_size": 64,
            "description": "Q-values should converge during training"
        }
        
        return test_cases
    
    def _generate_regression_test_cases(self) -> Dict[str, Any]:
        """
        Generate regression test cases against baseline models.
        
        Returns:
            Dictionary of regression test cases
        """
        test_cases = {}
        
        # Test case for performance regression
        test_cases["performance_regression"] = {
            "num_samples": 100,
            "input_size": 64,
            "metrics": ["inference_time", "memory_usage"],
            "description": "Performance should not regress compared to baseline"
        }
        
        # Test case for output consistency
        test_cases["output_consistency"] = {
            "num_samples": 50,
            "input_size": 64,
            "description": "Outputs should be consistent with baseline model"
        }
        
        # Test case for gradient consistency
        test_cases["gradient_consistency"] = {
            "num_samples": 30,
            "input_size": 64,
            "description": "Gradients should be consistent with baseline model"
        }
        
        # Test case for training convergence
        test_cases["training_convergence"] = {
            "num_iterations": 100,
            "batch_size": 32,
            "input_size": 64,
            "description": "Training convergence should be similar to baseline"
        }
        
        return test_cases
    
    def run_unit_tests(self) -> TestResults:
        """
        Run unit tests for individual components.

        Returns:
            TestResults object with test results
        """
        self.logger.info("Running unit tests...")
        start_time = time.time()

        results = []
        passed = 0
        failed = 0

        # Run each test and collect results
        test_methods = [
            self._test_basic_functionality,
            self._test_edge_cases,
            self._test_gradient_flow,
            self._test_batch_sizes
        ]
        for test_method in test_methods:
            test_results = test_method()
            for r in test_results:
                results.append(r)
                if r.passed:
                    passed += 1
                else:
                    failed += 1

        # Create test results
        total_time = time.time() - start_time
        test_results = TestResults(
            total_tests=passed + failed,
            passed_tests=passed,
            failed_tests=failed,
            execution_time=total_time,
            results=results
        )

        self.logger.info(f"Unit tests completed: {passed} passed, {failed} failed, {total_time:.2f}s elapsed")

        return test_results

    def _test_basic_functionality(self) -> List[TestResult]:
        results = []
        test_case = self.test_cases["unit"]["basic_functionality"]
        test_start = time.time()
        try:
            input_tensor = test_case["input"]
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, "forward") and callable(self.model.forward):
                    output = self.model(input_tensor)
                    if isinstance(output, tuple):
                        q_values = output[0]
                    else:
                        q_values = output
                    expected_shape = test_case["expected_output_shape"]
                    if expected_shape[1] is None:
                        assert q_values.shape[0] == expected_shape[0], f"Expected batch size {expected_shape[0]}, got {q_values.shape[0]}"
                    else:
                        assert q_values.shape == expected_shape, f"Expected shape {expected_shape}, got {q_values.shape}"
                    assert torch.isfinite(q_values).all(), "Output contains NaN or Inf values"
                    results.append(TestResult(
                        name="basic_functionality",
                        passed=True,
                        execution_time=time.time() - test_start,
                        details={"output_shape": q_values.shape}
                    ))
                else:
                    raise ValueError("Model does not have a callable forward method")
        except Exception as e:
            results.append(TestResult(
                name="basic_functionality",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            ))
            self.logger.error(f"Basic functionality test failed: {e}")
        return results

    def _test_edge_cases(self) -> List[TestResult]:
        results = []
        test_case = self.test_cases["unit"]["edge_cases"]
        for i, input_tensor in enumerate(test_case["inputs"]):
            test_name = f"edge_case_{i}"
            test_start = time.time()
            try:
                with torch.no_grad():
                    output = self.model(input_tensor)
                    if isinstance(output, tuple):
                        q_values = output[0]
                    else:
                        q_values = output
                    assert torch.isfinite(q_values).all(), "Output contains NaN or Inf values"
                    results.append(TestResult(
                        name=test_name,
                        passed=True,
                        execution_time=time.time() - test_start,
                        details={"input_type": f"Edge case {i}"}
                    ))
            except Exception as e:
                results.append(TestResult(
                    name=test_name,
                    passed=False,
                    execution_time=time.time() - test_start,
                    error_message=str(e)
                ))
                self.logger.error(f"Edge case {i} test failed: {e}")
        return results

    def _test_gradient_flow(self) -> List[TestResult]:
        results = []
        test_case = self.test_cases["unit"]["gradient_flow"]
        test_start = time.time()
        try:
            input_tensor = test_case["input"]
            self.model.train()
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                q_values = output[0]
            else:
                q_values = output
            loss = q_values.mean()
            loss.backward()
            assert input_tensor.grad is not None, "No gradients flowed back to input"
            assert not torch.all(input_tensor.grad == 0), "All input gradients are zero"
            has_grad = any(param.grad is not None for param in self.model.parameters())
            assert has_grad, "No model parameters have gradients"
            results.append(TestResult(
                name="gradient_flow",
                passed=True,
                execution_time=time.time() - test_start,
                details={"has_gradients": True}
            ))
        except Exception as e:
            results.append(TestResult(
                name="gradient_flow",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            ))
            self.logger.error(f"Gradient flow test failed: {e}")
        return results

    def _test_batch_sizes(self) -> List[TestResult]:
        results = []
        test_case = self.test_cases["unit"]["batch_sizes"]
        for i, input_tensor in enumerate(test_case["inputs"]):
            test_name = f"batch_size_{input_tensor.shape[0]}"
            test_start = time.time()
            try:
                self.model.eval()
                with torch.no_grad():
                    output = self.model(input_tensor)
                    if isinstance(output, tuple):
                        q_values = output[0]
                    else:
                        q_values = output
                    assert q_values.shape[0] == input_tensor.shape[0], f"Expected batch size {input_tensor.shape[0]}, got {q_values.shape[0]}"
                    assert torch.isfinite(q_values).all(), "Output contains NaN or Inf values"
                    results.append(TestResult(
                        name=test_name,
                        passed=True,
                        execution_time=time.time() - test_start,
                        details={"batch_size": input_tensor.shape[0]}
                    ))
            except Exception as e:
                results.append(TestResult(
                    name=test_name,
                    passed=False,
                    execution_time=time.time() - test_start,
                    error_message=str(e)
                ))
                self.logger.error(f"Batch size {input_tensor.shape[0]} test failed: {e}")
        return results
    
    def run_integration_tests(self) -> TestResults:
        """
        Run integration tests for component interactions.

        Returns:
            TestResults object with test results
        """
        self.logger.info("Running integration tests...")
        start_time = time.time()

        results = []
        passed = 0
        failed = 0

        # Run each integration test helper and collect results
        integration_helpers = [
            self._integration_component_interaction,
            self._integration_training_loop,
            self._integration_optimizer_integration,
            self._integration_loss_function_integration
        ]
        for helper in integration_helpers:
            helper_results = helper()
            for r in helper_results:
                results.append(r)
                if r.passed:
                    passed += 1
                else:
                    failed += 1

        # Create test results
        total_time = time.time() - start_time
        test_results = TestResults(
            total_tests=passed + failed,
            passed_tests=passed,
            failed_tests=failed,
            execution_time=total_time,
            results=results
        )

        self.logger.info(f"Integration tests completed: {passed} passed, {failed} failed, {total_time:.2f}s elapsed")

        return test_results

    def _integration_component_interaction(self) -> List[TestResult]:
        results = []
        test_case = self.test_cases["integration"]["component_interaction"]
        test_start = time.time()
        try:
            input_tensor = test_case["input"]
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
                assert isinstance(output, tuple) and len(output) == 2, "Expected tuple output with q_values and intermediates"
                _, intermediates = output
                for component in test_case["components_to_check"]:
                    found = any(component.lower() in key.lower() for key in intermediates.keys())
                    assert found, f"Component '{component}' not found in intermediates"
                results.append(TestResult(
                    name="component_interaction",
                    passed=True,
                    execution_time=time.time() - test_start,
                    details={"components_found": list(intermediates.keys())}
                ))
        except Exception as e:
            results.append(TestResult(
                name="component_interaction",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            ))
            self.logger.error(f"Component interaction test failed: {e}")
        return results

    def _integration_training_loop(self) -> List[TestResult]:
        results = []
        test_case = self.test_cases["integration"]["training_loop"]
        test_start = time.time()
        try:
            batch_size = test_case["batch_size"]
            input_size = test_case["input_size"]
            num_iterations = test_case["num_iterations"]
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=0.001,
                weight_decay=1e-4,
                eps=1e-8,
                betas=(0.9, 0.999)
            )
            self.model.train()
            losses = []
            for _ in range(num_iterations):
                input_tensor = torch.randn(batch_size, input_size)
                target = torch.randn(batch_size, 1)
                output = self.model(input_tensor)
                q_values = output[0] if isinstance(output, tuple) else output
                loss = torch.nn.functional.mse_loss(q_values.mean(dim=1, keepdim=True), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            assert all(np.isfinite(loss) for loss in losses), "Loss contains NaN or Inf values"
            results.append(TestResult(
                name="training_loop",
                passed=True,
                execution_time=time.time() - test_start,
                details={"losses": losses}
            ))
        except Exception as e:
            results.append(TestResult(
                name="training_loop",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            ))
            self.logger.error(f"Training loop test failed: {e}")
        return results

    def _integration_optimizer_integration(self) -> List[TestResult]:
        results = []
        test_case = self.test_cases["integration"]["optimizer_integration"]
        for optimizer_name in test_case["optimizers"]:
            for lr in test_case["learning_rates"]:
                test_name = f"optimizer_{optimizer_name}_lr_{lr}"
                test_start = time.time()
                try:
                    if optimizer_name == "Adam":
                        optimizer = torch.optim.Adam(
                            self.model.parameters(),
                            lr=lr,
                            weight_decay=1e-4,
                            eps=1e-8,
                            betas=(0.9, 0.999)
                        )
                    elif optimizer_name == "RMSprop":
                        optimizer = torch.optim.RMSprop(
                            self.model.parameters(),
                            lr=lr,
                            weight_decay=1e-4,
                            eps=1e-8,
                            alpha=0.99,
                            momentum=0.9
                        )
                    elif optimizer_name == "SGD":
                        optimizer = torch.optim.SGD(
                            self.model.parameters(),
                            lr=lr,
                            weight_decay=1e-4,
                            momentum=0.9,
                            nesterov=True
                        )
                    else:
                        raise ValueError(f"Unknown optimizer: {optimizer_name}")
                    self.model.train()
                    input_tensor = torch.randn(32, 64)
                    target = torch.randn(32, 1)
                    output = self.model(input_tensor)
                    q_values = output[0] if isinstance(output, tuple) else output
                    loss = torch.nn.functional.mse_loss(q_values.mean(dim=1, keepdim=True), target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    assert np.isfinite(loss.item()), "Loss contains NaN or Inf values"
                    results.append(TestResult(
                        name=test_name,
                        passed=True,
                        execution_time=time.time() - test_start,
                        details={"optimizer": optimizer_name, "learning_rate": lr, "loss": loss.item()}
                    ))
                except Exception as e:
                    results.append(TestResult(
                        name=test_name,
                        passed=False,
                        execution_time=time.time() - test_start,
                        error_message=str(e)
                    ))
                    self.logger.error(f"Optimizer {optimizer_name} with lr {lr} test failed: {e}")
        return results

    def _integration_loss_function_integration(self) -> List[TestResult]:
        results = []
        test_case = self.test_cases["integration"]["loss_function_integration"]
        for loss_name in test_case["loss_functions"]:
            test_name = f"loss_function_{loss_name}"
            test_start = time.time()
            try:
                self.model.train()
                input_tensor = torch.randn(32, 64)
                target = torch.randn(32, 1)
                output = self.model(input_tensor)
                q_values = output[0] if isinstance(output, tuple) else output
                if loss_name == "MSE":
                    loss = torch.nn.functional.mse_loss(q_values.mean(dim=1, keepdim=True), target)
                elif loss_name == "Huber":
                    loss = torch.nn.functional.huber_loss(q_values.mean(dim=1, keepdim=True), target)
                elif loss_name == "SmoothL1":
                    loss = torch.nn.functional.smooth_l1_loss(q_values.mean(dim=1, keepdim=True), target)
                else:
                    raise ValueError(f"Unknown loss function: {loss_name}")
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                assert np.isfinite(loss.item()), "Loss contains NaN or Inf values"
                results.append(TestResult(
                    name=test_name,
                    passed=True,
                    execution_time=time.time() - test_start,
                    details={"loss_function": loss_name, "loss": loss.item()}
                ))
            except Exception as e:
                results.append(TestResult(
                    name=test_name,
                    passed=False,
                    execution_time=time.time() - test_start,
                    error_message=str(e)
                ))
                self.logger.error(f"Loss function {loss_name} test failed: {e}")
        return results
    
    def run_property_tests(self) -> TestResults:
        """
        Run property-based tests for RL invariants.

        Returns:
            TestResults object with test results
        """
        self.logger.info("Running property-based tests...")
        start_time = time.time()

        results = []
        passed = 0
        failed = 0

        property_test_methods = [
            self._property_q_value_bounds,
            self._property_advantage_mean,
            self._property_q_value_ranking,
            self._property_exploration_decay,
            self._property_q_value_convergence
        ]

        for test_method in property_test_methods:
            test_result = test_method()
            results.append(test_result)
            if test_result.passed:
                passed += 1
            else:
                failed += 1

        total_time = time.time() - start_time
        test_results = TestResults(
            total_tests=passed + failed,
            passed_tests=passed,
            failed_tests=failed,
            execution_time=total_time,
            results=results
        )

        self.logger.info(f"Property tests completed: {passed} passed, {failed} failed, {total_time:.2f}s elapsed")

        return test_results

    def _property_q_value_bounds(self) -> TestResult:
        test_case = self.test_cases["property"]["q_value_bounds"]
        test_start = time.time()
        try:
            num_samples = test_case["num_samples"]
            input_size = test_case["input_size"]
            self.model.eval()
            inputs = [torch.randn(1, input_size) for _ in range(num_samples)]
            q_values_list = []
            with torch.no_grad():
                for input_tensor in inputs:
                    output = self.model(input_tensor)
                    q_values = output[0] if isinstance(output, tuple) else output
                    q_values_list.append(q_values)
            q_values_tensor = torch.cat(q_values_list, dim=0)
            q_min = q_values_tensor.min().item()
            q_max = q_values_tensor.max().item()
            q_range = q_max - q_min
            assert np.isfinite(q_min) and np.isfinite(q_max), "Q-values contain NaN or Inf"
            assert q_range < 1000, f"Q-value range too large: {q_range}"
            return TestResult(
                name="q_value_bounds",
                passed=True,
                execution_time=time.time() - test_start,
                details={"q_min": q_min, "q_max": q_max, "q_range": q_range}
            )
        except Exception as e:
            self.logger.error(f"Q-value bounds test failed: {e}")
            return TestResult(
                name="q_value_bounds",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            )

    def _property_advantage_mean(self) -> TestResult:
        test_case = self.test_cases["property"]["advantage_mean"]
        test_start = time.time()
        try:
            num_samples = test_case["num_samples"]
            input_size = test_case["input_size"]
            self.model.eval()
            inputs = [torch.randn(1, input_size) for _ in range(num_samples)]
            advantage_means = []
            with torch.no_grad():
                for input_tensor in inputs:
                    output = self.model(input_tensor)
                    if isinstance(output, tuple) and len(output) == 2:
                        _, intermediates = output
                        if "advantage" in intermediates:
                            advantage = intermediates["advantage"]
                            advantage_mean = advantage.mean().item()
                            advantage_means.append(advantage_mean)
            if advantage_means:
                avg_advantage_mean = np.mean(advantage_means)
                assert abs(avg_advantage_mean) < 0.1, f"Advantage mean too far from zero: {avg_advantage_mean}"
                return TestResult(
                    name="advantage_mean",
                    passed=True,
                    execution_time=time.time() - test_start,
                    details={"avg_advantage_mean": avg_advantage_mean}
                )
            else:
                self.logger.warning("Advantage mean test skipped: advantage not available in intermediates")
                return TestResult(
                    name="advantage_mean",
                    passed=True,
                    execution_time=time.time() - test_start,
                    details={"skipped": True, "reason": "Advantage not available"}
                )
        except Exception as e:
            self.logger.error(f"Advantage mean test failed: {e}")
            return TestResult(
                name="advantage_mean",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            )

    def _property_q_value_ranking(self) -> TestResult:
        test_case = self.test_cases["property"]["q_value_ranking"]
        test_start = time.time()
        try:
            num_samples = test_case["num_samples"]
            input_size = test_case["input_size"]
            self.model.eval()
            inputs = [torch.randn(1, input_size) for _ in range(num_samples)]
            with torch.no_grad():
                rankings_consistent = True
                for input_tensor in inputs:
                    output = self.model(input_tensor)
                    q_values = output[0] if isinstance(output, tuple) else output
                    _, original_indices = torch.sort(q_values, descending=True)
                    shifted_input = input_tensor + 0.1
                    output = self.model(shifted_input)
                    shifted_q_values = output[0] if isinstance(output, tuple) else output
                    _, shifted_indices = torch.sort(shifted_q_values, descending=True)
                    if original_indices[0, 0] != shifted_indices[0, 0]:
                        rankings_consistent = False
                        break
            assert rankings_consistent, "Q-value ranking not invariant to input shifts"
            return TestResult(
                name="q_value_ranking",
                passed=True,
                execution_time=time.time() - test_start,
                details={"rankings_consistent": rankings_consistent}
            )
        except Exception as e:
            self.logger.error(f"Q-value ranking test failed: {e}")
            return TestResult(
                name="q_value_ranking",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            )

    def _property_exploration_decay(self) -> TestResult:
        test_case = self.test_cases["property"]["exploration_decay"]
        test_start = time.time()
        try:
            num_steps = test_case["num_steps"]
            input_size = test_case["input_size"]
            has_noisy_layers = hasattr(self.model, "reset_noise") and callable(self.model.reset_noise)
            if has_noisy_layers:
                self.model.train()
                input_tensor = torch.randn(1, input_size)
                q_values_std_list = []
                for _ in range(num_steps):
                    self.model.reset_noise()
                    output = self.model(input_tensor)
                    q_values = output[0] if isinstance(output, tuple) else output
                    q_values_std = q_values.std().item()
                    q_values_std_list.append(q_values_std)
                assert np.mean(q_values_std_list) > 0, "No exploration variance detected"
                return TestResult(
                    name="exploration_decay",
                    passed=True,
                    execution_time=time.time() - test_start,
                    details={"mean_std": np.mean(q_values_std_list)}
                )
            else:
                self.logger.warning("Exploration decay test skipped: model doesn't have noisy layers")
                return TestResult(
                    name="exploration_decay",
                    passed=True,
                    execution_time=time.time() - test_start,
                    details={"skipped": True, "reason": "No noisy layers"}
                )
        except Exception as e:
            self.logger.error(f"Exploration decay test failed: {e}")
            return TestResult(
                name="exploration_decay",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            )

    def _property_q_value_convergence(self) -> TestResult:
        test_case = self.test_cases["property"]["q_value_convergence"]
        test_start = time.time()
        try:
            num_iterations = test_case["num_iterations"]
            batch_size = test_case["batch_size"]
            input_size = test_case["input_size"]
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
            input_tensor = torch.randn(batch_size, input_size)
            target = torch.randn(batch_size, 1)
            losses = []
            for _ in range(num_iterations):
                output = self.model(input_tensor)
                q_values = output[0] if isinstance(output, tuple) else output
                loss = torch.nn.functional.mse_loss(q_values.mean(dim=1, keepdim=True), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            first_loss = losses[0]
            last_loss = losses[-1]
            assert last_loss < first_loss, f"Loss did not decrease: {first_loss} -> {last_loss}"
            return TestResult(
                name="q_value_convergence",
                passed=True,
                execution_time=time.time() - test_start,
                details={"first_loss": first_loss, "last_loss": last_loss, "improvement": first_loss - last_loss}
            )
        except Exception as e:
            self.logger.error(f"Q-value convergence test failed: {e}")
            return TestResult(
                name="q_value_convergence",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            )
    
    def run_regression_tests(self, baseline_model: Optional[nn.Module] = None) -> TestResults:
        """
        Run regression tests against baseline models.

        Args:
            baseline_model: Optional baseline model to compare against

        Returns:
            TestResults object with test results
        """
        self.logger.info("Running regression tests...")
        start_time = time.time()

        results = []
        passed = 0
        failed = 0

        # Load baseline model if not provided
        baseline_model = self._load_baseline_model(baseline_model)
        if baseline_model is None:
            self.logger.warning("Regression tests skipped: no baseline model available")
            return TestResults(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                execution_time=time.time() - start_time,
                results=[]
            )

        # Run each regression test helper and collect results
        regression_helpers = [
            self._regression_performance,
            self._regression_output_consistency,
            self._regression_gradient_consistency,
            self._regression_training_convergence
        ]
        for helper in regression_helpers:
            result = helper(baseline_model)
            results.append(result)
            if result.passed:
                passed += 1
            else:
                failed += 1

        total_time = time.time() - start_time
        test_results = TestResults(
            total_tests=passed + failed,
            passed_tests=passed,
            failed_tests=failed,
            execution_time=total_time,
            results=results
        )

        self.logger.info(f"Regression tests completed: {passed} passed, {failed} failed, {total_time:.2f}s elapsed")

        return test_results

    def _load_baseline_model(self, baseline_model: Optional[nn.Module]) -> Optional[nn.Module]:
        if baseline_model is None and self.baseline_model_path:
            try:
                baseline_model = torch.load(self.baseline_model_path)
                self.logger.info(f"Loaded baseline model from {self.baseline_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load baseline model: {e}")
                baseline_model = None
        return baseline_model

    def _regression_performance(self, baseline_model: nn.Module) -> TestResult:
        test_case = self.test_cases["regression"]["performance_regression"]
        test_start = time.time()
        try:
            num_samples = test_case["num_samples"]
            input_size = test_case["input_size"]
            inputs = [torch.randn(1, input_size) for _ in range(num_samples)]
            self.model.eval()
            baseline_model.eval()
            current_times = []
            with torch.no_grad():
                for input_tensor in inputs:
                    start = time.time()
                    _ = self.model(input_tensor)
                    end = time.time()
                    current_times.append(end - start)
            current_avg_time = np.mean(current_times)
            baseline_times = []
            with torch.no_grad():
                for input_tensor in inputs:
                    start = time.time()
                    _ = baseline_model(input_tensor)
                    end = time.time()
                    baseline_times.append(end - start)
            baseline_avg_time = np.mean(baseline_times)
            time_ratio = current_avg_time / baseline_avg_time if baseline_avg_time > 0 else float('inf')
            assert time_ratio < 1.5, f"Current model is {time_ratio:.2f}x slower than baseline"
            return TestResult(
                name="performance_regression",
                passed=True,
                execution_time=time.time() - test_start,
                details={
                    "current_avg_time_ms": current_avg_time * 1000,
                    "baseline_avg_time_ms": baseline_avg_time * 1000,
                    "time_ratio": time_ratio
                }
            )
        except Exception as e:
            self.logger.error(f"Performance regression test failed: {e}")
            return TestResult(
                name="performance_regression",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            )

    def _regression_output_consistency(self, baseline_model: nn.Module) -> TestResult:
        test_case = self.test_cases["regression"]["output_consistency"]
        test_start = time.time()
        try:
            num_samples = test_case["num_samples"]
            input_size = test_case["input_size"]
            inputs = [torch.randn(1, input_size) for _ in range(num_samples)]
            self.model.eval()
            baseline_model.eval()
            with torch.no_grad():
                max_diff = 0
                for input_tensor in inputs:
                    current_output = self.model(input_tensor)
                    current_q_values = current_output[0] if isinstance(current_output, tuple) else current_output
                    baseline_output = baseline_model(input_tensor)
                    baseline_q_values = baseline_output[0] if isinstance(baseline_output, tuple) else baseline_output
                    if current_q_values.shape != baseline_q_values.shape:
                        raise ValueError(f"Output shape mismatch: {current_q_values.shape} vs {baseline_q_values.shape}")
                    diff = torch.abs(current_q_values - baseline_q_values).max().item()
                    max_diff = max(max_diff, diff)
            assert max_diff < 10.0, f"Output difference too large: {max_diff}"
            return TestResult(
                name="output_consistency",
                passed=True,
                execution_time=time.time() - test_start,
                details={"max_diff": max_diff}
            )
        except Exception as e:
            self.logger.error(f"Output consistency test failed: {e}")
            return TestResult(
                name="output_consistency",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            )

    def _regression_gradient_consistency(self, baseline_model: nn.Module) -> TestResult:
        test_case = self.test_cases["regression"]["gradient_consistency"]
        test_start = time.time()
        try:
            num_samples = test_case["num_samples"]
            input_size = test_case["input_size"]
            inputs = [torch.randn(1, input_size, requires_grad=True) for _ in range(num_samples)]
            self.model.train()
            baseline_model.train()
            max_grad_diff = 0
            for input_tensor in inputs:
                current_input = input_tensor.clone().detach().requires_grad_(True)
                current_output = self.model(current_input)
                current_q_values = current_output[0] if isinstance(current_output, tuple) else current_output
                current_loss = current_q_values.mean()
                current_loss.backward()
                current_grad = current_input.grad.clone()
                baseline_input = input_tensor.clone().detach().requires_grad_(True)
                baseline_output = baseline_model(baseline_input)
                baseline_q_values = baseline_output[0] if isinstance(baseline_output, tuple) else baseline_output
                baseline_loss = baseline_q_values.mean()
                baseline_loss.backward()
                baseline_grad = baseline_input.grad.clone()
                if current_grad.shape == baseline_grad.shape:
                    grad_diff = torch.abs(current_grad - baseline_grad).max().item()
                    max_grad_diff = max(max_grad_diff, grad_diff)
            assert max_grad_diff < 100.0, f"Gradient difference too large: {max_grad_diff}"
            return TestResult(
                name="gradient_consistency",
                passed=True,
                execution_time=time.time() - test_start,
                details={"max_grad_diff": max_grad_diff}
            )
        except Exception as e:
            self.logger.error(f"Gradient consistency test failed: {e}")
            return TestResult(
                name="gradient_consistency",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            )

    def _regression_training_convergence(self, baseline_model: nn.Module) -> TestResult:
        test_case = self.test_cases["regression"]["training_convergence"]
        test_start = time.time()
        try:
            num_iterations = test_case["num_iterations"]
            batch_size = test_case["batch_size"]
            input_size = test_case["input_size"]
            input_tensor = torch.randn(batch_size, input_size)
            target = torch.randn(batch_size, 1)
            self.model.train()
            current_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
            current_losses = []
            for _ in range(num_iterations):
                output = self.model(input_tensor)
                q_values = output[0] if isinstance(output, tuple) else output
                loss = torch.nn.functional.mse_loss(q_values.mean(dim=1, keepdim=True), target)
                current_optimizer.zero_grad()
                loss.backward()
                current_optimizer.step()
                current_losses.append(loss.item())
            baseline_model.train()
            baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001, weight_decay=1e-4)
            baseline_losses = []
            for _ in range(num_iterations):
                output = baseline_model(input_tensor)
                q_values = output[0] if isinstance(output, tuple) else output
                loss = torch.nn.functional.mse_loss(q_values.mean(dim=1, keepdim=True), target)
                baseline_optimizer.zero_grad()
                loss.backward()
                baseline_optimizer.step()
                baseline_losses.append(loss.item())
            current_improvement = current_losses[0] - current_losses[-1]
            baseline_improvement = baseline_losses[0] - baseline_losses[-1]
            convergence_ratio = current_improvement / baseline_improvement if baseline_improvement > 0 else float('inf')
            assert convergence_ratio > 0.5, f"Current model converges {convergence_ratio:.2f}x worse than baseline"
            return TestResult(
                name="training_convergence",
                passed=True,
                execution_time=time.time() - test_start,
                details={
                    "current_improvement": current_improvement,
                    "baseline_improvement": baseline_improvement,
                    "convergence_ratio": convergence_ratio
                }
            )
        except Exception as e:
            self.logger.error(f"Training convergence test failed: {e}")
            return TestResult(
                name="training_convergence",
                passed=False,
                execution_time=time.time() - test_start,
                error_message=str(e)
            )
    
    def run_all_tests(self) -> Dict[str, TestResults]:
        """
        Run all tests.
        
        Returns:
            Dictionary of test results by test type
        """
        self.logger.info("Running all tests...")
        
        results = {
            "unit": self.run_unit_tests(),
            "integration": self.run_integration_tests(),
            "property": self.run_property_tests(),
            "regression": self.run_regression_tests()
        }
        
        # Log summary
        total_tests = sum(r.total_tests for r in results.values())
        total_passed = sum(r.passed_tests for r in results.values())
        total_failed = sum(r.failed_tests for r in results.values())
        total_time = sum(r.execution_time for r in results.values())
        
        self.logger.info(f"All tests completed: {total_passed}/{total_tests} passed, {total_failed} failed, {total_time:.2f}s elapsed")
        
        return results
    
    def save_results(self, results: Dict[str, TestResults], filename: str = "test_results.json"):
        """
        Save test results to a file.
        
        Args:
            results: Dictionary of test results by test type
            filename: Name of the file to save results to
        """
        # Convert results to serializable format
        serializable_results = {}
        
        for test_type, test_results in results.items():
            serializable_results[test_type] = {
                "total_tests": test_results.total_tests,
                "passed_tests": test_results.passed_tests,
                "failed_tests": test_results.failed_tests,
                "execution_time": test_results.execution_time,
                "success_rate": test_results.success_rate,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "execution_time": r.execution_time,
                        "error_message": r.error_message,
                        "details": r.details
                    }
                    for r in test_results.results
                ]
            }
        
        # Save to file
        filepath = os.path.join(self.test_data_dir, filename)
        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Test results saved to {filepath}")