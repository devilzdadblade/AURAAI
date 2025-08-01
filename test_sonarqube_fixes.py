#!/usr/bin/env python3
"""
Test script to verify SonarQube fixes for JulesGemini
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pattern_detector():
    """Test PatternDetector import and basic functionality"""
    try:
        from src.data.pattern_detector import PatternDetector
        
        # Create test instance
        detector = PatternDetector()
        print("✅ PatternDetector import successful")
        
        # Test with empty data - should return empty list
        import pandas as pd
        empty_data = pd.DataFrame()
        result = detector.detect_patterns(empty_data)
        assert isinstance(result, list)
        print("✅ PatternDetector.detect_patterns() with empty data works")
        
        # Test with minimal data
        test_data = pd.DataFrame({
            'High': [100, 101, 102],
            'Low': [99, 100, 101], 
            'Close': [100.5, 101.5, 102.5]
        })
        result = detector.detect_patterns(test_data)
        assert isinstance(result, list)
        print("✅ PatternDetector.detect_patterns() with test data works")
        
        return True
        
    except Exception as e:
        print(f"❌ PatternDetector test failed: {e}")
        return False

def test_market_regime():
    """Test MarketRegimeDetector import and basic functionality"""
    try:
        from src.data.market_regime import MarketRegimeDetector
        
        # Create test instance
        detector = MarketRegimeDetector()
        print("✅ MarketRegimeDetector import successful")
        
        # Test with minimal data
        import pandas as pd
        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        })
        result = detector.detect_regime(test_data)
        assert result in ['bull', 'bear', 'sideways']
        print(f"✅ MarketRegimeDetector.detect_regime() works, result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ MarketRegimeDetector test failed: {e}")
        return False

def test_position_sizing():
    """Test PositionSizer import and basic functionality"""
    try:
        from src.rules.position_sizing import PositionSizer
        
        # Create test instance
        sizer = PositionSizer()
        print("✅ PositionSizer import successful")
        
        # Test kelly criterion
        result = sizer.kelly_criterion(0.6, 1.5, 1.0)
        assert isinstance(result, float)
        print(f"✅ PositionSizer.kelly_criterion() works, result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ PositionSizer test failed: {e}")
        return False

def test_filters():
    """Test filters import and functionality"""
    try:
        from src.rules.filters import filter_avoid_market_open, filter_avoid_time_range
        from datetime import datetime
        
        print("✅ Filters import successful")
        
        # Test filter_avoid_time_range
        current_time = datetime(2023, 1, 1, 10, 30)
        params = {'start_h': 9, 'start_m': 15, 'end_h': 11, 'end_m': 0}
        result = filter_avoid_time_range(current_time, params)
        assert isinstance(result, bool)
        print(f"✅ filter_avoid_time_range() works, result: {result}")
        
        # Test filter_avoid_market_open 
        params = {'minutes': 30}
        result = filter_avoid_market_open(current_time, params)
        assert isinstance(result, bool)
        print(f"✅ filter_avoid_market_open() works, result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Filters test failed: {e}")
        return False

def test_strategy_imports():
    """Test strategy module imports"""
    try:
        from src.strategy import __all__
        print(f"✅ Strategy module imports successful, available: {__all__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy imports test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 Testing SonarQube fixes for JulesGemini...\n")
    
    tests = [
        ("Pattern Detector", test_pattern_detector),
        ("Market Regime", test_market_regime),
        ("Position Sizing", test_position_sizing),
        ("Filters", test_filters),
        ("Strategy Imports", test_strategy_imports),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                failed += 1
                print(f"❌ {test_name} test failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print("\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"🎯 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All SonarQube fixes verified successfully!")
    else:
        print(f"\n⚠️  {failed} tests failed - review the issues above")

if __name__ == "__main__":
    main()
