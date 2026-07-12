"""Test script to verify the fix for option_pricer_half_time IndexError."""
import numpy as np
import sys

# Add current directory to path to import mmar
sys.path.insert(0, '/home/chris/Hermes-VPS-Workspace/coding-projects/MMAR-Options-Pricing')

from mmar import option_pricer_half_time, option_pricer


def test_original_issue():
    """Test the exact scenario described in the GitHub issue."""
    print("Testing original issue scenario...")
    
    days_to_expiration = 30
    T = days_to_expiration / 365
    
    # Create paths with the same shape the pipeline produces
    num_paths = 5
    paths = np.ones((num_paths, days_to_expiration)) * 100.0  # shape (5, 30)
    
    strike = 100.0
    r = 0.05
    option_type = "call"
    
    # This should NOT raise IndexError anymore
    try:
        price = option_pricer_half_time(paths, strike, r, T, option_type)
        print(f"✓ option_pricer_half_time returned: {price}")
        assert price >= 0, "Option price should be non-negative"
        print("✓ Price is valid (non-negative)")
    except IndexError as e:
        print(f"✗ FAILED: IndexError still occurs: {e}")
        return False
    
    return True


def test_half_time_index():
    """Verify that the index points to the actual midpoint."""
    print("\nTesting half-time index calculation...")
    
    days_to_expiration = 30
    T = days_to_expiration / 365
    
    paths = np.ones((5, days_to_expiration)) * 100.0
    
    # Expected: round(30 / 2) = 15
    expected_index = 15
    
    # We can't directly test the internal index, but we can verify
    # that it accesses the correct column by checking with different values
    paths[:, 15] = 150.0  # Set midpoint to 150
    
    strike = 100.0
    r = 0.0
    
    # For a call option, if S_T = 150 and strike = 100, payoff = 50
    price = option_pricer_half_time(paths, strike, r, T, "call")
    
    # With r=0, price = mean(payoff) = 50
    # With discounting, it should be close to 50 * exp(-0 * T) = 50
    print(f"✓ option_pricer_half_time with midpoint at 150: {price}")
    assert abs(price - 50.0) < 1.0, f"Expected price ~50, got {price}"
    print("✓ Correct column is being accessed")
    
    return True


def test_clamping_long_T():
    """Test that long T values are clamped to the last valid index."""
    print("\nTesting clamping for long T...")
    
    # Small number of columns
    days_to_expiration = 10
    T = 5.0  # 5 years, but we only have 10 columns
    
    paths = np.ones((5, days_to_expiration)) * 100.0
    
    strike = 100.0
    r = 0.05
    
    try:
        # This should clamp to index 9 (last column)
        price = option_pricer_half_time(paths, strike, r, T, "call")
        print(f"✓ option_pricer_half_time with long T returned: {price}")
        
        # Compare with option_pricer which uses the last column
        price_full = option_pricer(paths, strike, r, T, "call")
        print(f"  option_pricer (last column) returned: {price_full}")
        
        # They should be the same since we're clamped to the last column
        assert abs(price - price_full) < 0.01, "Prices should match when clamped to last column"
        print("✓ Correctly clamped to last valid column")
        
    except IndexError as e:
        print(f"✗ FAILED: IndexError with long T: {e}")
        return False
    
    return True


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")
    
    # Single day
    paths = np.array([[100.0]])  # shape (1, 1)
    T = 1 / 365
    
    try:
        price = option_pricer_half_time(paths, 100.0, 0.0, T, "call")
        print(f"✓ Single day case: {price}")
    except IndexError as e:
        print(f"✗ FAILED: Single day case: {e}")
        return False
    
    # Two days - half should be index 0 or 1
    paths = np.ones((5, 2)) * 100.0
    T = 2 / 365
    
    try:
        price = option_pricer_half_time(paths, 100.0, 0.0, T, "call")
        print(f"✓ Two days case: {price}")
    except IndexError as e:
        print(f"✗ FAILED: Two days case: {e}")
        return False
    
    return True


def test_put_option():
    """Test with put option type."""
    print("\nTesting put option...")
    
    days_to_expiration = 30
    T = days_to_expiration / 365
    paths = np.ones((5, days_to_expiration)) * 50.0  # All prices at 50
    
    try:
        price = option_pricer_half_time(paths, 100.0, 0.05, T, "put")
        print(f"✓ Put option price: {price}")
        assert price > 0, "Put should have positive value when strike > spot"
        print("✓ Put option works correctly")
    except IndexError as e:
        print(f"✗ FAILED: Put option: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing fix for option_pricer_half_time IndexError")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_original_issue()
    all_passed &= test_half_time_index()
    all_passed &= test_clamping_long_T()
    all_passed &= test_edge_cases()
    all_passed &= test_put_option()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)
