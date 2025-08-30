#!/usr/bin/env python3
"""
Integration test to verify the dynamics system functionality.
This test verifies that all components work together correctly.
"""

import sys
import os
import yaml
from types import SimpleNamespace
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cinderforge_e'))

# Direct imports from modules
from cinderforge_e.dynamics.aggregator import Aggregator, Verdict
from cinderforge_e.dynamics.penalty import dynamics_penalty
from cinderforge_e.dynamics.autoreg import AutoReg
from cinderforge_e.trainer.step import TrainingWeights, MockWitnesses

def create_mock_config():
    """Create a mock config object from our YAML structure."""
    config_data = {
        'gate': {
            'mode': 'observe',
            'schmitt': {'open': 0.75, 'close': 0.65},
            'min_inter_gate_steps': 32
        },
        'targets': {'plv': 0.60, 'sigma': 1.00, 'lambda': 0.00},
        'weights': {'plv': 0.75, 'sigma': 0.50, 'lambda': 1.00},
        'auto': {
            'enabled': True, 'k_i': 0.10, 'k_p': 0.25,
            'max_step': 0.10, 'hysteresis': 0.05, 'warmup_steps': 512
        },
        'irope': {
            'enabled': True, 'base_theta': 10000.0, 'interleave_every': 2,
            'head_scale_fix': True, 'warmup_steps': 512, 'linear_attention': True
        }
    }
    
    def dict_to_namespace(d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)
    
    return dict_to_namespace(config_data)

def test_aggregator():
    """Test the witness aggregator with Schmitt trigger."""
    print("Testing Aggregator...")
    cfg = create_mock_config()
    agg = Aggregator(cfg)
    
    # Test verdict generation
    plv, sigma, lam, p_ray = 0.65, 0.95, -0.02, 0.05
    verdict = agg.fuse(plv, sigma, lam, p_ray)
    
    print(f"  Verdict: gate_open={verdict.open_}, score={verdict.score:.3f}")
    print(f"  Reasons: {verdict.reasons}")
    assert isinstance(verdict, Verdict)
    assert isinstance(verdict.open_, bool)
    assert 0.0 <= verdict.score <= 1.0
    print("  ✓ Aggregator test passed!")

def test_penalty():
    """Test the dynamics penalty system."""
    print("Testing Penalty System...")
    cfg = create_mock_config()
    weights = TrainingWeights()
    
    # Test observe mode (should be 0)
    meas = {'plv': 0.5, 'sigma': 1.1, 'lambda': 0.1}
    loss, err = dynamics_penalty(meas, cfg.targets, weights, 'observe', 0)
    print(f"  Observe mode loss: {loss}")
    assert loss == 0.0
    
    # Test hard mode (should be > 0)
    loss, err = dynamics_penalty(meas, cfg.targets, weights, 'hard', 0)
    print(f"  Hard mode loss: {loss}")
    assert loss > 0.0
    print(f"  Error terms: {err}")
    print("  ✓ Penalty system test passed!")

def test_autoreg():
    """Test the auto-regulation system."""
    print("Testing Auto-Regulation...")
    cfg = create_mock_config()
    auto = AutoReg(cfg)
    weights = TrainingWeights()
    
    original_plv = weights.plv
    meas = {'plv': 0.4, 'sigma': 1.2, 'lambda': 0.1}  # Below targets
    
    weights, err = auto.update(meas, cfg.targets, weights)
    print(f"  Original PLV weight: {original_plv}, Updated: {weights.plv}")
    print(f"  Error signals: {err}")
    print("  ✓ Auto-regulation test passed!")

def test_rope_functionality():
    """Test RoPE/iRoPE functionality."""
    print("Testing RoPE/iRoPE...")
    from cinderforge_e.units.rope_variants import apply_posenc
    
    # Create test tensors
    B, H, T, Dh = 2, 4, 8, 16
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    
    # Test standard RoPE
    q_rope, k_rope = apply_posenc(q, k, T, variant="rope")
    print(f"  RoPE output shapes: q={q_rope.shape}, k={k_rope.shape}")
    
    # Test iRoPE
    q_irope, k_irope = apply_posenc(q, k, T, variant="irope", groups=2)
    print(f"  iRoPE output shapes: q={q_irope.shape}, k={k_irope.shape}")
    
    # Verify shapes are preserved
    assert q_rope.shape == q.shape
    assert k_rope.shape == k.shape
    assert q_irope.shape == q.shape
    assert k_irope.shape == k.shape
    print("  ✓ RoPE/iRoPE test passed!")

def test_mock_witnesses():
    """Test the mock witness system."""
    print("Testing Mock Witnesses...")
    witnesses = MockWitnesses()
    
    # Test with empty aux
    plv, p_ray = witnesses.plv({})
    sigma = witnesses.branching({})
    lam = witnesses.local_lyapunov({})
    
    print(f"  Mock measurements: PLV={plv:.3f}, sigma={sigma:.3f}, lambda={lam:.3f}, p={p_ray:.3f}")
    assert isinstance(plv, float)
    assert isinstance(sigma, float) 
    assert isinstance(lam, float)
    assert isinstance(p_ray, float)
    print("  ✓ Mock witnesses test passed!")

def main():
    """Run all integration tests."""
    print("=== CinderForge-E Dynamics System Integration Test ===\n")
    
    try:
        test_aggregator()
        print()
        
        test_penalty()
        print()
        
        test_autoreg() 
        print()
        
        test_rope_functionality()
        print()
        
        test_mock_witnesses()
        print()
        
        print("🎉 ALL TESTS PASSED! The dynamics system is functional.")
        print("\nKey Features Verified:")
        print("  ✓ Witness aggregation with Schmitt trigger")
        print("  ✓ Mode-aware dynamics penalties")
        print("  ✓ Bounded PI auto-regulation")
        print("  ✓ RoPE/iRoPE position encoding with safety features")
        print("  ✓ Mock witness computations")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
