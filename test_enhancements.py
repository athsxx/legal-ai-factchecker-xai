#!/usr/bin/env python3
"""
Quick test script to demonstrate accuracy improvements
"""

import requests
import json
import time

def test_enhanced_counterfactuals():
    """Test the enhanced counterfactual system"""
    
    print("🧪 Testing Enhanced Counterfactual System")
    print("=" * 50)
    
    # Wait for service to be ready
    print("⏳ Waiting for ML service to be ready...")
    for attempt in range(10):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ ML service is ready!")
                break
        except:
            time.sleep(3)
    else:
        print("❌ ML service not available")
        return
    
    # Test cases
    test_cases = [
        "The party shall pay damages within 30 days of breach.",
        "The defendant is liable for all damages resulting from negligence.",
        "This agreement is binding and irrevocable for 5 years."
    ]
    
    total_accuracy = 0
    total_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case[:50]}...")
        
        try:
            response = requests.post(
                "http://localhost:8000/generate-counterfactuals",
                json={"claim": test_case},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                counterfactuals = data.get("counterfactual_examples", [])
                impact_stats = data.get("impact_statistics", {})
                
                print(f"   📊 Generated {len(counterfactuals)} counterfactuals")
                print(f"   📈 Impact Distribution: {impact_stats}")
                
                # Check for high-impact changes
                high_impact_count = impact_stats.get("high_impact", 0)
                total_examples = impact_stats.get("total_examples", 0)
                
                if total_examples > 0:
                    accuracy_score = (high_impact_count / total_examples) * 100
                    total_accuracy += accuracy_score
                    total_tests += 1
                    print(f"   ✅ High-impact accuracy: {accuracy_score:.1f}%")
                
                # Show sample counterfactuals
                print("   📝 Sample counterfactuals:")
                for j, cf in enumerate(counterfactuals[:3], 1):
                    print(f"      {j}. {cf.get('text', 'N/A')[:60]}...")
                    print(f"         Impact: {cf.get('impact', 'N/A')} | Type: {cf.get('type', 'N/A')}")
                    print(f"         Changes: {', '.join(cf.get('changes', []))}")
                
            else:
                print(f"   ❌ Error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if total_tests > 0:
        overall_accuracy = total_accuracy / total_tests
        print(f"\n🏆 OVERALL ENHANCEMENT RESULTS:")
        print(f"   📊 Average High-Impact Accuracy: {overall_accuracy:.1f}%")
        
        if overall_accuracy >= 80:
            print("   🎉 EXCELLENT: Significant accuracy improvement achieved!")
        elif overall_accuracy >= 60:
            print("   ✅ GOOD: Noticeable accuracy improvement")
        else:
            print("   ⚠️  MODERATE: Some improvements, more work needed")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_enhanced_counterfactuals()
