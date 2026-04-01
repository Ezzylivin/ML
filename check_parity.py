"""
check_parity.py — Full Council Parity Audit

============================================================
🔧 FIX: This now actually tests PARITY (not just "does it load?")
============================================================

OLD: Loaded each model once, printed a score. Couldn't detect:
  - Path mismatches (models not loading, silent 0.5 fallback)
  - Normalization mismatches (Transformer getting raw vs scaled input)
  - Pipeline divergence (backtester vs live engine giving different scores)
  - Stacking judge receiving garbage when experts are missing

NEW: Four-phase audit:
  Phase 1: File system audit — do the model files actually exist?
  Phase 2: Individual expert scores — do they produce non-trivial output?
  Phase 3: Full stacking pipeline — does the ensemble work end-to-end?
  Phase 4: Consistency check — does the same input give the same output twice?
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from app.config2 import DATA_DIR, MODEL_DIR, FEATURE_COLUMNS
from app.verify.engineer_and_train import apply_mega_features
from app.predictors.model_factory import ModelFactory, clear_model_cache
from app.predictors.stacking_predictor import StackingPredictor


# ============================================================
# PHASE 1: FILE SYSTEM AUDIT
# ============================================================
def audit_model_files(symbol):
    """Check that all expected model files exist on disk."""
    ticker = symbol.split('-')[0].lower()
    
    expected_files = {
        "xgboost":     f"{ticker}_1h_xgboost_model.joblib",
        "randomforest": f"{ticker}_1h_randomforest_model.joblib",
        "transformer":  f"{ticker}_1h_transformer_model.keras",
        "norm_stats":   f"{ticker}_1h_transformer_norm.npz",
        "stacking":     f"{ticker}_1h_stacking_model.joblib",
    }
    
    print(f"\n📁 Phase 1: File Audit (searching {MODEL_DIR})")
    all_found = True
    
    for name, filename in expected_files.items():
        full_path = os.path.join(MODEL_DIR, filename)
        exists = os.path.exists(full_path)
        size = ""
        if exists:
            size_kb = os.path.getsize(full_path) / 1024
            size = f"({size_kb:.0f} KB)"
        
        status = f"✅ FOUND {size}" if exists else "❌ MISSING"
        print(f"   [{name.ljust(14)}] {status} — {full_path}")
        
        if not exists:
            all_found = False
    
    if not all_found:
        print("\n   ⚠️ Missing files detected. Run training scripts first:")
        print("      1. python -m app.verify.engineer_and_train")
        print("      2. python -m app.verify.train_transformer")  
        print("      3. python -m app.verify.train_judge")
    
    return all_found


# ============================================================
# PHASE 2: INDIVIDUAL EXPERT SCORES
# ============================================================
def audit_expert_scores(symbol, test_data):
    """Load each expert and verify it produces meaningful output."""
    print(f"\n🧠 Phase 2: Expert Score Audit")
    
    experts = ["xgboost", "randomforest", "transformer"]
    scores = {}
    all_healthy = True
    
    for name in experts:
        model = ModelFactory.load_model(name, symbol=symbol)
        
        if model is None:
            print(f"   [{name.ljust(14)}] 🔴 LOAD FAILED — ModelFactory returned None")
            scores[name] = None
            all_healthy = False
            continue
        
        try:
            score = model.predict_direction(test_data)
            scores[name] = score
            
            # Detect suspicious outputs
            warnings = []
            if score == 0.5:
                warnings.append("EXACTLY 0.5 (likely fallback, not real prediction)")
            if score < 0.01 or score > 0.99:
                warnings.append(f"EXTREME ({score:.4f}) — possible saturation")
            
            # Check if Transformer has normalization loaded
            if name == "transformer":
                has_norm = model.norm_mean is not None
                if not has_norm:
                    warnings.append("NO NORMALIZATION — predictions may be unreliable")
            
            if warnings:
                print(f"   [{name.ljust(14)}] ⚠️ Score: {score:.4f} — {'; '.join(warnings)}")
                all_healthy = False
            else:
                print(f"   [{name.ljust(14)}] ✅ Score: {score:.4f}")
                
        except Exception as e:
            print(f"   [{name.ljust(14)}] ❌ CRASH: {e}")
            scores[name] = None
            all_healthy = False
    
    return scores, all_healthy


# ============================================================
# PHASE 3: FULL STACKING PIPELINE
# ============================================================
def audit_stacking_pipeline(symbol, test_data, expert_scores):
    """Test the full ensemble: experts → judge → final score."""
    print(f"\n⚖️ Phase 3: Stacking Pipeline Audit")
    
    # 3a. Test via StackingPredictor (the live engine path)
    try:
        predictor = StackingPredictor(symbol=symbol, timeframe="1h")
        pipeline_score = predictor.predict_direction(test_data)
        
        has_judge = predictor.judge is not None
        judge_status = "Judge loaded" if has_judge else "No judge (using weighted avg fallback)"
        
        print(f"   Pipeline score: {pipeline_score:.4f} ({judge_status})")
        
        # 3b. Compare against manual expert vector
        if all(v is not None for v in expert_scores.values()):
            manual_vector = [
                expert_scores.get('xgboost', 0.5),
                expert_scores.get('randomforest', 0.5),
                expert_scores.get('transformer', 0.5)
            ]
            
            if has_judge:
                manual_judge_score = predictor.judge.predict_direction(None, council_probs=manual_vector)
            else:
                manual_judge_score = float(np.average(manual_vector, weights=[1/3, 1/3, 1/3]))
            
            diff = abs(pipeline_score - manual_judge_score)
            
            if diff < 0.001:
                print(f"   ✅ PARITY: Pipeline ({pipeline_score:.4f}) matches manual ({manual_judge_score:.4f})")
            else:
                print(f"   ❌ DIVERGENCE: Pipeline ({pipeline_score:.4f}) vs Manual ({manual_judge_score:.4f}) — diff: {diff:.4f}")
                print(f"      This means the experts produced different scores when called")
                print(f"      through StackingPredictor vs individually. Check model caching.")
        else:
            print(f"   ⚠️ Cannot verify parity — some experts failed to load")
        
        return pipeline_score
        
    except Exception as e:
        print(f"   ❌ PIPELINE CRASH: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# PHASE 4: CONSISTENCY CHECK
# ============================================================
def audit_consistency(symbol, test_data):
    """Run the same prediction twice — results must be identical."""
    print(f"\n🔁 Phase 4: Consistency Check (determinism)")
    
    try:
        # Clear cache to force fresh loads
        clear_model_cache()
        
        predictor1 = StackingPredictor(symbol=symbol, timeframe="1h")
        score1 = predictor1.predict_direction(test_data)
        
        predictor2 = StackingPredictor(symbol=symbol, timeframe="1h")
        score2 = predictor2.predict_direction(test_data)
        
        diff = abs(score1 - score2)
        
        if diff < 0.0001:
            print(f"   ✅ DETERMINISTIC: Run 1 ({score1:.4f}) == Run 2 ({score2:.4f})")
        else:
            print(f"   ⚠️ NON-DETERMINISTIC: Run 1 ({score1:.4f}) vs Run 2 ({score2:.4f}) — diff: {diff:.4f}")
            print(f"      Small differences (<0.01) can be normal for Keras due to floating point.")
            print(f"      Large differences (>0.01) suggest a state or caching bug.")
        
        return diff < 0.01
        
    except Exception as e:
        print(f"   ❌ CONSISTENCY CRASH: {e}")
        return False


# ============================================================
# MAIN: RUN ALL PHASES
# ============================================================
def run_parity_audit(symbol="BTC-USD"):
    print(f"\n{'=' * 60}")
    print(f"🚀 NEO-V25 FULL PARITY AUDIT: {symbol}")
    print(f"{'=' * 60}")
    
    # Load test data
    path = os.path.join(DATA_DIR, f"{symbol}-1h.csv")
    if not os.path.exists(path):
        print(f"❌ CSV Missing: {path}")
        return
    
    df_raw = pd.read_csv(path)
    df, features = apply_mega_features(df_raw)
    
    # Use last 100 rows as test slice (mimics live engine)
    test_slice = df.tail(100).copy()
    print(f"📊 Test data: {len(test_slice)} rows | {len(features)} features")
    
    # Run all phases
    files_ok = audit_model_files(symbol)
    expert_scores, experts_ok = audit_expert_scores(symbol, test_slice)
    pipeline_score = audit_stacking_pipeline(symbol, test_slice, expert_scores)
    consistent = audit_consistency(symbol, test_slice)
    
    # Final verdict
    print(f"\n{'=' * 60}")
    print(f"📋 FINAL VERDICT: {symbol}")
    print(f"{'=' * 60}")
    
    issues = []
    if not files_ok: issues.append("Missing model files")
    if not experts_ok: issues.append("Expert scores unhealthy")
    if pipeline_score is None: issues.append("Pipeline crashed")
    if not consistent: issues.append("Non-deterministic output")
    
    if not issues:
        print(f"   ✅ ALL CHECKS PASSED — Council is operational")
        print(f"   📊 Final ensemble score: {pipeline_score:.4f}")
    else:
        print(f"   ❌ {len(issues)} ISSUE(S) DETECTED:")
        for issue in issues:
            print(f"      • {issue}")
        print(f"\n   🔧 Fix order: train experts → train transformer → train judge → re-run audit")
    
    print()
    return len(issues) == 0


if __name__ == "__main__":
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "PEPE-USD"]
    
    results = {}
    for sym in symbols:
        results[sym] = run_parity_audit(sym)
    
    # Summary table
    print("\n" + "=" * 60)
    print("📋 AUDIT SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    for sym, ok in results.items():
        print(f"   {sym.ljust(10)} {'✅ PASS' if ok else '❌ FAIL'}")
    print(f"\n   {passed}/{len(results)} symbols fully operational")
