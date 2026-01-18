"""
Test script to validate the dashboard setup and data loading
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test if all required libraries are installed"""
    print("🔍 Testing imports...")
    
    required_packages = [
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-Learn'),
        ('xgboost', 'XGBoost'),
        ('statsmodels', 'StatsModels'),
        ('plotly', 'Plotly'),
        ('streamlit', 'Streamlit'),
    ]
    
    failed = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print(f"   Run: pip install {' '.join(failed)}")
        return False
    
    print("✅ All imports successful!\n")
    return True


def test_data_loading():
    """Test if CSV can be loaded"""
    print("🔍 Testing data loading...")
    
    csv_paths = [
        "../data_raw/date_24_art_loreal_20%.csv",
        "../../data_raw/date_24_art_loreal_20%.csv",
        "c:/Users/Mara/disertatie/data_raw/date_24_art_loreal_20%.csv",
    ]
    
    from preprocessing import load_and_prepare_data
    
    for csv_path in csv_paths:
        try:
            df, target = load_and_prepare_data(csv_path)
            print(f"  ✅ Loaded from: {csv_path}")
            print(f"     Shape: {df.shape}")
            print(f"     Target column: {target}")
            print(f"     Date range: {df['DATA'].min()} to {df['DATA'].max()}\n")
            return True
        except FileNotFoundError:
            print(f"  ⚠️  Not found: {csv_path}")
        except Exception as e:
            print(f"  ❌ Error loading {csv_path}: {str(e)}\n")
    
    print("❌ Could not load CSV file from any expected path\n")
    return False


def test_preprocessing():
    """Test preprocessing functions"""
    print("🔍 Testing preprocessing functions...")
    
    try:
        from preprocessing import (
            load_and_prepare_data,
            filter_by_article,
            handle_missing_dates,
            TimeSeriesScaler,
            make_sequences
        )
        import numpy as np
        
        # Load data
        csv_path = "../data_raw/date_24_art_loreal_20%.csv"
        df, target = load_and_prepare_data(csv_path)
        
        # Test article filtering
        article_id = df.iloc[0]['ID_ARTICOL'] if 'ID_ARTICOL' in df.columns else None
        if article_id:
            df_article = filter_by_article(df, article_id)
            print(f"  ✅ Article filtering: {len(df_article)} rows")
        
        # Test missing dates handling
        df_filled = handle_missing_dates(df_article, fill_method='zero')
        print(f"  ✅ Missing dates filled: {len(df_filled)} rows")
        
        # Test scaler
        scaler = TimeSeriesScaler()
        data = df_article[target].values.reshape(-1, 1)
        data_scaled = scaler.fit_transform(data)
        print(f"  ✅ Data scaled: min={data_scaled.min():.4f}, max={data_scaled.max():.4f}")
        
        # Test sequences
        X, y = make_sequences(data_scaled, lookback=14)
        print(f"  ✅ Sequences created: X.shape={X.shape}, y.shape={y.shape}\n")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}\n")
        return False


def test_xgboost_model():
    """Test XGBoost model building"""
    print("🔍 Testing XGBoost model...")
    
    try:
        from xgboost_model import build_and_train_xgboost
        import pandas as pd
        import numpy as np
        
        # Create minimal test data
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        data = {
            'DATA': dates,
            'ID_ARTICOL': [1] * 60,
            'CANTITATE': np.random.randint(1, 100, 60)
        }
        df = pd.DataFrame(data)
        
        # Build and train model
        model, info = build_and_train_xgboost(df, 'DATA', 'ID_ARTICOL', 'CANTITATE')
        print(f"  ✅ XGBoost model trained successfully")
        print(f"     Features used: {info.get('n_features', 0)}")
        print(f"     Train size: {info.get('train_size', 0)}\n")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("  DASHBOARD TEST SUITE")
    print("="*50 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Preprocessing", test_preprocessing),
        ("XGBoost Model", test_xgboost_model),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}\n")
            results.append((test_name, False))
    
    # Summary
    print("="*50)
    print("  TEST SUMMARY")
    print("="*50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed\n")
    
    if passed == total:
        print("🎉 All tests passed! Ready to run: streamlit run app.py\n")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.\n")
        return 1


if __name__ == "__main__":
    exit(main())
