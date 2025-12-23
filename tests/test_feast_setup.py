from feast import FeatureStore

def test_feast():
    """Test Feast connection"""
    store = FeatureStore(repo_path="/app/feast")
    
    print("Feast version:", store.version())
    print("Feature views:", store.list_feature_views())
    
    print("Feast setup successful")

if __name__ == "__main__":
    test_feast()
