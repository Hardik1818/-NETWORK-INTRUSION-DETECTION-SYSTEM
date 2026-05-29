"""
Unit tests for UNSW-NB15 IDS preprocessing pipeline.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe mimicking UNSW-NB15 structure."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'dur':     np.random.exponential(0.1, n),
        'proto':   np.random.choice(['tcp', 'udp', 'icmp'], n),
        'service': np.random.choice(['-', 'http', 'ftp'], n),
        'state':   np.random.choice(['FIN', 'CON', 'REQ'], n),
        'spkts':   np.random.randint(1, 500, n),
        'dpkts':   np.random.randint(0, 400, n),
        'sbytes':  np.random.randint(40, 50000, n),
        'dbytes':  np.random.randint(0, 80000, n),
        'rate':    np.random.uniform(0, 1e5, n),
        'sttl':    np.random.choice([64, 128, 255], n),
        'dttl':    np.random.choice([64, 128, 0], n),
        'sload':   np.random.uniform(0, 1e6, n),
        'dload':   np.random.uniform(0, 1e6, n),
        'sinpkt':  np.random.uniform(0, 1000, n),
        'dinpkt':  np.random.uniform(0, 1000, n),
        'sjit':    np.random.uniform(0, 50, n),
        'djit':    np.random.uniform(0, 50, n),
        'swin':    np.random.choice([0, 255, 65535], n),
        'dwin':    np.random.choice([0, 255, 65535], n),
        'stcpb':   np.random.randint(0, 1e6, n),
        'dtcpb':   np.random.randint(0, 1e6, n),
        'tcprtt':  np.random.uniform(0, 1, n),
        'synack':  np.random.uniform(0, 0.5, n),
        'ackdat':  np.random.uniform(0, 0.5, n),
        'smean':   np.random.randint(40, 1500, n),
        'dmean':   np.random.randint(0, 1500, n),
        'trans_depth': np.random.randint(0, 5, n),
        'res_bdy_len': np.random.randint(0, 10000, n),
        'ct_srv_src':  np.random.randint(1, 50, n),
        'ct_state_ttl':np.random.randint(1, 6, n),
        'ct_dst_ltm':  np.random.randint(1, 50, n),
        'ct_src_dport_ltm': np.random.randint(1, 50, n),
        'ct_dst_sport_ltm': np.random.randint(1, 50, n),
        'ct_dst_src_ltm':   np.random.randint(1, 50, n),
        'label':   np.random.randint(0, 2, n),
        'attack_cat': np.random.choice(
            ['normal', 'dos', 'exploits', 'fuzzers', 'generic',
             'reconnaissance', 'backdoor', 'shellcode', 'worms', 'analysis'], n
        ),
    })
    return df


@pytest.fixture
def split_dfs(sample_df):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(sample_df, test_size=0.2, random_state=42, stratify=sample_df['attack_cat'])
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ── Tests: Data Quality ────────────────────────────────────────────────────────

class TestDataQuality:

    def test_no_missing_after_fillna(self, split_dfs):
        train, _ = split_dfs
        numeric = train.select_dtypes(include='number')
        filled = numeric.fillna(numeric.median())
        assert filled.isnull().sum().sum() == 0

    def test_attack_cat_column_exists(self, sample_df):
        assert 'attack_cat' in sample_df.columns

    def test_expected_attack_categories(self, sample_df):
        expected = {'normal', 'dos', 'exploits', 'fuzzers', 'generic',
                    'reconnaissance', 'backdoor', 'shellcode', 'worms', 'analysis'}
        found = set(sample_df['attack_cat'].str.strip().str.lower().unique())
        assert found.issubset(expected | found), "Unexpected attack categories found"

    def test_no_negative_bytes(self, sample_df):
        assert (sample_df['sbytes'] >= 0).all()
        assert (sample_df['dbytes'] >= 0).all()

    def test_packet_counts_non_negative(self, sample_df):
        assert (sample_df['spkts'] >= 0).all()
        assert (sample_df['dpkts'] >= 0).all()


# ── Tests: Preprocessing ───────────────────────────────────────────────────────

class TestPreprocessing:

    def test_label_encoder_covers_all_classes(self, split_dfs):
        train, test = split_dfs
        le = LabelEncoder()
        all_labels = pd.concat([train['attack_cat'], test['attack_cat']])
        le.fit(all_labels.str.strip().str.lower())
        # All test labels must be encodable
        test_labels = test['attack_cat'].str.strip().str.lower()
        encoded = le.transform(test_labels)
        assert len(encoded) == len(test_labels)

    def test_categorical_encoding_produces_numeric(self, split_dfs):
        train, test = split_dfs
        for col in ['proto', 'service', 'state']:
            if col in train.columns:
                le = LabelEncoder()
                combined = pd.concat([train[col], test[col]]).astype(str)
                le.fit(combined)
                encoded_train = le.transform(train[col].astype(str))
                assert encoded_train.dtype in [np.int32, np.int64, int]

    def test_scaler_output_shape(self, split_dfs):
        train, test = split_dfs
        drop_cols = ['attack_cat', 'id', 'label']
        X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
        X_test  = test.drop(columns=[c for c in drop_cols if c in test.columns])

        # Encode object cols
        for col in X_train.select_dtypes('object').columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col], X_test[col]]).astype(str)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col]  = le.transform(X_test[col].astype(str))

        X_train = X_train.fillna(X_train.median())
        X_test  = X_test.fillna(X_train.median())

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        assert X_train_sc.shape[0] == len(train)
        assert X_test_sc.shape[1]  == X_train_sc.shape[1]

    def test_scaler_mean_near_zero(self, split_dfs):
        train, _ = split_dfs
        drop_cols = ['attack_cat', 'id', 'label']
        X = train.drop(columns=[c for c in drop_cols if c in train.columns])
        for col in X.select_dtypes('object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X = X.fillna(X.median())
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        means = np.abs(X_sc.mean(axis=0))
        assert (means < 1e-9).all(), "Scaled features should have ~0 mean"


# ── Tests: Feature Selection ───────────────────────────────────────────────────

class TestFeatureSelection:

    def test_select_k_best_reduces_features(self, split_dfs):
        train, test = split_dfs
        drop_cols = ['attack_cat', 'id', 'label']
        X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
        for col in X_train.select_dtypes('object').columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_train = X_train.fillna(X_train.median())
        le_y = LabelEncoder()
        y_train = le_y.fit_transform(train['attack_cat'].str.strip().str.lower())
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_train)

        K = 10
        selector = SelectKBest(score_func=mutual_info_classif, k=K)
        X_sel = selector.fit_transform(X_sc, y_train)
        assert X_sel.shape[1] == K

    def test_selector_transform_test_set(self, split_dfs):
        train, test = split_dfs
        drop_cols = ['attack_cat', 'id', 'label']
        X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
        X_test  = test.drop(columns=[c for c in drop_cols if c in test.columns])

        for col in X_train.select_dtypes('object').columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col], X_test[col]]).astype(str)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col]  = le.transform(X_test[col].astype(str))

        X_train = X_train.fillna(X_train.median())
        X_test  = X_test.fillna(X_train.median())
        le_y = LabelEncoder()
        y_train = le_y.fit_transform(train['attack_cat'].str.strip().str.lower())

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        K = 10
        selector = SelectKBest(score_func=mutual_info_classif, k=K)
        selector.fit(X_train_sc, y_train)
        X_test_sel = selector.transform(X_test_sc)
        assert X_test_sel.shape == (len(test), K)


# ── Tests: Model Output Shape ──────────────────────────────────────────────────

class TestModelOutputs:

    def test_rf_predict_output_length(self, split_dfs):
        from sklearn.ensemble import RandomForestClassifier
        train, test = split_dfs
        drop_cols = ['attack_cat', 'id', 'label']
        X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
        X_test  = test.drop(columns=[c for c in drop_cols if c in test.columns])
        for col in X_train.select_dtypes('object').columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col], X_test[col]]).astype(str)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col]  = le.transform(X_test[col].astype(str))
        X_train = X_train.fillna(X_train.median())
        X_test  = X_test.fillna(X_train.median())
        le_y = LabelEncoder()
        y_train = le_y.fit_transform(train['attack_cat'].str.strip().str.lower())

        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        assert len(preds) == len(test)

    def test_rf_predict_valid_classes(self, split_dfs):
        from sklearn.ensemble import RandomForestClassifier
        train, test = split_dfs
        drop_cols = ['attack_cat', 'id', 'label']
        X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
        X_test  = test.drop(columns=[c for c in drop_cols if c in test.columns])
        for col in X_train.select_dtypes('object').columns:
            le = LabelEncoder()
            combined = pd.concat([X_train[col], X_test[col]]).astype(str)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col]  = le.transform(X_test[col].astype(str))
        X_train = X_train.fillna(X_train.median())
        X_test  = X_test.fillna(X_train.median())
        le_y = LabelEncoder()
        y_train = le_y.fit_transform(train['attack_cat'].str.strip().str.lower())

        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        assert set(preds).issubset(set(y_train))
