import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import h5py

from pipe_bci_toolkit.core.base_type import MOABBData, CovarianceData
from pipe_bci_toolkit.data.io import MOABBDataManager, HDF5Manager

@pytest.fixture
def mock_moabb_data():
    return MOABBData(
        x=np.random.rand(10, 22, 1000).astype(np.float32),
        y=np.array(['left', 'right'] * 5),
        subjects=np.arange(10),
        dataset=None,
        paradigm=None,
        freqr=250.0,
        channel_names=['C3', 'C4', 'Cz']
    )

@pytest.fixture
def mock_cov_data():
    return CovarianceData(
        x=np.random.rand(10, 22, 22), # 10 trials of 22x22 matrices
        y=np.array([0, 1] * 5),
        subjects=np.arange(10),
        channel_names=['C3', 'C4']
    )

def test_moabb_manager_save_load(tmp_path, mock_moabb_data):
    file_path = tmp_path / "test_moabb.h5"
    manager = MOABBDataManager()

    # Test Save
    with h5py.File(file_path, 'w') as f:
        manager.save(f, mock_moabb_data, compression='gzip', chunk_shape=True)

    # Test Load
    with h5py.File(file_path, 'r') as f:
        loaded_data = manager.load(f)

    # Assertions
    np.testing.assert_array_equal(loaded_data.x, mock_moabb_data.x)
    np.testing.assert_array_equal(loaded_data.y, mock_moabb_data.y)
    assert loaded_data.freqr == mock_moabb_data.freqr
    assert loaded_data.channel_names == mock_moabb_data.channel_names

def test_hdf5_manager_routing_save():
    """Verify that save() picks the right strategy and adds .h5 suffix."""
    manager = HDF5Manager(verbose=False)
    
    # Mock the strategy for 'moabb'
    mock_strategy = MagicMock()
    manager.strategies['moabb'] = mock_strategy
    
    fake_data = MagicMock()
    filename = "test_experiment"
    
    # We patch h5py.File so it doesn't actually create a file on disk
    with patch("h5py.File") as mock_file_class:
        manager.save(fake_data, filename, data_type='moabb')
        
        # 1. Check if the suffix was added correctly
        mock_file_class.assert_called_once()
        called_path = mock_file_class.call_args[0][0]
        assert str(called_path).endswith(".h5")
        
        # 2. Check if the strategy's save method was called
        mock_strategy.save.assert_called_once()
        
        # 3. Verify compression was passed down
        _, kwargs = mock_strategy.save.call_args
        assert kwargs['compression'] == 'gzip'

def test_hdf5_manager_invalid_type():
    """Ensure it raises ValueError for unknown strategies."""
    manager = HDF5Manager()
    with pytest.raises(ValueError, match="Unknown data type"):
        manager.save(data=None, filename="fail", data_type="invalid_type")

def test_guess_chunk_shape_logic():
    """Unit test for the internal chunking helper."""
    import numpy as np
    manager = HDF5Manager()
    
    # Case 1: Standard EEG array (Trials, Channels, Time)
    # 100 trials, 32 channels, 1000 samples (float64 = 8 bytes)
    # Row size = 32 * 1000 * 8 = 256,000 bytes (~0.25 MB)
    # Ideal rows for 1MB = 4 rows
    data = np.zeros((100, 32, 1000))
    chunks = manager._guess_chunk_shape(data)
    
    assert chunks[0] == 4  # Should suggest 4 trials per chunk
    assert chunks[1:] == (32, 1000)

    # Case 2: Not a numpy array
    assert manager._guess_chunk_shape(None) is True

def test_hdf5_manager_routing_load():
    manager = HDF5Manager(verbose=False)
    
    # Mock strategy
    mock_strategy = MagicMock()
    mock_strategy.load.return_value = "final_data"
    manager.strategies['moabb'] = mock_strategy

    with patch("h5py.File") as mock_file_class:
        # Simulate the h5py.File context manager
        mock_hf = mock_file_class.return_value.__enter__.return_value
        # Simulate the attribute 'data_type' being in the file
        mock_hf.attrs.get.return_value = 'moabb'
        
        result = manager.load("dummy_file")
        
        assert result == "final_data"
        mock_strategy.load.assert_called_once_with(mock_hf)