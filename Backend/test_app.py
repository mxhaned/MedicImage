import pytest
import io
import numpy as np
from unittest.mock import patch, MagicMock

# Mock all the heavy model imports and loads before importing app2
with patch('tensorflow.keras.models.load_model') as mock_tf_load, \
     patch('torch.load') as mock_torch_load, \
     patch('joblib.load') as mock_joblib_load, \
     patch('app2.ImprovedUNet3D') as mock_unet:
    
    # Setup mock models
    mock_tumor_detection = MagicMock()
    mock_tumor_type = MagicMock()
    mock_survival = MagicMock()
    mock_segmentation = MagicMock()
    
    mock_tf_load.side_effect = [mock_tumor_detection, mock_tumor_type]
    mock_torch_load.return_value = mock_segmentation
    mock_joblib_load.return_value = mock_survival
    
    # Now import app2 with mocked models
    from app2 import app, tumor_analysis_pipeline, generate_patient_report

# --------- Pytest Fixture pour Flask test client ---------
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --------- Test route /analyze avec fichier simulé ---------
def test_analyze_route(client):
    # Créer un fichier image simulé
    data = {
        'file': (io.BytesIO(b"fake image content"), 'test_image.jpg')
    }

    # Mock du pipeline pour ne pas charger les vrais modèles
    with patch('app2.tumor_analysis_pipeline') as mock_pipeline, \
         patch('app2.generate_patient_report') as mock_report:

        mock_pipeline.return_value = (
            {"Tumor Type": "Glioma", "Predicted Survival Days": 365}, 
            {"t1_3d_tumor_volume": 1000.0}
        )
        mock_report.return_value = "Fake patient report"

        response = client.post('/analyze', data=data, content_type='multipart/form-data')
        json_data = response.get_json()

        assert response.status_code == 200
        assert json_data['tumor_detected'] is True
        assert json_data['tumor_type'] == "Glioma"
        assert json_data['report'] == "Fake patient report"

# --------- Test pipeline tumor_analysis_pipeline simple ---------
def test_tumor_analysis_pipeline_mocked():
    # Mock des fonctions internes pour éviter vrais modèles
    with patch('app2.preprocess_image_for_detection') as mock_preprocess, \
         patch('app2.tumor_detection_model.predict') as mock_detect, \
         patch('app2.tumor_type_model.predict') as mock_type, \
         patch('app2.survival_model.predict') as mock_survival:

        mock_preprocess.return_value = 'dummy_array'
        # Return numpy arrays instead of lists for proper indexing
        import numpy as np
        mock_detect.return_value = np.array([[0.9]])  # forte probabilité de tumeur
        mock_type.return_value = np.array([[0.8, 0.1, 0.1]])  # type Glioma
        mock_survival.return_value = np.array([365])  # survival prediction

        # Test with non-NIfTI file (expects "Requires NIfTI" for Glioma)
        result, features = tumor_analysis_pipeline("fake_image.jpg")
        assert result["Tumor Type"] == "Glioma"
        assert result["Requires NIfTI"] is True

        # Test with NIfTI file (should process normally)
        with patch('app2.preprocess_image_for_segmentation') as mock_seg_preprocess, \
             patch('app2.tumor_segmentation_model') as mock_seg_model, \
             patch('app2.visualize_segmentation'), \
             patch('app2.extract_features_from_mask') as mock_extract_features:
            
            mock_seg_preprocess.return_value = ('tensor', 'volume')
            mock_seg_model.return_value = 'segmentation_result'
            mock_extract_features.return_value = {
                "t1_3d_tumor_volume": 1000.0,
                "t1_3d_max_intensity": 200.0,
                "t1_3d_major_axis_length": 50.0,
                "t1_3d_area": 400.0,
                "t1_3d_minor_axis_length": 40.0,
                "t1_3d_extent": 0.8,
                "t1_3d_surface_to_volume_ratio": 0.1,
                "t1_3d_glcm_contrast": 15.0,
                "t1_3d_mean_intensity": 100.0,
                "t1_2d_area_median": 350.0
            }
            
            result_nifti, features_nifti = tumor_analysis_pipeline("fake_image.nii")
            assert result_nifti["Tumor Type"] == "Glioma"
            assert "Predicted Survival Days" in result_nifti

# --------- Test route /submit_manual_features ---------
def test_submit_manual_features(client):
    form_data = {
        't1_3d_tumor_volume': 1200,
        't1_3d_max_intensity': 200,
        't1_3d_major_axis_length': 50,
        't1_3d_area': 400,
        't1_3d_minor_axis_length': 40,
        't1_3d_extent': 0.8,
        't1_3d_surface_to_volume_ratio': 0.1,
        't1_3d_glcm_contrast': 15,
        't1_3d_mean_intensity': 100,
        't1_2d_area_median': 350,
        'tumor_type': 'Glioma'
    }

    with patch('app2.generate_patient_report') as mock_report, \
         patch('app2.survival_model.predict') as mock_survival:
        
        mock_report.return_value = "Fake manual report"
        import numpy as np
        mock_survival.return_value = np.array([365])  # Mock survival prediction

        response = client.post('/submit_manual_features', data=form_data)
        json_data = response.get_json()

        assert response.status_code == 200
        assert json_data['tumor_type'] == 'Glioma'
        assert json_data['report'] == "Fake manual report"

# --------- Test route /get_audio et /get_segmentation_image ---------
def test_get_audio_routes(client):
    # Sans fichier réel → renvoie d'erreur
    response = client.get('/get_audio')
    assert response.status_code == 404

    response = client.get('/get_audio_manual')
    assert response.status_code == 404

    response = client.get('/get_segmentation_image')
    assert response.status_code == 404
