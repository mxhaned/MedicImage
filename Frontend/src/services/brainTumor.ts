export interface TumorFeatures {
  t1_3d_tumor_volume: number;
  t1_3d_max_intensity: number;
  t1_3d_major_axis_length: number;
  t1_3d_area: number;
  t1_3d_minor_axis_length: number;
  t1_3d_extent: number;
  t1_3d_surface_to_volume_ratio: number;
  t1_3d_glcm_contrast: number;
  t1_3d_mean_intensity: number;
  t1_2d_area_median: number;
}

// Types
export interface TumorAnalysisResult {
  tumor_type: string;
  confidence: number;
  features?: TumorFeatures;
  survival_days?: number;
  risk_category?: string;
  model_info?: {
    model_name: string;
    version: string;
  };
}

export interface GliomaSegmentationResult {
  tumor_type: string;
  confidence: number;
  segmentation_info: {
    volume: number;
    max_intensity: number;
    mean_intensity: number;
    std_intensity: number;
    compactness: number;
  };
  extracted_features: TumorFeatures;
  survival_prediction: {
    survival_days: number;
    risk_category: string;
    confidence: number;
  };
  model_info: {
    segmentation_model: string;
    prediction_model: string;
    version: string;
  };
  report: string;
  segmentation_image_url: string;
  success: boolean;
  message: string;
}

export interface NonGliomaAnalysisResult {
  analysis_method: string;
  features: any;
  message: string;
  recovery_info: {
    features_mapped: number;
    features_recovered: number;
    mapped_details: any[];
    recovery_percentage: number;
  };
  report: string;
  success: boolean;
  survival_days: number;
  tumor_detected: boolean;
  tumor_type: string;
}

export interface ManualFeatures {
  t1_3d_tumor_volume: number;
  t1_3d_max_intensity: number;
  t1_3d_major_axis_length: number;
  t1_3d_area: number;
  t1_3d_extent: number;
  t1_3d_mean_intensity: number;
  t1_3d_minor_axis_length: number;
  t1_3d_surface_to_volume_ratio: number;
  t1_3d_glcm_contrast: number;
  t1_2d_area_median: number;
}

const API_BASE_URL = 'http://localhost:5001';  // Port 5001 pour Brain Tumor service

export const BrainTumorService = {
  // Étape 1: Analyse initiale pour détecter le type de tumeur
  async analyzeImage(imageFile: File): Promise<TumorAnalysisResult> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Analysis failed' }));
      throw new Error(error.error || 'Analysis failed');
    }

    return response.json();
  },

  // Step 2a: Glioma segmentation analysis
  async analyzeGliomaSegmentation(
    niftiFile: File, 
    initialAnalysis: TumorAnalysisResult
  ): Promise<GliomaSegmentationResult> {
    const formData = new FormData();
    formData.append('nifti_file', niftiFile);
    formData.append('initial_analysis', JSON.stringify(initialAnalysis));

    const response = await fetch(`${API_BASE_URL}/analyze-glioma-segmentation`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ message: 'Analysis failed' }));
      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  },

  // Step 2b: Non-Glioma manual analysis
  async analyzeNonGliomaManual(
    features: ManualFeatures,
    initialAnalysis: TumorAnalysisResult
  ): Promise<NonGliomaAnalysisResult> {
    const response = await fetch(`${API_BASE_URL}/analyze-non-glioma-manual`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        features,
        initial_analysis: initialAnalysis
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ message: 'Analysis failed' }));
      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  },

  async checkImageType(file: File): Promise<{ is_3d: boolean }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/check-image-type`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('Failed to check image type');
    }

    return response.json();
  },

  async generateReport(
    analysisResult: TumorAnalysisResult | GliomaSegmentationResult | NonGliomaAnalysisResult
  ): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/generate-pdf-report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ 
        analysis_result: analysisResult,
        segmentation_image_url: (analysisResult as GliomaSegmentationResult).segmentation_image_url
      })
    });

    if (!response.ok) {
      throw new Error('Failed to generate report');
    }

    return response.blob();
  }
};