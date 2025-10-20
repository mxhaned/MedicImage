import axios from 'axios';
import { toast } from 'sonner';

const API_BASE_URL = 'http://localhost:5001';

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

export interface TumorAnalysisResult {
  tumor_detected: boolean;
  tumor_type: string;
  survival_days: number;
  segmentation_image: string;
  report: string;
  features: TumorFeatures;
}

export interface ApiError {
  message: string;
  status: number;
  details?: string;
}

export const BrainTumorService = {
  async analyzeImage(imageFile: File, niftiFile?: File): Promise<TumorAnalysisResult> {
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      if (niftiFile) {
        formData.append('nifti_file', niftiFile);
      }

      const config = {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent: any) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            // You can use this for progress tracking
          }
        }
      };
      const response = await axios.post(`${API_BASE_URL}/analyze`, formData, config);
      return response.data as TumorAnalysisResult;
    } catch (error) {
      handleApiError(error);
      throw error;
    }
  },

    async generateReport(result: TumorAnalysisResult): Promise<Blob> {
    try {
      const response = await axios.post(`${API_BASE_URL}/generate-pdf-report`, result, {
        responseType: 'blob'
      });
      return response.data as Blob;
    } catch (error) {
      handleApiError(error);
      throw error;
    }
  },

  async checkImageType(file: File): Promise<{ is_3d: boolean }> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await axios.post(`${API_BASE_URL}/check-image-type`, formData);
      return response.data as { is_3d: boolean };
    } catch (error) {
      handleApiError(error);
      throw error;
    }
  }
};

const handleApiError = (error: unknown) => {
  if (error && typeof error === 'object' && 'response' in error) {
    const axiosError = error as any;
    toast.error(axiosError.response?.data?.message || 'An error occurred');
  } else {
    toast.error('An unexpected error occurred');
  }
};
