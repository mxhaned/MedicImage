const API_BASE_URL = 'http://localhost:5002/api';  // Port 5002 pour DermaScan

export interface ClassificationResult {
  success: boolean;
  predictions: Record<string, number>;
  primary_condition: string;
  confidence: number;
  class_names: string[];
}

export interface ModelInfo {
  model_type: string;
  num_classes: number;
  class_names: string[];
  device: string;
}

export interface ReportRequest {
  image: string;
  analysis_data: Record<string, any>;
  patient_name?: string;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async healthCheck(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }
    return response.json();
  }

  async getModelInfo(): Promise<ModelInfo> {
    const response = await fetch(`${this.baseUrl}/model-info`);
    if (!response.ok) {
      throw new Error(`Failed to get model info: ${response.statusText}`);
    }
    return response.json();
  }

  async classifyImage(imageData: string): Promise<ClassificationResult> {
    const response = await fetch(`${this.baseUrl}/classify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: imageData }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.error || `Classification failed: ${response.statusText}`);
    }

    return response.json();
  }

  async generateReport(request: ReportRequest): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/generate-pdf-report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.error || `Report generation failed: ${response.statusText}`);
    }

    return response.blob();
  }

  async downloadReport(request: ReportRequest, filename?: string): Promise<void> {
    try {
      const blob = await this.generateReport(request);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Set filename
      if (filename) {
        link.download = filename;
      } else {
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        link.download = `medical_report_${timestamp}.pdf`;
      }
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      throw new Error(`Failed to download report: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
}

export const apiService = new ApiService();
export default apiService; 