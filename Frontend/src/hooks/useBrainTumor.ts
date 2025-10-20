import { useState } from 'react';
import { 
  BrainTumorService, 
  TumorAnalysisResult, 
  GliomaSegmentationResult, 
  NonGliomaAnalysisResult,
  ManualFeatures 
} from '@/services/brainTumor';

export type AnalysisStep = 'detection' | 'segmentation' | 'manual' | 'completed';

export const useBrainTumor = () => {
  const [currentStep, setCurrentStep] = useState<AnalysisStep>('detection');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [progress, setProgress] = useState(0);
  
  // Results for different steps
  const [detectionResult, setDetectionResult] = useState<TumorAnalysisResult | null>(null);
  const [segmentationResult, setSegmentationResult] = useState<GliomaSegmentationResult | null>(null);
  const [manualResult, setManualResult] = useState<NonGliomaAnalysisResult | null>(null);
  
  const [error, setError] = useState<string | null>(null);

  // Step 1: Initial tumor detection
  const analyzeImage = async (imageFile: File) => {
    setIsAnalyzing(true);
    setError(null);
    setProgress(0);
    setCurrentStep('detection');

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 500);

      const data = await BrainTumorService.analyzeImage(imageFile);
      
      clearInterval(progressInterval);
      setProgress(100);
      setDetectionResult(data);
      
      // Determine next step based on tumor type
      if (data.tumor_type && data.tumor_type.toLowerCase().includes('glioma')) {
        setCurrentStep('segmentation');
      } else {
        setCurrentStep('manual');
      }
      
      return data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Analysis failed';
      setError(errorMessage);
      throw error;
    } finally {
      setIsAnalyzing(false);
    }
  };

    // Step 2a: Glioma segmentation with NIfTI file
  const performSegmentation = async (niftiFile: File) => {
    if (!detectionResult) {
      throw new Error('No detection result available');
    }

    setIsSegmenting(true);
    setIsAnalyzing(true); // Double protection contre écran blanc
    setError(null);
    setProgress(0);

    try {
      // Progression plus réaliste avec étapes spécifiques
      const progressSteps = [
        { value: 15, message: 'Validating NIfTI file...' },
        { value: 30, message: 'Preprocessing 3D volume...' },
        { value: 50, message: 'Running U-Net segmentation...' },
        { value: 70, message: 'Extracting tumor features...' },
        { value: 85, message: 'Predicting survival...' },
        { value: 95, message: 'Generating report...' }
      ];
      
      let stepIndex = 0;
      
      const progressInterval = setInterval(() => {
        if (stepIndex < progressSteps.length) {
          setProgress(progressSteps[stepIndex].value);
          stepIndex++;
        }
      }, 3000); // Plus lent pour une meilleure UX

      const data = await BrainTumorService.analyzeGliomaSegmentation(niftiFile, detectionResult);
      
      clearInterval(progressInterval);
      setProgress(100);
      
      // Vérifier que nous avons bien reçu les données attendues
      if (data.success && data.extracted_features && data.survival_prediction) {
        console.log('✅ Segmentation data received:', {
          features: Object.keys(data.extracted_features).length,
          survival_days: data.survival_prediction.survival_days,
          report_length: data.report?.length || 0
        });
        
        // Petit délai pour que l'utilisateur voie 100%
        setTimeout(() => {
          setSegmentationResult(data);
          setCurrentStep('completed');
          setIsSegmenting(false);
          setIsAnalyzing(false);
        }, 1500);
      } else {
        throw new Error('Données de segmentation incomplètes reçues du serveur');
      }
      
      return data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Segmentation failed';
      setError(errorMessage);
      setIsSegmenting(false);
      setIsAnalyzing(false);
      throw error;
    }
  };

  // Step 2b: Manual feature entry for non-Glioma
  const submitManualFeatures = async (features: ManualFeatures) => {
    if (!detectionResult) {
      throw new Error('No detection result available');
    }

    setIsAnalyzing(true);
    setError(null);
    setProgress(0);

    try {
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 500);

      const data = await BrainTumorService.analyzeNonGliomaManual(features, detectionResult);
      
      clearInterval(progressInterval);
      setProgress(100);
      setManualResult(data);
      setCurrentStep('completed');
      
      return data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Manual analysis failed';
      setError(errorMessage);
      throw error;
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Generate report based on current results
  const generateReport = async (resultWithPatientName?: any) => {
    let finalResult = resultWithPatientName;
    
    if (!finalResult) {
      if (segmentationResult) {
        finalResult = segmentationResult;
      } else if (manualResult) {
        finalResult = manualResult;
      } else if (detectionResult) {
        finalResult = detectionResult;
      }
    }
    
    if (!finalResult) {
      throw new Error('No analysis result available for report generation');
    }
    
    setIsGeneratingReport(true);
    try {
      const reportBlob = await BrainTumorService.generateReport(finalResult);
      
      // Create filename with patient name if available
      const patientName = finalResult.patient_name || 'Patient';
      const sanitizedName = patientName.replace(/[^a-zA-Z0-9\-_]/g, '_');
      const timestamp = new Date().toISOString().slice(0, 16).replace(/[:-]/g, '');
      
      // Download the report
      const url = window.URL.createObjectURL(reportBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Rapport_Tumeur_Cerebrale_${sanitizedName}_${timestamp}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Report generation failed';
      setError(errorMessage);
      throw error;
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const resetState = () => {
    setCurrentStep('detection');
    setIsAnalyzing(false);
    setIsSegmenting(false);
    setIsGeneratingReport(false);
    setProgress(0);
    setDetectionResult(null);
    setSegmentationResult(null);
    setManualResult(null);
    setError(null);
  };

  // Helper functions
  const isGlioma = () => {
    return detectionResult?.tumor_type && 
           detectionResult.tumor_type.toLowerCase().includes('glioma');
  };

  const canProceedToNext = () => {
    switch (currentStep) {
      case 'detection':
        return detectionResult !== null;
      case 'segmentation':
        return segmentationResult !== null;
      case 'manual':
        return manualResult !== null;
      case 'completed':
        return true;
      default:
        return false;
    }
  };

  const getFinalResult = () => {
    if (segmentationResult) return segmentationResult;
    if (manualResult) return manualResult;
    return detectionResult;
  };

  return {
    // State
    currentStep,
    isAnalyzing,
    isSegmenting,
    isGeneratingReport,
    progress,
    detectionResult,
    segmentationResult,
    manualResult,
    error,
    
    // Actions
    analyzeImage,
    performSegmentation,
    submitManualFeatures,
    generateReport,
    resetState,
    
    // Helpers
    isGlioma,
    canProceedToNext,
    getFinalResult
  };
};