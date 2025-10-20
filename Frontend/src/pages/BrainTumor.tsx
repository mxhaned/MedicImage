import React, { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { FileImage, Upload, AlertCircle, FileText, ArrowRight, CheckCircle, Info } from 'lucide-react';
import { toast } from 'sonner';
import { Link } from 'react-router-dom';
import { useBrainTumor } from '@/hooks/useBrainTumor';
import { BrainTumorService, ManualFeatures } from '@/services/brainTumor';
import PatientNameModal from '@/components/PatientNameModal';

const BrainTumor = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [selectedNifti, setSelectedNifti] = useState<File | null>(null);
  const [manualFeatures, setManualFeatures] = useState<ManualFeatures>({
    t1_3d_tumor_volume: 0,
    t1_3d_max_intensity: 0,
    t1_3d_major_axis_length: 0,
    t1_3d_area: 0,
    t1_3d_extent: 0,
    t1_3d_mean_intensity: 0,
    t1_3d_minor_axis_length: 0,
    t1_3d_surface_to_volume_ratio: 0,
    t1_3d_glcm_contrast: 0,
    t1_2d_area_median: 0,
  });
  
  const [fieldErrors, setFieldErrors] = useState<Set<keyof ManualFeatures>>(new Set());
  const [showValidation, setShowValidation] = useState(false);
  const [showPatientNameModal, setShowPatientNameModal] = useState(false);
  const [showDisclaimer, setShowDisclaimer] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const niftiInputRef = useRef<HTMLInputElement>(null);
  
  const {
    currentStep,
    isAnalyzing,
    isSegmenting,
    isGeneratingReport,
    progress,
    detectionResult,
    segmentationResult,
    manualResult,
    error,
    analyzeImage,
    performSegmentation,
    submitManualFeatures,
    generateReport,
    resetState,
    isGlioma,
    canProceedToNext,
    getFinalResult
  } = useBrainTumor();

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      toast.success('Image uploaded successfully');
    }
  };

  const handleNiftiUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedNifti(file);
      toast.success('NIfTI file uploaded successfully');
    }
  };

  const handleInitialAnalysis = async () => {
    if (!selectedImage) {
      toast.error('Please select an image to analyze');
      return;
    }

    try {
      await analyzeImage(selectedImage);
      toast.success('Initial analysis completed');
    } catch (error) {
      console.error('Analysis failed:', error);
      toast.error('Analysis failed');
    }
  };

  const handleSegmentation = async () => {
    if (!selectedNifti) {
      toast.error('Please upload a NIfTI file for segmentation');
      return;
    }

    try {
      await performSegmentation(selectedNifti);
      toast.success('Segmentation analysis completed');
    } catch (error) {
      console.error('Segmentation failed:', error);
      toast.error('Segmentation failed');
    }
  };

  const handleManualSubmission = async () => {
    if (!validateManualFeatures()) {
      toast.error('Please fill in all fields with valid values (> 0)');
      return;
    }

    try {
      await submitManualFeatures(manualFeatures);
      toast.success('Manual analysis completed');
      setShowValidation(false);
      setFieldErrors(new Set());
    } catch (error) {
      console.error('Manual analysis failed:', error);
      toast.error('Manual analysis failed');
    }
  };

  const handleFeatureChange = (key: keyof ManualFeatures, value: string) => {
    const numericValue = parseFloat(value) || 0;
    setManualFeatures(prev => ({
      ...prev,
      [key]: numericValue
    }));
    
    // Remove field from errors if valid value is entered
    if (numericValue > 0 && fieldErrors.has(key)) {
      setFieldErrors(prev => {
        const newErrors = new Set(prev);
        newErrors.delete(key);
        return newErrors;
      });
    }
  };

  const validateManualFeatures = (): boolean => {
    const errors = new Set<keyof ManualFeatures>();
    
    Object.entries(manualFeatures).forEach(([key, value]) => {
      if (value <= 0) {
        errors.add(key as keyof ManualFeatures);
      }
    });
    
    setFieldErrors(errors);
    setShowValidation(true);
    
    return errors.size === 0;
  };

  const getFieldLabel = (key: keyof ManualFeatures): string => {
    const labels: Record<keyof ManualFeatures, string> = {
      t1_3d_tumor_volume: 'Tumor Volume (3D)',
      t1_3d_max_intensity: 'Maximum Intensity (3D)',
      t1_3d_major_axis_length: 'Major Axis Length (3D)',
      t1_3d_area: 'Area (3D)',
      t1_3d_extent: 'Extent (3D)',
      t1_3d_mean_intensity: 'Mean Intensity (3D)',
      t1_3d_minor_axis_length: 'Minor Axis Length (3D)',
      t1_3d_surface_to_volume_ratio: 'Surface/Volume Ratio (3D)',
      t1_3d_glcm_contrast: 'GLCM Contrast (3D)',
      t1_2d_area_median: 'Median Area (2D)',
    };
    return labels[key];
  };

  const isFieldInError = (key: keyof ManualFeatures): boolean => {
    return showValidation && fieldErrors.has(key);
  };

  const handleGenerateReportClick = () => {
    setShowPatientNameModal(true);
  };

  const handlePatientNameConfirm = async (patientName: string) => {
    try {
      const finalResult = getFinalResult();
      if (!finalResult) {
        toast.error('No analysis results available to generate the report');
        setShowPatientNameModal(false);
        return;
      }

      // Add patient name to data
      const resultWithPatientName = {
        ...finalResult,
        patient_name: patientName
      };

      await generateReport(resultWithPatientName);
      setShowPatientNameModal(false);
      toast.success(`Report generated successfully for ${patientName}`);
    } catch (error) {
      console.error('Error generating report:', error);
      toast.error('Error generating report');
      setShowPatientNameModal(false);
    }
  };

  const handlePatientNameCancel = () => {
    setShowPatientNameModal(false);
  };

  const handleStartOver = () => {
    resetState();
    setSelectedImage(null);
    setSelectedNifti(null);
    setFieldErrors(new Set());
    setShowValidation(false);
    setShowPatientNameModal(false);
    setManualFeatures({
      t1_3d_tumor_volume: 0,
      t1_3d_max_intensity: 0,
      t1_3d_major_axis_length: 0,
      t1_3d_area: 0,
      t1_3d_extent: 0,
      t1_3d_mean_intensity: 0,
      t1_3d_minor_axis_length: 0,
      t1_3d_surface_to_volume_ratio: 0,
      t1_3d_glcm_contrast: 0,
      t1_2d_area_median: 0,
    });
  };

  // Global loading screen to avoid blank screen
  if (isSegmenting || (isAnalyzing && currentStep === 'segmentation')) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-medical-gray-50 via-white to-medical-teal/5">
        {/* Always visible header */}
        <header className="bg-white/80 backdrop-blur-md border-b border-gray-200/50 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <Link to="/" className="flex items-center space-x-2">
                <div className="w-8 h-8 medical-gradient rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">M</span>
                </div>
                <span className="text-xl font-bold text-gray-900">MedicImage</span>
              </Link>
            </div>
          </div>
        </header>

        {/* Full screen segmentation display */}
        <div className="flex items-center justify-center min-h-[calc(100vh-4rem)] px-4">
          <div className="max-w-2xl w-full">
            <Card className="medical-card text-center">
              <CardContent className="pt-8 pb-8">
                <div className="space-y-6">
                  <div className="flex justify-center">
                    <div className="w-16 h-16 medical-gradient rounded-full flex items-center justify-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                    </div>
                  </div>
                  
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900 mb-2">
                      🧠 Analyzing Brain Tumor
                    </h2>
                    <p className="text-gray-600 mb-4">
                      Our AI is performing deep 3D segmentation of your MRI scan...
                    </p>
                  </div>

                  <div className="space-y-3">
                    <Progress value={progress} className="h-3" />
                    <p className="text-lg font-medium text-medical-teal">
                      {progress}% Complete
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                    <div className={`p-4 rounded-lg border-2 transition-all ${
                      progress > 25 ? 'border-green-500 bg-green-50' : 'border-gray-200 bg-gray-50'
                    }`}>
                      <div className="text-2xl mb-2">🔍</div>
                      <h4 className="font-semibold text-sm">Preprocessing</h4>
                      <p className="text-xs text-gray-600">Preparing NIfTI data</p>
                    </div>
                    
                    <div className={`p-4 rounded-lg border-2 transition-all ${
                      progress > 60 ? 'border-green-500 bg-green-50' : 'border-gray-200 bg-gray-50'
                    }`}>
                      <div className="text-2xl mb-2">🧠</div>
                      <h4 className="font-semibold text-sm">U-Net Analysis</h4>
                      <p className="text-xs text-gray-600">3D tumor segmentation</p>
                    </div>
                    
                    <div className={`p-4 rounded-lg border-2 transition-all ${
                      progress > 85 ? 'border-green-500 bg-green-50' : 'border-gray-200 bg-gray-50'
                    }`}>
                      <div className="text-2xl mb-2">📊</div>
                      <h4 className="font-semibold text-sm">Feature Extraction</h4>
                      <p className="text-xs text-gray-600">Computing features</p>
                    </div>
                  </div>

                  <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                    <p className="text-sm text-blue-800">
                      <strong>Please wait...</strong> This process may take 30-60 seconds to complete.
                      The AI is analyzing your brain MRI with deep learning models.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-gray-50 via-white to-medical-teal/5">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-gray-200/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 medical-gradient rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">M</span>
              </div>
              <span className="text-xl font-bold text-gray-900">MedicImage</span>
            </Link>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8 animate-fade-in">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">Brain Tumor Analysis</h1>
              <p className="text-gray-600">Upload MRI scans for AI-powered tumor detection</p>
            </div>
            <Button
              onClick={() => setShowDisclaimer(true)}
              variant="outline"
              size="sm"
              className="flex items-center space-x-2 border-orange-200 text-orange-700 hover:bg-orange-50"
            >
              <Info className="w-4 h-4" />
              <span>Important Notice</span>
            </Button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className={`flex items-center space-x-2 ${currentStep === 'detection' ? 'text-medical-teal' : 'text-gray-400'}`}>
              <CheckCircle className={`w-5 h-5 ${detectionResult ? 'text-green-600' : 'text-gray-400'}`} />
              <span>Detection</span>
            </div>
            <ArrowRight className="w-4 h-4 text-gray-400" />
            <div className={`flex items-center space-x-2 ${currentStep === 'segmentation' || currentStep === 'manual' ? 'text-medical-teal' : 'text-gray-400'}`}>
              <CheckCircle className={`w-5 h-5 ${segmentationResult || manualResult ? 'text-green-600' : 'text-gray-400'}`} />
              <span>{isGlioma() ? 'Segmentation' : 'Manual Input'}</span>
            </div>
            <ArrowRight className="w-4 h-4 text-gray-400" />
            <div className={`flex items-center space-x-2 ${currentStep === 'completed' ? 'text-medical-teal' : 'text-gray-400'}`}>
              <CheckCircle className={`w-5 h-5 ${currentStep === 'completed' ? 'text-green-600' : 'text-gray-400'}`} />
              <span>Completed</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Step 1: Image Upload and Detection */}
          {currentStep === 'detection' && (
            <Card className="medical-card">
              <CardHeader>
                <CardTitle>Step 1: Upload MRI Image</CardTitle>
                <CardDescription>
                  Select an MRI scan for initial tumor detection
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    MRI Scan Image
                  </label>
                  <div 
                    className="border-2 border-dashed border-medical-teal/30 rounded-lg p-8 text-center hover:border-medical-teal/60 transition-colors cursor-pointer"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <FileImage className="w-12 h-12 text-medical-teal/40 mx-auto mb-4" />
                    <p className="text-gray-600 mb-2">Click to upload or drag and drop</p>
                    <p className="text-sm text-gray-400">PNG, JPG, DICOM supported</p>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*,.dcm"
                    onChange={handleImageUpload}
                    className="hidden"
                    aria-label="Upload MRI scan image"
                  />
                </div>

                {selectedImage && (
                  <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <p className="text-green-800">✓ Image uploaded: {selectedImage.name}</p>
                  </div>
                )}

                {selectedImage && (
                  <Button
                    onClick={handleInitialAnalysis}
                    disabled={isAnalyzing}
                    className="w-full medical-gradient text-white"
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Start Detection Analysis'}
                  </Button>
                )}

                {isAnalyzing && (
                  <div className="space-y-2">
                    <Progress value={progress} />
                    <p className="text-sm text-center text-gray-600">{progress}% Complete</p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Step 2a: Glioma Segmentation */}
          {currentStep === 'segmentation' && detectionResult && isGlioma() && (
            <Card className="medical-card">
              <CardHeader>
                <CardTitle>Step 2: Glioma Segmentation</CardTitle>
                <CardDescription>
                  Upload NIfTI file for 3D segmentation analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <h3 className="font-semibold text-blue-800">Glioma Detected</h3>
                  <p className="text-blue-700">Type: {detectionResult.tumor_type}</p>
                  <p className="text-sm text-blue-600">A NIfTI file is required for precise segmentation analysis</p>
                </div>

                {!isSegmenting ? (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        NIfTI File (.nii or .nii.gz)
                      </label>
                      <div 
                        className="border-2 border-dashed border-medical-teal/30 rounded-lg p-8 text-center hover:border-medical-teal/60 transition-colors cursor-pointer"
                        onClick={() => niftiInputRef.current?.click()}
                      >
                        <Upload className="w-12 h-12 text-medical-teal/40 mx-auto mb-4" />
                        <p className="text-gray-600 mb-2">Upload NIfTI file for segmentation</p>
                        <p className="text-sm text-gray-400">.nii or .nii.gz files</p>
                      </div>
                      <input
                        ref={niftiInputRef}
                        type="file"
                        accept=".nii,.nii.gz"
                        onChange={handleNiftiUpload}
                        className="hidden"
                        aria-label="Upload NIfTI file for segmentation"
                      />
                    </div>

                    {selectedNifti && (
                      <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                        <p className="text-green-800">✓ NIfTI file uploaded: {selectedNifti.name}</p>
                      </div>
                    )}

                    {selectedNifti && (
                      <Button
                        onClick={handleSegmentation}
                        disabled={isSegmenting}
                        className="w-full medical-gradient text-white"
                      >
                        Start Segmentation Analysis
                      </Button>
                    )}
                  </>
                ) : (
                  <div className="space-y-4">
                    <div className="p-4 bg-medical-teal/10 border border-medical-teal/30 rounded-lg">
                      <div className="flex items-center space-x-3 mb-3">
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-medical-teal"></div>
                        <h3 className="font-semibold text-medical-teal">Processing Segmentation...</h3>
                      </div>
                      <Progress value={progress} className="mb-2" />
                      <p className="text-sm text-center text-gray-600">
                        Analyzing 3D brain structure and tumor boundaries...
                      </p>
                      <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-center">
                        <div className={`p-2 rounded ${progress > 33 ? 'bg-green-100 text-green-800' : 'bg-gray-100'}`}>
                          ✓ Preprocessing
                        </div>
                        <div className={`p-2 rounded ${progress > 66 ? 'bg-green-100 text-green-800' : 'bg-gray-100'}`}>
                          🧠 U-Net Analysis
                        </div>
                        <div className={`p-2 rounded ${progress > 90 ? 'bg-green-100 text-green-800' : 'bg-gray-100'}`}>
                          📊 Feature Extraction
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Step 2b: Manual Feature Entry */}
          {currentStep === 'manual' && detectionResult && !isGlioma() && (
            <Card className="medical-card">
              <CardHeader>
                <CardTitle>Step 2: Manual Feature Input</CardTitle>
                <CardDescription>
                  Enter tumor features manually for analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <h3 className="font-semibold text-yellow-800">Non-Glioma Tumor Detected</h3>
                  <p className="text-yellow-700">Type: {detectionResult.tumor_type}</p>
                  <p className="text-sm text-yellow-600">Please enter the tumor features manually</p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {(Object.keys(manualFeatures) as Array<keyof ManualFeatures>).map((key) => (
                    <div key={key} className="space-y-2">
                      <Label 
                        htmlFor={key} 
                        className={isFieldInError(key) ? 'text-red-600 font-semibold' : ''}
                      >
                        {getFieldLabel(key)}
                        {isFieldInError(key) && <span className="text-red-500 ml-1">*</span>}
                      </Label>
                      <Input
                        id={key}
                        type="number"
                        step="0.01"
                        placeholder="Enter a value"
                        value={manualFeatures[key] === 0 ? '' : manualFeatures[key]}
                        onChange={(e) => handleFeatureChange(key, e.target.value)}
                        className={`${isFieldInError(key) 
                          ? 'border-red-500 border-2 focus:border-red-600 focus:ring-red-200' 
                          : 'border-gray-300 focus:border-medical-teal focus:ring-medical-teal/20'
                        } transition-colors`}
                      />
                      {isFieldInError(key) && (
                        <p className="text-red-500 text-xs mt-1">This field is required</p>
                      )}
                    </div>
                  ))}
                </div>

                {showValidation && fieldErrors.size > 0 && (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <AlertCircle className="w-5 h-5 text-red-600" />
                      <h4 className="font-semibold text-red-800">Required Fields Missing</h4>
                    </div>
                    <p className="text-red-700 text-sm">
                      Please fill in all fields with valid values (greater than 0) to proceed with the analysis.
                    </p>
                    <p className="text-red-600 text-xs mt-1">
                      {fieldErrors.size} field(s) to complete
                    </p>
                  </div>
                )}

                <Button
                  onClick={handleManualSubmission}
                  disabled={isAnalyzing}
                  className="w-full medical-gradient text-white hover:opacity-90 transition-opacity"
                >
                  {isAnalyzing ? 'Processing...' : 'Start AI Analysis'}
                </Button>

                {isAnalyzing && (
                  <div className="space-y-2">
                    <Progress value={progress} />
                    <p className="text-sm text-center text-gray-600">Analyzing manual features...</p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Results Section */}
          <Card className="medical-card">
            <CardHeader>
              <CardTitle>Analysis Results</CardTitle>
              <CardDescription>
                {currentStep === 'detection' && 'Initial detection results will appear here'}
                {currentStep === 'segmentation' && 'Segmentation results will appear here'}
                {currentStep === 'manual' && 'Manual analysis results will appear here'}
                {currentStep === 'completed' && 'Complete analysis results'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {currentStep === 'detection' && !detectionResult && !isAnalyzing && (
                <div className="text-center py-8">
                  <AlertCircle className="w-12 h-12 text-medical-teal/40 mx-auto mb-4" />
                  <p className="text-gray-600">Upload an image to start the analysis</p>
                </div>
              )}

              {detectionResult && (
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <h3 className="font-semibold text-blue-800">Initial Detection</h3>
                    <p className="text-blue-700">Tumor Type: {detectionResult.tumor_type}</p>
                    <p className="text-blue-600">Confidence: {parseFloat((detectionResult.confidence * 100).toFixed(1)).toString()}%</p>
                  </div>

                  {isGlioma() && currentStep !== 'completed' && (
                    <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <p className="text-yellow-800">→ Glioma detected. Please upload NIfTI file for segmentation.</p>
                    </div>
                  )}

                  {!isGlioma() && currentStep !== 'completed' && (
                    <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <p className="text-yellow-800">→ Non-Glioma tumor detected. Please enter features manually.</p>
                    </div>
                  )}
                </div>
              )}

              {segmentationResult && (
                <div className="space-y-4">
                  <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <h3 className="font-semibold text-green-800">✅ Segmentation Complete</h3>
                    <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
                      <p className="text-green-700">Volume: <strong>{Math.round(segmentationResult.segmentation_info.volume)} mm³</strong></p>
                      <p className="text-green-700">Max Intensity: <strong>{Math.round(segmentationResult.segmentation_info.max_intensity)}</strong></p>
                      <p className="text-green-700">Mean Intensity: <strong>{Math.round(segmentationResult.segmentation_info.mean_intensity)}</strong></p>
                      <p className="text-green-700">Compactness: <strong>{parseFloat(segmentationResult.segmentation_info.compactness.toFixed(3)).toString()}</strong></p>
                    </div>
                  </div>

                  <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <h3 className="font-semibold text-blue-800">🧬 Extracted Features</h3>
                    <div className="grid grid-cols-2 gap-2 mt-2 text-xs">
                      <div className="space-y-1">
                        <p>Tumor Volume: <strong>{Math.round(segmentationResult.extracted_features.t1_3d_tumor_volume)}</strong></p>
                        <p>Max Intensity: <strong>{Math.round(segmentationResult.extracted_features.t1_3d_max_intensity)}</strong></p>
                        <p>Major Axis Length: <strong>{parseFloat(segmentationResult.extracted_features.t1_3d_major_axis_length.toFixed(2)).toString()}</strong></p>
                        <p>Area: <strong>{Math.round(segmentationResult.extracted_features.t1_3d_area)}</strong></p>
                        <p>Minor Axis Length: <strong>{parseFloat(segmentationResult.extracted_features.t1_3d_minor_axis_length.toFixed(2)).toString()}</strong></p>
                      </div>
                      <div className="space-y-1">
                        <p>Extent: <strong>{parseFloat(segmentationResult.extracted_features.t1_3d_extent.toFixed(3)).toString()}</strong></p>
                        <p>Surface/Volume Ratio: <strong>{parseFloat(segmentationResult.extracted_features.t1_3d_surface_to_volume_ratio.toFixed(4)).toString()}</strong></p>
                        <p>GLCM Contrast: <strong>{Math.round(segmentationResult.extracted_features.t1_3d_glcm_contrast)}</strong></p>
                        <p>Mean Intensity: <strong>{Math.round(segmentationResult.extracted_features.t1_3d_mean_intensity)}</strong></p>
                        <p>2D Area Median: <strong>{Math.round(segmentationResult.extracted_features.t1_2d_area_median)}</strong></p>
                      </div>
                    </div>
                  </div>

                  <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                    <h3 className="font-semibold text-purple-800">📅 Survival Prediction</h3>
                    <div className="mt-2">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-purple-700">Estimated Survival:</span>
                        <span className="font-bold text-lg text-purple-900">{Math.round(segmentationResult.survival_prediction.survival_days)} days</span>
                      </div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-purple-700">Risk Category:</span>
                        <span className={`font-semibold px-2 py-1 rounded text-sm ${
                          segmentationResult.survival_prediction.risk_category === 'High Risk' 
                            ? 'bg-red-100 text-red-800'
                            : segmentationResult.survival_prediction.risk_category === 'Medium Risk'
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-green-100 text-green-800'
                        }`}>
                          {segmentationResult.survival_prediction.risk_category}
                        </span>
                      </div>
                    </div>
                  </div>

                  {segmentationResult.segmentation_image_url && (
                    <div className="p-4 bg-teal-50 border border-teal-200 rounded-lg">
                      <h3 className="font-semibold text-teal-800 mb-2">🎯 Segmentation Image</h3>
                      <img 
                        src={`http://localhost:5001${segmentationResult.segmentation_image_url}`}
                        alt="Brain tumor segmentation"
                        className="w-full max-w-md mx-auto rounded-lg border shadow-sm"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none';
                        }}
                      />
                    </div>
                  )}

                  <div className="p-3 bg-blue-100 border border-blue-300 rounded-lg">
                    <p className="text-xs text-blue-800">
                      <strong>Model Info:</strong> {segmentationResult.model_info.segmentation_model} + {segmentationResult.model_info.prediction_model} (v{segmentationResult.model_info.version})
                    </p>
                  </div>
                </div>
              )}

              {manualResult && (
                <div className="space-y-4">
                  <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <h3 className="font-semibold text-green-800">Manual Analysis Complete</h3>
                    <p className="text-green-700">Features processed successfully</p>
                  </div>

                  <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <h3 className="font-semibold text-blue-800">🧬 Manual Features</h3>
                    <div className="grid grid-cols-2 gap-2 mt-2 text-xs">
                      <div className="space-y-1">
                        <p>Tumor Volume: <strong>{Math.round(manualFeatures.t1_3d_tumor_volume)}</strong></p>
                        <p>Max Intensity: <strong>{Math.round(manualFeatures.t1_3d_max_intensity)}</strong></p>
                        <p>Major Axis Length: <strong>{parseFloat(manualFeatures.t1_3d_major_axis_length.toFixed(2)).toString()}</strong></p>
                        <p>Area: <strong>{Math.round(manualFeatures.t1_3d_area)}</strong></p>
                        <p>Minor Axis Length: <strong>{parseFloat(manualFeatures.t1_3d_minor_axis_length.toFixed(2)).toString()}</strong></p>
                      </div>
                      <div className="space-y-1">
                        <p>Extent: <strong>{parseFloat(manualFeatures.t1_3d_extent.toFixed(3)).toString()}</strong></p>
                        <p>Surface/Volume Ratio: <strong>{parseFloat(manualFeatures.t1_3d_surface_to_volume_ratio.toFixed(4)).toString()}</strong></p>
                        <p>GLCM Contrast: <strong>{Math.round(manualFeatures.t1_3d_glcm_contrast)}</strong></p>
                        <p>Mean Intensity: <strong>{Math.round(manualFeatures.t1_3d_mean_intensity)}</strong></p>
                        <p>2D Area Median: <strong>{Math.round(manualFeatures.t1_2d_area_median)}</strong></p>
                      </div>
                    </div>
                  </div>

                  <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                    <h3 className="font-semibold text-purple-800">📅 Survival Prediction</h3>
                    <div className="mt-2">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-purple-700">Estimated Survival:</span>
                        <span className="font-bold text-lg text-purple-900">{Math.round(manualResult.survival_days)} days</span>
                      </div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-purple-700">Analysis Method:</span>
                        <span className="font-semibold text-purple-800">{manualResult.analysis_method}</span>
                      </div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-purple-700">Status:</span>
                        <span className={`font-semibold px-2 py-1 rounded text-sm ${
                          manualResult.success 
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {manualResult.success ? 'Success' : 'Failed'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {currentStep === 'completed' && (
                <div className="space-y-4">
                  <Button
                    onClick={handleGenerateReportClick}
                    disabled={isGeneratingReport}
                    className="w-full bg-green-600 hover:bg-green-700 text-white"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    {isGeneratingReport ? 'Generating Report...' : 'Generate Complete Report'}
                  </Button>

                  <Button
                    onClick={handleStartOver}
                    variant="outline"
                    className="w-full"
                  >
                    Start New Analysis
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Disclaimer Modal */}
      {showDisclaimer && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="max-w-2xl w-full bg-white rounded-lg shadow-xl">
            <div className="p-6">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-10 h-10 bg-orange-100 rounded-full flex items-center justify-center">
                  <AlertCircle className="w-5 h-5 text-orange-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">Important Disclaimer</h3>
              </div>
              
              <div className="space-y-4 text-gray-600">
                <p>
                  <strong>This AI brain tumor analysis system is for research and educational purposes only.</strong>
                </p>
                
                <div className="space-y-2">
                  <p>• The results shown are <strong>computational predictions</strong> based on AI models</p>
                  <p>• This tool is <strong>not a substitute</strong> for professional medical diagnosis or treatment</p>
                  <p>• Always consult with qualified neuro-oncologists or neurosurgeons for accurate diagnosis</p>
                  <p>• Survival predictions are statistical estimates and may not reflect individual patient outcomes</p>
                  <p>• Treatment decisions should never be based solely on AI-generated results</p>
                  <p>• Segmentation results require validation by experienced radiologists</p>
                </div>
                
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <p className="text-red-800 text-sm">
                    <strong>URGENT:</strong> If this analysis suggests concerning results, please immediately consult with a qualified neuro-oncologist or neurosurgeon for proper evaluation and treatment planning.
                  </p>
                </div>
              </div>
              
              <div className="flex justify-end mt-6">
                <Button
                  onClick={() => setShowDisclaimer(false)}
                  className="bg-orange-600 hover:bg-orange-700 text-white"
                >
                  I Understand
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Patient Name Input Modal */}
      <PatientNameModal
        isOpen={showPatientNameModal}
        onClose={handlePatientNameCancel}
        onConfirm={handlePatientNameConfirm}
        isGenerating={isGeneratingReport}
      />
    </div>
  );
};

export default BrainTumor;