
import { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { User, ArrowUp, FileImage, AlertCircle, CheckCircle, Info, FileText } from "lucide-react";
import ImageCropper from "@/components/ImageCropper";
import PatientNameModal from "@/components/PatientNameModal";
import apiService, { ClassificationResult } from "@/services/api";
import { toast } from "sonner";

const DermaScan = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [croppedImage, setCroppedImage] = useState<string | null>(null);
  const [showCropper, setShowCropper] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [progress, setProgress] = useState(0);
  const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [showDisclaimer, setShowDisclaimer] = useState(false);
  const [showPatientNameModal, setShowPatientNameModal] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check API connection on component mount
  useEffect(() => {
    const checkApiConnection = async () => {
      try {
        await apiService.healthCheck();
        setApiConnected(true);
      } catch (error) {
        console.error('API connection failed:', error);
        setApiConnected(false);
        toast.error('Backend service is not available. Please ensure the server is running.');
      }
    };
    
    checkApiConnection();
  }, []);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
        setCroppedImage(null);
        setAnalysisComplete(false);
        setClassificationResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleCropComplete = (croppedImageData: string) => {
    setCroppedImage(croppedImageData);
    setShowCropper(false);
  };

  const handleAnalyze = async () => {
    if (!croppedImage) {
      toast.error('Please crop the image first');
      return;
    }
    
    if (apiConnected === false) {
      toast.error('Backend service is not available');
      return;
    }
    
    setIsAnalyzing(true);
    setProgress(0);
    
    try {
      // Simulate progress while API call is in progress
      const progressInterval = setInterval(() => {
      setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Make API call to classify the image
      const result = await apiService.classifyImage(croppedImage);
      
      clearInterval(progressInterval);
      setProgress(100);
      
      setClassificationResult(result);
          setIsAnalyzing(false);
          setAnalysisComplete(true);
      
      toast.success('Analysis completed successfully!');
      
    } catch (error) {
      console.error('Analysis failed:', error);
      setIsAnalyzing(false);
      setProgress(0);
      toast.error(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleGenerateReport = () => {
    if (!classificationResult || !croppedImage) {
      toast.error('No analysis results available for report generation');
      return;
    }
    
    setShowPatientNameModal(true);
  };

  const handleGenerateReportWithPatientName = async (patientName: string) => {
    if (!classificationResult || !croppedImage) {
      toast.error('No analysis results available for report generation');
      return;
    }

    setShowPatientNameModal(false);
    setIsGeneratingReport(true);
    
    try {
      // Prepare analysis data for the report
      const analysisData = {
        predictions: classificationResult.predictions,
        primary_condition: classificationResult.primary_condition,
        confidence: classificationResult.confidence,
        recommendations: getAnalysisResults()?.recommendations || [],
        products: getProductRecommendations()
      };

      // Generate and download the report
      await apiService.downloadReport({
        image: croppedImage,
        analysis_data: analysisData,
        patient_name: patientName
      });

      toast.success('Medical report downloaded successfully!');
      
    } catch (error) {
      console.error('Report generation failed:', error);
      toast.error(`Report generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  // Generate analysis results from classification
  const getAnalysisResults = () => {
    if (!classificationResult) return null;
    
    const { predictions, primary_condition, confidence } = classificationResult;
    
    // Convert predictions to findings array - show all conditions with percentages
    const findings = Object.entries(predictions).map(([condition, probability]) => ({
      condition,
      probability: Math.round(probability * 100),
      severity: probability >= 0.7 ? "High" : 
               probability >= 0.4 ? "Medium" : "Low"
    }));
    
    // Sort by probability (highest first)
    findings.sort((a, b) => b.probability - a.probability);
    
    // Generate recommendations based on primary condition
    const recommendations = getRecommendations(primary_condition);
    
    return {
      overall: `${primary_condition} Detected`,
      confidence: Math.round(confidence * 100),
      findings,
      recommendations,
      meets_threshold: confidence >= 0.7
    };
  };

  const getRecommendations = (condition: string): string[] => {
    const recommendations: Record<string, string[]> = {
      'Acne': [
        "Use a gentle cleanser twice daily",
        "Apply benzoyl peroxide or salicylic acid treatment",
        "Avoid touching your face frequently",
        "Consider seeing a dermatologist for severe cases"
      ],
      'Actinic Keratosis': [
        "Protect skin from UV radiation with broad-spectrum sunscreen",
        "Schedule a dermatologist appointment for evaluation",
        "Monitor for changes in size, color, or texture",
        "Consider cryotherapy or other treatment options"
      ],
      'Basal Cell Carcinoma': [
        "Immediately consult a dermatologist",
        "Protect the area from sun exposure",
        "Document any changes in size or appearance",
        "Consider surgical removal or other treatments"
      ],
      'Eczemaa': [
        "Use fragrance-free moisturizers regularly",
        "Avoid hot showers and harsh soaps",
        "Apply prescribed topical corticosteroids if needed",
        "Identify and avoid triggers"
      ],
      'Rosacea': [
        "Use gentle, non-irritating skincare products",
        "Avoid triggers like spicy foods and alcohol",
        "Protect skin from sun and wind exposure",
        "Consider prescription treatments from a dermatologist"
      ]
    };
    
    return recommendations[condition] || [
      "Consult with a healthcare professional",
      "Monitor for any changes in symptoms",
      "Maintain good skincare practices"
    ];
  };

  // Product recommendations based on condition
  const getProductRecommendations = () => {
    if (!classificationResult) return [];
    
    const { primary_condition } = classificationResult;
    
    const productRecommendations: Record<string, Array<{
      name: string;
      brand: string;
      rating: number;
      price: string;
    }>> = {
      'Acne': [
        {
          name: "Benzoyl Peroxide Cleanser",
          brand: "CeraVe",
          rating: 4.7,
          price: "$14.99"
        },
        {
          name: "Salicylic Acid Treatment",
          brand: "The Ordinary",
          rating: 4.5,
          price: "$12.50"
        },
        {
          name: "Oil-Free Moisturizer",
          brand: "Neutrogena",
          rating: 4.6,
          price: "$16.99"
        }
      ],
      'Actinic Keratosis': [
        {
          name: "Broad-Spectrum Sunscreen SPF 50",
          brand: "La Roche-Posay",
          rating: 4.8,
          price: "$24.99"
        },
        {
          name: "Vitamin C Serum",
          brand: "Skinceuticals",
          rating: 4.9,
          price: "$169.00"
        },
        {
          name: "Retinol Treatment",
          brand: "Differin",
          rating: 4.4,
          price: "$29.99"
        }
      ],
      'Basal Cell Carcinoma': [
        {
          name: "Medical Grade Sunscreen",
          brand: "EltaMD",
          rating: 4.9,
          price: "$35.00"
        },
        {
          name: "Antioxidant Serum",
          brand: "SkinCeuticals",
          rating: 4.8,
          price: "$165.00"
        },
        {
          name: "Protective Clothing",
          brand: "Coolibar",
          rating: 4.6,
          price: "$45.00"
        }
      ],
      'Eczemaa': [
        {
          name: "Fragrance-Free Moisturizer",
          brand: "Vanicream",
          rating: 4.7,
          price: "$18.99"
        },
        {
          name: "Gentle Cleanser",
          brand: "Cetaphil",
          rating: 4.5,
          price: "$12.99"
        },
        {
          name: "Colloidal Oatmeal Bath",
          brand: "Aveeno",
          rating: 4.6,
          price: "$8.99"
        }
      ],
      'Rosacea': [
        {
          name: "Gentle Cleanser",
          brand: "CeraVe",
          rating: 4.8,
          price: "$15.99"
        },
        {
          name: "Azelaic Acid Treatment",
          brand: "The Ordinary",
          rating: 4.4,
          price: "$9.90"
        },
        {
          name: "Mineral Sunscreen",
          brand: "Colorescience",
          rating: 4.7,
          price: "$39.00"
        }
      ]
    };
    
    return productRecommendations[primary_condition] || [
    {
      name: "Hydrating Daily Moisturizer",
      brand: "CeraVe",
      rating: 4.8,
      price: "$12.99"
    },
    {
      name: "Gentle Foaming Cleanser",
      brand: "Neutrogena",
      rating: 4.6,
      price: "$8.99"
    },
    {
      name: "Vitamin C Serum",
      brand: "The Ordinary",
      rating: 4.7,
      price: "$15.50"
    }
  ];
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-gray-50 to-white">
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
            <div className="flex items-center space-x-4">
              <Link to="/dashboard">
                <Button variant="ghost" className="text-gray-600 hover:text-gray-900">
                  Dashboard
                </Button>
              </Link>
              <Button 
                variant="ghost" 
                size="sm"
                className="w-8 h-8 rounded-full bg-medical-teal/10 hover:bg-medical-teal/20"
              >
                <User className="w-4 h-4 text-medical-teal" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8 animate-fade-in">
          <div className="flex items-center justify-between">
            <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            DermaScan - AI Skin Analysis
          </h1>
          <p className="text-gray-600">
            Upload or capture an image for comprehensive skin analysis
          </p>
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

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <Card className="medical-card animate-fade-in">
            <CardHeader>
              <CardTitle className="text-xl text-gray-900">Upload Image</CardTitle>
              <CardDescription>
                Select a clear, well-lit image of the skin area you want to analyze
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {!selectedImage ? (
                  <div 
                    className="border-2 border-dashed border-medical-teal/30 rounded-lg p-8 text-center hover:border-medical-teal/60 transition-colors cursor-pointer"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <FileImage className="w-16 h-16 text-medical-teal/40 mx-auto mb-4" />
                    <p className="text-gray-600 mb-2">Click to upload or drag and drop</p>
                    <p className="text-sm text-gray-400">PNG, JPG up to 10MB</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                  <div className="relative">
                    <img 
                      src={selectedImage} 
                      alt="Selected for analysis" 
                      className="w-full h-64 object-cover rounded-lg"
                    />
                    <Button
                      variant="secondary"
                      size="sm"
                      className="absolute top-2 right-2"
                        onClick={() => {
                          setSelectedImage(null);
                          setCroppedImage(null);
                          setAnalysisComplete(false);
                          setClassificationResult(null);
                        }}
                    >
                      Change Image
                    </Button>
                    </div>
                    
                    {croppedImage && (
                      <div className="relative">
                        <p className="text-sm text-gray-600 mb-2">Cropped area:</p>
                        <img 
                          src={croppedImage} 
                          alt="Cropped for analysis" 
                          className="w-full h-32 object-cover rounded-lg border-2 border-medical-teal"
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          className="absolute top-2 right-2"
                          onClick={() => setCroppedImage(null)}
                        >
                          Re-crop
                        </Button>
                      </div>
                    )}
                    
                    {!croppedImage && (
                      <Button
                        onClick={() => setShowCropper(true)}
                        className="w-full medical-gradient text-white glow-button"
                      >
                        Crop Image for Analysis
                      </Button>
                    )}
                  </div>
                )}
                
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                  aria-label="Upload skin image for analysis"
                  title="Upload skin image for analysis"
                />
                
                <div className="flex gap-2">
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    variant="outline"
                    className="flex-1 border-medical-teal text-medical-teal hover:bg-medical-teal hover:text-white"
                  >
                    <ArrowUp className="w-4 h-4 mr-2" />
                    Upload Image
                  </Button>
                </div>
                
                {croppedImage && !isAnalyzing && !analysisComplete && (
                  <Button 
                    onClick={handleAnalyze}
                    className="w-full medical-gradient text-white glow-button"
                    size="lg"
                    disabled={apiConnected === false}
                  >
                    {apiConnected === false ? 'Backend Unavailable' : 'Start AI Analysis'}
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Analysis Results */}
          <Card className="medical-card animate-fade-in" style={{ animationDelay: '0.2s' }}>
            <CardHeader>
              <CardTitle className="text-xl text-gray-900">Analysis Results</CardTitle>
              <CardDescription>
                AI-powered skin analysis and recommendations
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!selectedImage && (
                <div className="text-center py-8">
                  <div className="w-16 h-16 bg-medical-teal/10 rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="text-2xl">🔬</span>
                  </div>
                  <p className="text-gray-600">Upload an image to begin analysis</p>
                </div>
              )}
              
              {isAnalyzing && (
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="w-16 h-16 bg-medical-teal/10 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse">
                      <span className="text-2xl">🧬</span>
                    </div>
                    <p className="text-gray-900 font-semibold mb-2">Analyzing your image...</p>
                    <p className="text-gray-600 text-sm mb-4">Our AI is examining skin patterns and conditions</p>
                  </div>
                  <Progress value={progress} className="w-full" />
                  <p className="text-center text-sm text-gray-600">{progress}% Complete</p>
                </div>
              )}
              
              {analysisComplete && (
                <div className="space-y-6">
                  {/* Overall Result */}
                  <div className="text-center p-4 rounded-lg bg-blue-50 border border-blue-200">
                    <h3 className="text-lg font-semibold mb-2 text-blue-800">
                      {getAnalysisResults()?.overall}
                    </h3>
                    <p className="text-blue-600">
                      Confidence: {getAnalysisResults()?.confidence}%
                    </p>
                  </div>
                  
                  {/* Detailed Findings */}
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-3">Condition Probabilities</h4>
                    <div className="space-y-2">
                      {getAnalysisResults()?.findings.map((finding, index) => (
                        <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center space-x-3">
                            <span className="font-medium text-gray-900">{finding.condition}</span>
                            <span className={`px-2 py-1 text-xs rounded-full ${
                              finding.severity === "High" ? "bg-red-100 text-red-800" :
                              finding.severity === "Medium" ? "bg-yellow-100 text-yellow-800" :
                              "bg-green-100 text-green-800"
                            }`}>
                              {finding.severity}
                            </span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Progress value={finding.probability} className="w-20 h-2" />
                            <span className="text-gray-600 font-medium min-w-[3rem]">{finding.probability}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  {/* Recommendations */}
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-3">Recommendations</h4>
                    <ul className="space-y-2">
                      {getAnalysisResults()?.recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start">
                          <span className="text-medical-teal mr-2">•</span>
                          <span className="text-gray-600">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Generate Report Button */}
                  <div className="pt-4 border-t border-gray-200">
                    <Button
                      onClick={handleGenerateReport}
                      disabled={isGeneratingReport}
                      className="w-full bg-green-600 hover:bg-green-700 text-white"
                      size="lg"
                    >
                      {isGeneratingReport ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                          Generating Report...
                        </>
                      ) : (
                        <>
                          <FileText className="w-4 h-4 mr-2" />
                          Generate Medical Report
                        </>
                      )}
                    </Button>
                    <p className="text-xs text-gray-500 text-center mt-2">
                      Download a professional PDF report with analysis results and recommendations
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Image Cropper Modal */}
        {showCropper && selectedImage && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="max-w-4xl w-full max-h-[90vh] overflow-y-auto">
              <ImageCropper
                imageSrc={selectedImage}
                onCropComplete={handleCropComplete}
                onCancel={() => setShowCropper(false)}
              />
            </div>
          </div>
        )}

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
                    <strong>This AI skin classifier is for educational and informational purposes only.</strong>
                  </p>
                  
                  <div className="space-y-2">
                    <p>• The results shown are <strong>approximations</strong> based on AI analysis</p>
                    <p>• This tool is <strong>not a substitute</strong> for professional medical diagnosis</p>
                    <p>• Always consult with a qualified dermatologist for accurate diagnosis</p>
                    <p>• The confidence percentages indicate the model's certainty, not medical accuracy</p>
                    <p>• Results may vary and should not be used for self-diagnosis</p>
                  </div>
                  
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <p className="text-blue-800 text-sm">
                      <strong>Medical Advice:</strong> If you have concerns about your skin, please schedule an appointment with a dermatologist for proper evaluation and treatment.
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

        {/* Product Recommendations */}
        {analysisComplete && (
          <div className="mt-12 animate-fade-in [animation-delay:0.4s]">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Recommended Products</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {getProductRecommendations().map((product, index) => (
                <Card key={index} className="medical-card hover:scale-105 transition-transform">
                  <CardHeader>
                    <CardTitle className="text-lg text-gray-900">{product.name}</CardTitle>
                    <CardDescription className="text-gray-600">{product.brand}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex justify-between items-center mb-4">
                      <div className="flex items-center">
                        <span className="text-yellow-400">★</span>
                        <span className="text-gray-600 ml-1">{product.rating}</span>
                      </div>
                      <span className="font-semibold text-medical-teal">{product.price}</span>
                    </div>
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="w-full border-medical-teal text-medical-teal hover:bg-medical-teal hover:text-white"
                    >
                      View Product
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Patient Name Modal */}
        <PatientNameModal
          isOpen={showPatientNameModal}
          onClose={() => setShowPatientNameModal(false)}
          onConfirm={handleGenerateReportWithPatientName}
        />
      </main>
    </div>
  );
};

export default DermaScan;
