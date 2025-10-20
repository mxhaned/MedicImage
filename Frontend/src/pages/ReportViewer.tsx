
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { User } from "lucide-react";

const ReportViewer = () => {
  const reports = [
    {
      id: 1,
      type: "DermaScan",
      date: "2024-01-15",
      time: "14:30",
      status: "Completed",
      result: "Healthy Skin",
      confidence: 94,
      summary: "Normal skin condition detected with minor dryness. No immediate concerns identified.",
      findings: [
        { condition: "Normal Skin", probability: 94, severity: "Normal" },
        { condition: "Mild Dryness", probability: 12, severity: "Minor" }
      ],
      recommendations: [
        "Use daily moisturizer with SPF",
        "Stay hydrated",
        "Use gentle cleanser"
      ],
      image: "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=300&h=300&fit=crop"
    },
    {
      id: 2,
      type: "DermaScan",
      date: "2024-01-12",
      time: "09:15",
      status: "Completed",
      result: "Mild Acne Detected",
      confidence: 87,
      summary: "Mild inflammatory acne detected. Treatment recommendations provided.",
      findings: [
        { condition: "Inflammatory Acne", probability: 87, severity: "Mild" },
        { condition: "Sebaceous Hyperplasia", probability: 23, severity: "Minor" }
      ],
      recommendations: [
        "Use salicylic acid cleanser",
        "Apply benzoyl peroxide treatment",
        "Avoid over-washing"
      ],
      image: "https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?w=300&h=300&fit=crop"
    },
    {
      id: 3,
      type: "DermaScan",
      date: "2024-01-08",
      time: "16:45",
      status: "Completed",
      result: "Excellent Skin Health",
      confidence: 98,
      summary: "Optimal skin condition with excellent hydration and texture.",
      findings: [
        { condition: "Healthy Skin", probability: 98, severity: "Excellent" },
        { condition: "Good Hydration", probability: 95, severity: "Excellent" }
      ],
      recommendations: [
        "Continue current skincare routine",
        "Maintain UV protection",
        "Regular skin check-ups"
      ],
      image: "https://images.unsplash.com/photo-1515377905703-c4788e51af15?w=300&h=300&fit=crop"
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Completed":
        return "bg-green-100 text-green-800";
      case "Processing":
        return "bg-yellow-100 text-yellow-800";
      case "Failed":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getResultColor = (result: string) => {
    if (result.includes("Healthy") || result.includes("Excellent")) {
      return "text-green-600";
    } else if (result.includes("Mild")) {
      return "text-yellow-600";
    } else if (result.includes("Severe")) {
      return "text-red-600";
    }
    return "text-gray-600";
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
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Medical Reports
          </h1>
          <p className="text-gray-600">
            View and manage your AI-powered medical analysis reports
          </p>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="medical-card text-center animate-fade-in">
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-medical-teal mb-2">{reports.length}</div>
              <div className="text-sm text-gray-600">Total Reports</div>
            </CardContent>
          </Card>
          <Card className="medical-card text-center animate-fade-in" style={{ animationDelay: '0.1s' }}>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-green-600 mb-2">
                {reports.filter(r => r.status === "Completed").length}
              </div>
              <div className="text-sm text-gray-600">Completed</div>
            </CardContent>
          </Card>
          <Card className="medical-card text-center animate-fade-in" style={{ animationDelay: '0.2s' }}>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-medical-blue mb-2">93%</div>
              <div className="text-sm text-gray-600">Avg Confidence</div>
            </CardContent>
          </Card>
          <Card className="medical-card text-center animate-fade-in" style={{ animationDelay: '0.3s' }}>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-medical-teal mb-2">7</div>
              <div className="text-sm text-gray-600">Days Active</div>
            </CardContent>
          </Card>
        </div>

        {/* Reports List */}
        <div className="space-y-6">
          {reports.map((report, index) => (
            <Card key={report.id} className="medical-card animate-fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
              <CardHeader>
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="flex items-start gap-4">
                    <img 
                      src={report.image} 
                      alt="Scan result" 
                      className="w-16 h-16 rounded-lg object-cover"
                    />
                    <div>
                      <CardTitle className="text-xl text-gray-900">
                        {report.type} - Report #{report.id}
                      </CardTitle>
                      <CardDescription className="text-gray-600 mt-1">
                        {report.date} at {report.time}
                      </CardDescription>
                      <div className="flex items-center gap-2 mt-2">
                        <Badge className={getStatusColor(report.status)}>
                          {report.status}
                        </Badge>
                        <span className="text-sm text-gray-500">
                          Confidence: {report.confidence}%
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`font-semibold ${getResultColor(report.result)}`}>
                      {report.result}
                    </p>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 mb-4">{report.summary}</p>
                <div className="flex flex-col sm:flex-row gap-2">
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button 
                        variant="outline"
                        className="border-medical-teal text-medical-teal hover:bg-medical-teal hover:text-white"
                      >
                        View Details
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
                      <DialogHeader>
                        <DialogTitle className="text-xl text-gray-900">
                          {report.type} Report #{report.id}
                        </DialogTitle>
                        <DialogDescription className="text-gray-600">
                          Analysis completed on {report.date} at {report.time}
                        </DialogDescription>
                      </DialogHeader>
                      <div className="space-y-6">
                        {/* Image */}
                        <div className="text-center">
                          <img 
                            src={report.image} 
                            alt="Analysis result" 
                            className="w-48 h-48 rounded-lg object-cover mx-auto"
                          />
                        </div>
                        
                        {/* Overall Result */}
                        <div className="text-center p-4 bg-green-50 rounded-lg">
                          <h3 className={`text-lg font-semibold mb-2 ${getResultColor(report.result)}`}>
                            {report.result}
                          </h3>
                          <p className="text-gray-600">
                            AI Confidence: {report.confidence}%
                          </p>
                        </div>
                        
                        {/* Detailed Findings */}
                        <div>
                          <h4 className="font-semibold text-gray-900 mb-3">Detailed Analysis</h4>
                          <div className="space-y-2">
                            {report.findings.map((finding, idx) => (
                              <div key={idx} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                                <div>
                                  <span className="font-medium text-gray-900">{finding.condition}</span>
                                  <span className={`ml-2 px-2 py-1 text-xs rounded-full ${
                                    finding.severity === "Normal" || finding.severity === "Excellent" ? "bg-green-100 text-green-800" :
                                    finding.severity === "Minor" || finding.severity === "Mild" ? "bg-yellow-100 text-yellow-800" :
                                    "bg-red-100 text-red-800"
                                  }`}>
                                    {finding.severity}
                                  </span>
                                </div>
                                <span className="text-gray-600">{finding.probability}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                        
                        {/* Recommendations */}
                        <div>
                          <h4 className="font-semibold text-gray-900 mb-3">Recommendations</h4>
                          <ul className="space-y-2">
                            {report.recommendations.map((rec, idx) => (
                              <li key={idx} className="flex items-start">
                                <span className="text-medical-teal mr-2">•</span>
                                <span className="text-gray-600">{rec}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </DialogContent>
                  </Dialog>
                  <Button 
                    variant="ghost"
                    size="sm"
                    className="text-gray-600 hover:text-gray-800"
                  >
                    Download PDF
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Empty State */}
        {reports.length === 0 && (
          <Card className="medical-card text-center py-12 animate-fade-in">
            <CardContent>
              <div className="w-16 h-16 bg-medical-teal/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">📊</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No Reports Yet</h3>
              <p className="text-gray-600 mb-4">Start your first medical scan to see results here</p>
              <Link to="/derma-scan">
                <Button className="medical-gradient text-white glow-button">
                  Start DermaScan
                </Button>
              </Link>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
};

export default ReportViewer;
