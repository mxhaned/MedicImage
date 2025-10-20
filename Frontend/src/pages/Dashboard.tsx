
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { User, ArrowUp } from "lucide-react";

const Dashboard = () => {
  const scanModules = [
    {
      id: "derma-scan",
      title: "DermaScan",
      description: "AI-powered skin analysis and dermatology insights",
      icon: "🔬",
      color: "from-medical-teal to-medical-teal-dark",
      link: "/derma-scan",
      status: "Active"
    },
    {
      id: "brain-tumor",
      title: "Brain Tumor Analysis",
      description: "Advanced MRI brain tumor detection and classification",
      icon: "🧠",
      color: "from-medical-blue to-medical-blue-dark",
      link: "/brain-tumor",
      status: "Active"
    }
  ];

  const recentReports = [
    {
      id: 1,
      type: "DermaScan",
      date: "2024-01-15",
      status: "Completed",
      summary: "Healthy skin detected with minor dryness"
    },
    {
      id: 2,
      type: "DermaScan",
      date: "2024-01-12",
      status: "Completed", 
      summary: "Mild acne detected - recommendations provided"
    }
  ];

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
              <Link to="/reports">
                <Button variant="ghost" className="text-gray-600 hover:text-gray-900">
                  Reports
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
        {/* Welcome Section */}
        <div className="mb-8 animate-fade-in">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Welcome to Your Medical Dashboard
          </h1>
          <p className="text-gray-600">
            Choose a scan type to begin your AI-powered medical analysis
          </p>
        </div>

        {/* Scan Modules */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {scanModules.map((module, index) => (
            <Card 
              key={module.id} 
              className={`medical-card hover:scale-105 transition-all duration-300 animate-fade-in ${
                module.status === "Coming Soon" ? "opacity-75" : ""
              }`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className={`w-12 h-12 bg-gradient-to-r ${module.color} rounded-lg flex items-center justify-center text-2xl mb-4`}>
                    {module.icon}
                  </div>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    module.status === "Active" 
                      ? "bg-green-100 text-green-800" 
                      : "bg-gray-100 text-gray-600"
                  }`}>
                    {module.status}
                  </span>
                </div>
                <CardTitle className="text-xl text-gray-900">{module.title}</CardTitle>
                <CardDescription className="text-gray-600">
                  {module.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Link to={module.link}>
                  <Button 
                    className={`w-full ${
                      module.status === "Active"
                        ? "medical-gradient text-white glow-button"
                        : "bg-gray-200 text-gray-500 cursor-not-allowed"
                    }`}
                    disabled={module.status !== "Active"}
                  >
                    {module.status === "Active" ? "Start Scan" : "Coming Soon"}
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Recent Reports */}
        <div className="animate-fade-in" style={{ animationDelay: '0.4s' }}>
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-gray-900">Recent Reports</h2>
            <Link to="/reports">
              <Button variant="outline" className="border-medical-teal text-medical-teal hover:bg-medical-teal hover:text-white">
                View All Reports
              </Button>
            </Link>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {recentReports.map((report) => (
              <Card key={report.id} className="medical-card">
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle className="text-lg text-gray-900">{report.type}</CardTitle>
                      <CardDescription className="text-gray-600">
                        Report #{report.id} • {report.date}
                      </CardDescription>
                    </div>
                    <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                      {report.status}
                    </span>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600 mb-4">{report.summary}</p>
                  <Link to="/reports">
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="border-medical-teal text-medical-teal hover:bg-medical-teal hover:text-white"
                    >
                      View Details
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-4 gap-6 animate-fade-in" style={{ animationDelay: '0.6s' }}>
          <Card className="medical-card text-center">
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-medical-teal mb-2">12</div>
              <div className="text-sm text-gray-600">Total Scans</div>
            </CardContent>
          </Card>
          <Card className="medical-card text-center">
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-medical-blue mb-2">98%</div>
              <div className="text-sm text-gray-600">Accuracy Rate</div>
            </CardContent>
          </Card>
          <Card className="medical-card text-center">
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-green-600 mb-2">24/7</div>
              <div className="text-sm text-gray-600">Available</div>
            </CardContent>
          </Card>
          <Card className="medical-card text-center">
            <CardContent className="pt-6">
              <div className="flex items-center justify-center text-2xl font-bold text-medical-teal mb-2">
                <ArrowUp className="w-6 h-6 mr-1" />
                15%
              </div>
              <div className="text-sm text-gray-600">Improvement</div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
