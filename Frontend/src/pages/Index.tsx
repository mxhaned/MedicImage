
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowDown, User } from "lucide-react";
import { useEffect, useState } from "react";

const Index = () => {
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-gray-50 via-white to-medical-teal/5">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-md border-b border-gray-200/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 medical-gradient rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">M</span>
              </div>
              <span className="text-xl font-bold text-gray-900">MedicImage</span>
            </div>
            <div className="flex items-center space-x-4">
              <Link to="/login">
                <Button variant="ghost" className="text-gray-600 hover:text-gray-900">
                  Login
                </Button>
              </Link>
              <Link to="/register">
                <Button className="medical-gradient text-white glow-button">
                  Get Started
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div 
            className="animate-fade-in"
            style={{ transform: `translateY(${scrollY * 0.1}px)` }}
          >
            <h1 className="text-5xl md:text-7xl font-bold text-gray-900 mb-6">
              AI-Powered
              <span className="block bg-gradient-to-r from-medical-teal to-medical-blue bg-clip-text text-transparent">
                Medical Imaging
              </span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
              Revolutionize healthcare with intelligent image analysis. Get instant, 
              accurate medical insights powered by advanced AI technology.
            </p>
          </div>

          {/* Floating Medical Icons */}
          <div className="relative mt-16">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-64 h-64 medical-gradient rounded-full opacity-10 animate-float"></div>
            </div>
            <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 gap-8 mt-8 max-w-2xl mx-auto">
              <div className="medical-card p-6 animate-fade-in" style={{ animationDelay: '0.2s' }}>
                <div className="w-12 h-12 medical-gradient rounded-lg flex items-center justify-center mb-4 mx-auto">
                  <span className="text-white font-bold">D</span>
                </div>
                <Link to="/derma-scan" className="hover:opacity-75 transition-opacity">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">DermaScan</h3>
                </Link>
                <p className="text-gray-600">Advanced skin analysis with AI-powered recommendations</p>
              </div>
              <div className="medical-card p-6 animate-fade-in" style={{ animationDelay: '0.4s' }}>
                <div className="w-12 h-12 medical-gradient rounded-lg flex items-center justify-center mb-4 mx-auto">
                  <span className="text-white font-bold">B</span>
                </div>
                <Link to="/brain-tumor" className="hover:opacity-75 transition-opacity">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Brain Tumor Analysis</h3>
                </Link>
                <p className="text-gray-600">Intelligent MRI brain tumor detection and classification</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Scroll Indicator */}
      <div className="text-center pb-8">
        <ArrowDown className="w-6 h-6 text-medical-teal mx-auto animate-bounce" />
      </div>

      {/* Features Section */}
      <section className="py-20 bg-gradient-to-r from-medical-teal/5 to-medical-blue/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Why Choose MedicImage?
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Experience the future of medical imaging with our cutting-edge AI technology
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                title: "Instant Analysis",
                description: "Get comprehensive medical insights in seconds, not hours",
                icon: "⚡"
              },
              {
                title: "Expert Accuracy",
                description: "AI trained on millions of medical images for precise results",
                icon: "🎯"
              },
              {
                title: "Detailed Reports",
                description: "Comprehensive, easy-to-understand diagnostic reports",
                icon: "📊"
              },
              {
                title: "Secure & Private",
                description: "Your medical data is encrypted and completely confidential",
                icon: "🔒"
              },
              {
                title: "24/7 Available",
                description: "Access medical imaging analysis anytime, anywhere",
                icon: "🌐"
              },
              {
                title: "Continuous Learning",
                description: "Our AI improves with every scan for better accuracy",
                icon: "🧠"
              }
            ].map((feature, index) => (
              <Card key={index} className="medical-card animate-fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
                <CardHeader className="text-center">
                  <div className="text-4xl mb-4">{feature.icon}</div>
                  <CardTitle className="text-xl text-gray-900">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-center text-gray-600">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 medical-gradient text-white">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <h2 className="text-4xl font-bold mb-6">
            Ready to Transform Healthcare?
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Join thousands of healthcare professionals already using MedicImage
          </p>
          <Link to="/register">
            <Button 
              size="lg" 
              variant="secondary" 
              className="bg-white text-medical-teal px-8 py-4 text-lg hover:bg-gray-100 glow-button"
            >
              Sign in
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-2 mb-4 md:mb-0">
              <div className="w-8 h-8 medical-gradient rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">M</span>
              </div>
              <span className="text-xl font-bold">MedicImage</span>
            </div>
            <div className="text-gray-400">
              © 2024 MedicImage. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
