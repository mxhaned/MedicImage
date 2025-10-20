import React, { useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { User, AlertCircle } from 'lucide-react';

interface PatientNameModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (patientName: string) => void;
  isGenerating?: boolean;
}

const PatientNameModal: React.FC<PatientNameModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  isGenerating = false
}) => {
  const [patientName, setPatientName] = useState('');
  const [error, setError] = useState('');

  // Patient name validation
  const validatePatientName = (name: string): boolean => {
    // Remove leading and trailing spaces
    const trimmedName = name.trim();
    
    // Check if name is not empty
    if (trimmedName.length === 0) {
      setError('Patient name is required');
      return false;
    }
    
    // Check minimum length
    if (trimmedName.length < 2) {
      setError('Name must contain at least 2 characters');
      return false;
    }
    
    // Check maximum length
    if (trimmedName.length > 50) {
      setError('Name cannot exceed 50 characters');
      return false;
    }
    
    // Regex to allow only letters, spaces, hyphens and apostrophes
    // Excludes numbers and special symbols
    const validNameRegex = /^[a-zA-ZÀ-ÿ\u0100-\u017F\s\-'\.]+$/;
    
    if (!validNameRegex.test(trimmedName)) {
      setError('Name can only contain letters, spaces, hyphens and apostrophes');
      return false;
    }
    
    // Check that there are not only spaces or special characters
    const hasLetters = /[a-zA-ZÀ-ÿ\u0100-\u017F]/.test(trimmedName);
    if (!hasLetters) {
      setError('Name must contain at least one letter');
      return false;
    }
    
    setError('');
    return true;
  };

  const handleInputChange = (value: string) => {
    setPatientName(value);
    // Clear error when user types
    if (error) {
      setError('');
    }
  };

  const handleConfirm = () => {
    const trimmedName = patientName.trim();
    
    if (validatePatientName(trimmedName)) {
      onConfirm(trimmedName);
      setPatientName('');
      setError('');
    }
  };

  const handleClose = () => {
    if (!isGenerating) {
      setPatientName('');
      setError('');
      onClose();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isGenerating) {
      handleConfirm();
    }
    if (e.key === 'Escape' && !isGenerating) {
      handleClose();
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <User className="w-5 h-5 text-medical-teal" />
            Patient Name
          </DialogTitle>
          <DialogDescription>
            Please enter the patient's name for the medical report.
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="patient-name">Patient full name</Label>
          <Input
            id="patient-name"
            type="text"
            placeholder="e.g., John Doe"
            value={patientName}
            onChange={(e) => setPatientName(e.target.value)}
            className="mt-1"
            maxLength={50}
          />            {error && (
              <div className="flex items-center gap-2 text-red-600 text-sm">
                <AlertCircle className="w-4 h-4" />
                <span>{error}</span>
              </div>
            )}
            
            <p className="text-xs text-gray-500">
              Only letters, spaces, hyphens and apostrophes are allowed.
            </p>
          </div>
          
          <div className="flex gap-3 pt-4">
            <Button
              onClick={handleClose}
              variant="outline"
              className="flex-1"
              disabled={isGenerating}
            >
              Cancel
            </Button>
            <Button
              onClick={handleConfirm}
              className="flex-1 medical-gradient text-white"
              disabled={isGenerating || patientName.trim().length === 0}
            >
              {isGenerating ? (
                <div className="flex items-center gap-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Generating...
                </div>
              ) : (
                'Generate Report'
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default PatientNameModal;