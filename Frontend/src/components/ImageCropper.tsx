import React, { useState, useRef } from 'react';
import ReactCrop, { Crop, PixelCrop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Crop as CropIcon, RotateCcw } from 'lucide-react';

interface ImageCropperProps {
  imageSrc: string;
  onCropComplete: (croppedImage: string) => void;
  onCancel: () => void;
}

const ImageCropper: React.FC<ImageCropperProps> = ({ imageSrc, onCropComplete, onCancel }) => {
  const [crop, setCrop] = useState<Crop>({
    unit: '%',
    width: 50,
    height: 50,
    x: 25,
    y: 25,
  });
  const [completedCrop, setCompletedCrop] = useState<PixelCrop>();
  const imgRef = useRef<HTMLImageElement>(null);

  const getCroppedImg = (image: HTMLImageElement, crop: PixelCrop): Promise<string> => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      throw new Error('No 2d context');
    }

    const scaleX = image.naturalWidth / image.width;
    const scaleY = image.naturalHeight / image.height;

    canvas.width = crop.width;
    canvas.height = crop.height;

    ctx.drawImage(
      image,
      crop.x * scaleX,
      crop.y * scaleY,
      crop.width * scaleX,
      crop.height * scaleY,
      0,
      0,
      crop.width,
      crop.height
    );

    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        if (blob) {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.readAsDataURL(blob);
        }
      }, 'image/jpeg', 0.9);
    });
  };

  const handleCropComplete = async () => {
    if (!completedCrop || !imgRef.current) return;

    try {
      const croppedImage = await getCroppedImg(imgRef.current, completedCrop);
      onCropComplete(croppedImage);
    } catch (error) {
      console.error('Error cropping image:', error);
    }
  };

  const resetCrop = () => {
    setCrop({
      unit: '%',
      width: 50,
      height: 50,
      x: 25,
      y: 25,
    });
    setCompletedCrop(undefined);
  };

  return (
    <Card className="medical-card">
      <CardHeader>
        <CardTitle className="text-xl text-gray-900 flex items-center gap-2">
          <CropIcon className="w-5 h-5" />
          Crop Image
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="relative max-w-full overflow-hidden rounded-lg">
            <ReactCrop
              crop={crop}
              onChange={(c) => setCrop(c)}
              onComplete={(c) => setCompletedCrop(c)}
              aspect={1}
              minWidth={100}
              minHeight={100}
            >
              <img
                ref={imgRef}
                src={imageSrc}
                alt="Crop preview"
                className="max-w-full h-auto"
              />
            </ReactCrop>
          </div>
          
          <div className="flex gap-2">
            <Button
              onClick={resetCrop}
              variant="outline"
              className="flex-1 border-medical-teal text-medical-teal hover:bg-medical-teal hover:text-white"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset Crop
            </Button>
            <Button
              onClick={handleCropComplete}
              disabled={!completedCrop}
              className="flex-1 medical-gradient text-white glow-button"
            >
              <CropIcon className="w-4 h-4 mr-2" />
              Apply Crop
            </Button>
            <Button
              onClick={onCancel}
              variant="outline"
              className="flex-1"
            >
              Cancel
            </Button>
          </div>
          
          <p className="text-sm text-gray-600 text-center">
            Drag the corners to select the skin area you want to analyze
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default ImageCropper; 