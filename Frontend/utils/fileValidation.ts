export const validateImageFile = (file: File) => {
  const validImageTypes = ['image/jpeg', 'image/png', 'image/gif'];
  const validNiftiTypes = ['.nii', '.nii.gz'];
  const maxSize = 10 * 1024 * 1024; // 10MB

  // Check file size
  if (file.size > maxSize) {
    throw new Error('File size must be less than 10MB');
  }

  // Check file type
  const fileExtension = file.name.toLowerCase().split('.').pop();
  if (!validImageTypes.includes(file.type) && 
      !validNiftiTypes.includes(`.${fileExtension}`)) {
    throw new Error('Invalid file type. Supported formats: JPG, PNG, GIF, NIfTI');
  }

  return true;
};

export const isNiftiFile = (fileName: string): boolean => {
  const extension = fileName.toLowerCase().split('.').pop();
  return extension === 'nii' || (extension === 'gz' && fileName.toLowerCase().endsWith('.nii.gz'));
};