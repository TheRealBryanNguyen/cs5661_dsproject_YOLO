/*=============================================================================
 * Upload Component - Handles file selection and preview
 * 
 * This component provides a drag and drop interface for image uploading
 * and shows a preview of the selected image.
 *============================================================================*/

import { useRef } from 'react';
import { Box, Typography, Paper } from '@mui/material';

const UploadComponent = ({ onFileSelect, previewUrl }) => {
  const fileInputRef = useRef(null);

  /*======================= EVENT HANDLERS =======================*/
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onFileSelect(file);
      }
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  /*======================= RENDER FUNCTION =======================*/
  return (
    <Paper 
      elevation={3} 
      sx={{ p: 3, borderRadius: 2, mb: 4 }}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={handleUploadClick}
      className="upload-area"
    >
      <input
        id="file-input"
        type="file"
        ref={fileInputRef}
        accept=".jpg,.jpeg,.png"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      
      {previewUrl ? (
        <Box sx={{ position: 'relative', width: '100%', textAlign: 'center' }}>
          <img 
            src={previewUrl} 
            alt="Preview" 
            style={{ maxWidth: '100%', maxHeight: '300px', borderRadius: '8px' }}
          />
          <Box 
            sx={{ 
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'rgba(0,0,0,0.5)',
              color: 'white',
              opacity: 0,
              transition: '0.3s',
              borderRadius: '8px',
              '&:hover': { opacity: 1 }
            }}
          >
            <Typography>Click to replace</Typography>
          </Box>
        </Box>
      ) : (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="h6" gutterBottom>
            Drop an image here or click to browse
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Upload any image to detect if it was created by AI or a human
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default UploadComponent;