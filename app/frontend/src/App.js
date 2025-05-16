/*=============================================================================
 * AI vs Human Image Detector - Main Application Component
 * 
 * This component serves as the entry point for the application and handles:
 * - Image file selection and upload
 * - Communication with the backend API
 * - Displaying detection results
 * - Managing application state and UI
 *============================================================================*/

import { useState, useEffect } from 'react';
import { Container, Typography, Box, Paper, CircularProgress } from '@mui/material';
import UploadComponent from './components/UploadComponent';
import Models from './components/Models';
import ResultsComponent from './components/ResultsComponent';
import './App.css';
/*=============================================================================
 * AI vs Human Image Detector - Main Application Component
 * 
 * This component serves as the entry point for the application and handles:
 * - Image file selection and upload
 * - Communication with the backend API
 * - Displaying detection results
 * - Managing application state and UI
 *============================================================================*/

function App() {
  /*======================= STATE MANAGEMENT =======================*/
// eslint-disable-next-line
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  
  /*======================= LIFECYCLE HOOKS =======================*/
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  /*======================= API FUNCTIONS =======================*/
  const fetchAvailableModels = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/models');
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      
      const data = await response.json();
      const availableModelsOnly = data.filter(model => model.available);
      setAvailableModels(availableModelsOnly);
    } catch (err) {
      console.error('Error fetching available models:', err);
      setError('Unable to fetch available models');
      setAvailableModels([
        { id: 'vit', name: 'Vision Transformer (ViT)', description: 'Base model', available: true }
      ]);
    }
  };
  
  const handleFileSelect = (file) => {
    if (!file) return;
    
    setSelectedFile(file);
    
    const fileReader = new FileReader();
    fileReader.onload = () => {
      setPreviewUrl(fileReader.result);
    };
    fileReader.readAsDataURL(file);
    
    setResults(null);
    setError(null);
    
    analyzeImage(file);
  };
  
  const analyzeImage = async (file) => {
    if (!file) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:5001/api/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      
      const data = await response.json();
      
      const formattedData = {
        model_results: {},
        ensemble_result: null
      };
      
      if (data.model_results) {
        Object.entries(data.model_results).forEach(([id, result]) => {
          if (result && result.success === true) {
            formattedData.model_results[id] = result;
          }
        });
      } else if (data.success === true) {
        formattedData.model_results = { 'vit': data };
      }
      
      if (data.ensemble_result && data.ensemble_result.success === true) {
        formattedData.ensemble_result = data.ensemble_result;
      }
      
      if (Object.keys(formattedData.model_results).length === 0) {
        throw new Error('No models were able to successfully analyze the image');
      }
      
      setResults(formattedData);
    } catch (err) {
      console.error('Error analyzing image:', err);
      setError('An error occurred while analyzing the image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  /*======================= RENDERING HELPERS =======================*/
  const renderResult = (modelId, result) => {
    const isAI = result.prediction === 'AI-Generated';
    const confidence = Math.round(result.probability_ai > 0.5 ? result.probability_ai * 100 : (1 - result.probability_ai) * 100);
    
    return (
      <Paper 
        key={modelId}
        elevation={3} 
        className={isAI ? 'ai-result' : 'human-result'}
        sx={{ 
          p: 3, 
          my: 2,
          borderRadius: 2,
          borderLeft: isAI ? '5px solid #f44336' : '5px solid #4caf50'
        }}
      >
        <Typography variant="h5" gutterBottom>
          {result.prediction}
        </Typography>
        <Typography variant="body1">
          Confidence: {confidence}%
        </Typography>
        <Box 
          className="confidence-bar" 
          sx={{ 
            mt: 2,
            height: 10, 
            width: '100%', 
            backgroundColor: '#e0e0e0',
            borderRadius: 5,
            overflow: 'hidden'
          }}
        >
          <Box 
            sx={{ 
              width: `${confidence}%`, 
              height: '100%', 
              backgroundColor: isAI ? '#f44336' : '#4caf50' 
            }} 
          />
        </Box>
      </Paper>
    );
  };

  /*======================= MAIN RENDER FUNCTION =======================*/
  return (
    <Container maxWidth="sm" className="app-container">
      <Typography variant="h4" component="h1" align="center" gutterBottom>
        AI vs Human Image Detector
      </Typography>

      <UploadComponent onFileSelect={handleFileSelect} previewUrl={previewUrl} />
      
      <Models 
        availableModels={availableModels} 
        isLoadingModels={availableModels.length === 0} 
      />

      {isLoading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Paper sx={{ p: 2, bgcolor: '#ffebee', color: '#c62828', borderRadius: 2, my: 2 }}>
          <Typography>{error}</Typography>
        </Paper>
      )}

      {results && !isLoading && (
        <Box>
          <Typography variant="h5" gutterBottom align="center">
            Results
          </Typography>
          
          <ResultsComponent results={results} />
          
          {results.ensemble_result && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Ensemble Prediction
              </Typography>
              {renderResult('ensemble', results.ensemble_result)}
            </Box>
          )}
          
          {Object.entries(results.model_results).map(([modelId, result]) => {
            const modelInfo = availableModels.find(m => m.id === modelId) || { name: modelId };
            return (
              <Box key={modelId} sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  {modelInfo.name}
                </Typography>
                {renderResult(modelId, result)}
              </Box>
            );
          })}
        </Box>
      )}
    </Container>
  );
}

export default App;