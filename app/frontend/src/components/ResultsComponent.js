/*=============================================================================
 * Results Component
 * 
 * This component creates a visual representation of the AI detection results 
 * with a prominent verdict display and confidence score.
 *============================================================================*/

import { Box, Typography } from '@mui/material';

const ResultsComponent = ({ results }) => {
  /*======================= INPUT VALIDATION =======================*/
  if (!results || (!results.ensemble_result && Object.keys(results.model_results).length === 0)) {
    return null;
  }
  
  /*======================= DATA PREPARATION =======================*/
  // Prioritize ensemble result if available
  const mainResult = results.ensemble_result || Object.values(results.model_results)[0];
  const aiProbability = mainResult.probability_ai;
  const isAI = aiProbability > 0.5;
  const confidence = Math.round(isAI ? aiProbability * 100 : (1 - aiProbability) * 100);
  
  // Check if there's disagreement between models
  const hasMixedResults = () => {
    if (!results.model_results || Object.keys(results.model_results).length <= 1) {
      return false;
    }
    
    const predictions = Object.values(results.model_results).map(result => result.prediction);
    const uniquePredictions = new Set(predictions);
    return uniquePredictions.size > 1;
  };
  
  const showMixedResults = hasMixedResults();
  
  /*======================= RENDERING =======================*/
  return (
    <Box 
      sx={{
        mt: 4, 
        mb: 3,
        textAlign: 'center',
        p: 3,
        borderRadius: 2,
        backgroundColor: showMixedResults ? 'rgba(255, 193, 7, 0.05)' : 
                          isAI ? 'rgba(244, 67, 54, 0.05)' : 'rgba(76, 175, 80, 0.05)',
        border: `1px solid ${showMixedResults ? 'rgba(255, 193, 7, 0.2)' : 
                  isAI ? 'rgba(244, 67, 54, 0.2)' : 'rgba(76, 175, 80, 0.2)'}`,
      }}
    >
      <Typography 
        variant="h3" 
        gutterBottom
        sx={{ 
          fontWeight: 'bold',
          color: showMixedResults ? '#ff9800' : isAI ? '#d32f2f' : '#2e7d32'
        }}
      >
        {showMixedResults ? 'MIXED RESULTS' : isAI ? 'AI-GENERATED' : 'HUMAN-CREATED'}
      </Typography>
      
      <Typography variant="h6" gutterBottom>
        {showMixedResults ? 'Ensemble ' : ''}Confidence: {confidence}%
      </Typography>
      
      <Box 
        className="confidence-bar" 
        sx={{ 
          mt: 2,
          mx: 'auto',
          height: 10, 
          width: '100%', 
          maxWidth: '500px',
          backgroundColor: '#e0e0e0',
          borderRadius: 5,
          overflow: 'hidden'
        }}
      >
        <Box 
          sx={{ 
            width: `${confidence}%`, 
            height: '100%', 
            backgroundColor: showMixedResults ? '#ff9800' : isAI ? '#f44336' : '#4caf50',
            transition: 'width 0.8s ease-in-out'
          }} 
        />
      </Box>
    </Box>
  );
};

export default ResultsComponent;