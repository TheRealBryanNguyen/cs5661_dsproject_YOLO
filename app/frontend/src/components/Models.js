/*=============================================================================
 * Model Availability Component - Shows loaded models with status indicators
 * 
 * This component displays which AI detection models are available in the system
 * and their current loading status.
 *============================================================================*/
import { Box, Typography, Grid} from '@mui/material';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';
import SyncIcon from '@mui/icons-material/Sync';

const Models = ({ availableModels, isLoadingModels }) => {
  if (isLoadingModels) {
    return (
      <Box display="flex" alignItems="center" justifyContent="center" py={2}>
        <SyncIcon className="spinning-icon" sx={{ mr: 1, color: 'primary.main' }} />
        <Typography variant="body2" color="textSecondary">
          Loading available models...
        </Typography>
      </Box>
    );
  }
  
  // Don't show this section if we have no models or only one model
  if (!availableModels || availableModels.length <= 0) {
    return null;
  }
  
  return (
    <Box className="available-models-info">
      <Typography variant="subtitle2" gutterBottom>
        Available Models: {availableModels.length}
      </Typography>
      <Grid container spacing={1}>
        {availableModels.map(model => (
          <Grid item xs={6} sm={4} key={model.id}>
            <Box className="model-chip available">
              <FiberManualRecordIcon 
                sx={{ 
                  fontSize: 10, 
                  color: 'success.main',
                  mr: 1
                }} 
              />
              <Typography variant="body2" noWrap>
                {model.name}
              </Typography>
            </Box>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default Models;