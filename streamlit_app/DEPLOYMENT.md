# Streamlit Cloud Deployment Guide

This guide covers deploying the Banner Agent v2 to Streamlit Cloud.

## Prerequisites

1. A Streamlit Cloud account (https://streamlit.io/cloud)
2. A GitHub repository with your code
3. OpenAI API key
4. Assistant ID from OpenAI

## Configuration Files

### 1. Secrets Configuration (`.streamlit/secrets.toml`)

**Important**: This file contains sensitive information and should NOT be committed to version control.

For Streamlit Cloud deployment:
1. Go to your app settings in Streamlit Cloud
2. Navigate to the "Secrets" section
3. Copy the contents of `.streamlit/secrets.toml` into the secrets editor
4. Save the secrets

### 2. App Configuration (`.streamlit/config.toml`)

This file contains non-sensitive configuration and can be committed to version control.

## Environment Variable Handling

The app uses a unified configuration system that:
- Reads from `st.secrets` when deployed on Streamlit Cloud
- Falls back to environment variables for local development
- Supports `.env` files for local development

### Key Configuration Variables

- `OPENAI_API_KEY`: Required for OpenAI API access
- `ASSISTANT_ID`: Required for OpenAI Assistant functionality
- `ENVIRONMENT`: Set to "production" for Streamlit Cloud
- `LOG_LEVEL`: Control logging verbosity
- Database and other service configurations

## Deployment Steps

1. **Prepare Your Repository**
   - Ensure all code is committed
   - Verify `.gitignore` includes `.streamlit/secrets.toml`
   - Push to GitHub

2. **Configure Streamlit Cloud**
   - Connect your GitHub repository
   - Select the main branch
   - Set the main file path: `streamlit_app/app.py`
   - Add secrets from `.streamlit/secrets.toml`

3. **Deploy**
   - Click "Deploy"
   - Monitor the deployment logs
   - Test the deployed application

## Local Development

For local development:
1. Copy `.env.example` to `.env`
2. Fill in your API keys and configuration
3. Run: `streamlit run streamlit_app/app.py`

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Verify secrets are properly configured in Streamlit Cloud
   - Check that the key names match exactly

2. **FastAPI Backend Issues**
   - The app runs FastAPI embedded within Streamlit
   - Check logs for startup errors

3. **File Upload Issues**
   - Verify `maxUploadSize` in config.toml
   - Check supported file types in configuration

### Debug Mode

To enable debug logging:
1. Set `LOG_LEVEL = "DEBUG"` in secrets
2. Check Streamlit Cloud logs for detailed information

## Security Best Practices

1. Never commit secrets.toml to version control
2. Rotate API keys regularly
3. Use environment-specific configurations
4. Monitor API usage and costs
5. Enable rate limiting in production

## Performance Optimization

1. The app uses caching for API responses
2. Configure cache TTL based on your needs
3. Monitor OpenAI API usage to control costs
4. Use appropriate model configurations for your use case