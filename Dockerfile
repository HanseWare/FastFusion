FROM cyb4black/litfusion-base:latest
# Now copy python files and config file, not needed while installation but needed while running
# so we have a small layer and build time is faster when onyl changing code
COPY *.py /app/
COPY model_config.json /app/model_config.json
ENV FASTFUSION_LOGLEVEL=info
# Make port 8000 available to the world outside this container
EXPOSE 8000
# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]