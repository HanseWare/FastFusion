FROM cyb4black/litfusion-base:latest
# Now copy python files and config file, not needed while installation but needed while running
# so we have a small layer and build time is faster when onyl changing code
COPY *.py /app/
COPY model_config.json /app/model_config.json
ENV LITFUSION_LOGLEVEL=info
# Make port 8000 available to the world outside this container
EXPOSE 8000
CMD ["python", "/app/litfusion_api.py"]
