FROM hanseware/fastfusion-base:1.1.0
# Now copy python files and config file, not needed while installation but needed while running
# so we have a small layer and build time is faster when onyl changing code
COPY *.py /app/
COPY model_config.json /app/model_config.json
ENV FASTFUSION_LOGLEVEL=info
ENV FASTFUSION_HOST=0.0.0.0
ENV FASTFUSION_PORT=9999
# Make port 8000 available to the world outside this container
EXPOSE 9999
# Run the application
#CMD ["python", "/app/app.py"]
CMD uvicorn app:fastfusion_app --host $FASTFUSION_HOST --port $FASTFUSION_PORT