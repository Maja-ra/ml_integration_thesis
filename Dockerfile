# Use the official Python base image 3.13-slim 3.13.10-slim-trixie
FROM python:3.13.10-slim-bookworm

# Set the working directory inside the container
WORKDIR /thesis_repos

# Copy the requirements file to the working directory
COPY ./requirements_api.txt /thesis_repos/requirements_api.txt

# Copy the application code to the working directory 
COPY . .

# Install the Python dependencies # RUN pip install --upgrade .
RUN pip install --no-cache-dir --upgrade -r /thesis_repos/requirements_api.txt

EXPOSE 8000
# Run the FastAPI application using uvicorn server
#CMD ["uvicorn", "app.api:app","--host", "0.0.0.0", "--port", "8000"]
CMD ["fastapi", "run", "app/api.py", "--port", "8000"]