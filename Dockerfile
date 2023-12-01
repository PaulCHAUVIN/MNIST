# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy only the files necessary for poetry install (for caching)
COPY pyproject.toml poetry.lock /app/

# Install the dependencies without the overhead of virtualenv
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction

# Copy the rest of the application code
COPY . /app/

# Specify the command to run when the container starts
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]