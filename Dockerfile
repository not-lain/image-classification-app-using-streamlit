# Specify a base image
FROM python:3.8


# Install any necessary dependencies
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

# Copy the files from your Streamlit app into the Docker image
COPY . /app

# Specify the command to run when the Docker container is started
CMD streamlit run app.py
