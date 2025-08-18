# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /src

# Copy the requirements file into the container at /src
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /src
COPY . .

# Define the command to run the application
CMD ["python", "train.py"] 