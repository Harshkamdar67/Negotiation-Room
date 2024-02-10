FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .


COPY models/certainty_prediction_model models/certainty_prediction_model

RUN python -c "from transformers import BertTokenizer, BertForSequenceClassification; \
BertTokenizer.from_pretrained('bert-base-uncased'); \
BertForSequenceClassification.from_pretrained('models/certainty_prediction_model');"

# Command to run the app
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "chat:app", "--bind", "0.0.0.0:8080"]
