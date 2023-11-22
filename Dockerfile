# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./src/requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./src /code/app
COPY ./data /code/data

WORKDIR /code/app

CMD ["uvicorn", "winequality:app", "--host", "0.0.0.0", "--port", "80"]
