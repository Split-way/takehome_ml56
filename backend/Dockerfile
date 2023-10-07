FROM python:3.11-slim
WORKDIR /backend
COPY ./requirements.txt /backend/
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . /backend/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]