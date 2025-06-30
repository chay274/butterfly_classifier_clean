FROM python:3.12-slim

WORKDIR /app

COPY model/butterfly_model_v1.h5 /app/model/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]

