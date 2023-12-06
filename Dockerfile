FROM python:3.10.6-buster

WORKDIR /app

COPY ./ .
RUN pip install -r ./requirements-prod.txt

RUN apt-get update
RUN apt-get install direnv

EXPOSE 8000

ENV PATH="/.env:$PATH"

CMD uvicorn fast:app --host="0.0.0.0" --reload
