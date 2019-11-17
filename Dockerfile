FROM python:3.8.0

RUN apt-get update --fix-missing && apt-get -y upgrade

RUN mkdir /app/
COPY . /app/
RUN python -m pip install /app/
RUN python -m pip install -r /app/requirements.txt

WORKDIR /app/
CMD ["python", "-m", "unittest", "discover"]
