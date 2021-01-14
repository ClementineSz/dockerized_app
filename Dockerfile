FROM python:3.8-buster


WORKDIR /opt
ADD / /opt
RUN pip install -r requirements.txt



ENTRYPOINT ["python", "-u", "/opt/inf_file.py"]