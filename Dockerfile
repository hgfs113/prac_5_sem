FROM python:3.8-slim

COPY /Task3 /root/Task3

RUN chown -R root:root /root/Task3

WORKDIR /root/Task3
RUN pip3 install -r requirements.txt

ENV F_APP run.py

RUN chmod +x run.py
CMD ["python3", "run.py"]

