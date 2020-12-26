FROM python:3.8-slim

COPY /Flask /root/Flask

RUN chown -R root:root /root/Flask

WORKDIR /root/Flask
RUN pip3 install -r requirements.txt

ENV F_APP run.py

RUN chmod +x run.py
CMD ["python3", "run.py"]

