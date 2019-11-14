FROM python:3.6.5-slim-stretch

EXPOSE 8080

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN pip3 install Flask===1.0.3 gunicorn===19.9.0 pandas===0.24.2 scikit-learn===0.21.2 stop-words===2018.7.23

COPY server.py /app/server.py
WORKDIR /app

CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "4", "--timeout", "300", "--access-logfile",  "-", "server:app"]