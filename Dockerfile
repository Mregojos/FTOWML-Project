FROM python:slim
WORKDIR /app
COPY . .
VOLUME /app
RUN pip install -U -r requirements.txt -q
EXPOSE 8501
CMD ["streamlit", "run", "main.py"]
