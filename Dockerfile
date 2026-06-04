FROM python:3.11

# Mengatur user non-root untuk Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user

# Mengatur path agar bin dari pip install terbaca
ENV PATH="/home/user/.local/bin:$PATH"
ENV USE_TF="0"

WORKDIR /home/user/app

# Copy requirements lebih dulu agar build Docker lebih cepat
COPY --chown=user ./requirements.txt /home/user/app/requirements.txt

# Install library
RUN pip install --no-cache-dir --upgrade -r /home/user/app/requirements.txt

# Copy semua file project dengan hak milik user
COPY --chown=user . /home/user/app

# Jalankan uvicorn di port 7860 (Standar HF Spaces)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
