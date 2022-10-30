FROM python:3.11

WORKDIR /usr/src

RUN apt-get update && apt-get install -y \
    # cm-super \
    # texlive-fonts-extra \
    # texlive-lang-cjk \
    # dvipng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt

RUN { \
        echo 'export LS_OPTIONS="--color=auto"'; \
        echo 'alias ls="ls -F $LS_OPTIONS"'; \
        echo 'export PS1="\\n\[\\e[1;32m\]\u\[\\e[1;37m\]:\[\\e[1;34m\]\w\\n\[\\e[1;32m\]>\[\\e[1;37m\] "'; \
    } >> /root/.bashrc
