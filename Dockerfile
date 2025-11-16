FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    wget unzip \
    libxext6 libxrender1 libxt6 libxcomposite1 libglib2.0-0 \
    libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pobierz MATLAB Runtime
RUN wget https://ssd.mathworks.com/supportfiles/downloads/R2024b/Release/0/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2024b_glnxa64.zip \
    && mkdir /mcr_install \
    && unzip -q MATLAB_Runtime_R2024b_glnxa64.zip -d /mcr_install \
    && rm MATLAB_Runtime_R2024b_glnxa64.zip

# Zainstaluj
RUN /mcr_install/install -mode silent -agreeToLicense yes -destinationFolder /opt/mcr || \
    (echo "INSTALL FAILED" && ls -R /tmp && cat /tmp/mathworks*/*.log && exit 1)

RUN apt-get install libglew-dev

WORKDIR /matlab

CMD ["./generate_maps_and_psf"]
